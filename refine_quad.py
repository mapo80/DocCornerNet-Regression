"""
Pixel-Level Quad Refinement Module for Document Corner Detection.

This module implements a post-processing refinement step that uses edge detection
and line fitting to improve corner localization from neural network predictions.

The refinement works by:
1. Warping the image using predicted corners to a canonical rectangle
2. Detecting edges in the warped image
3. Finding line segments along the document borders
4. Fitting 4 robust lines (2 vertical + 2 horizontal)
5. Computing intersections to get refined corners
6. Unwarping back to original image space

Fallback to original prediction when refinement fails or produces unreliable results.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List


@dataclass
class RefinementResult:
    """Result of quad refinement operation."""
    refined_quad: np.ndarray  # [4, 2] refined corners in pixel coords (TL, TR, BR, BL)
    original_quad: np.ndarray  # [4, 2] original input corners
    refine_ok: bool  # True if refinement was applied successfully
    reason: str  # Explanation of result (success or failure reason)

    # Diagnostic metrics
    num_lines_found: int = 0
    num_horizontal: int = 0
    num_vertical: int = 0
    edge_strength: float = 0.0
    fit_residual: float = 0.0
    area_ratio: float = 1.0  # refined_area / original_area


@dataclass
class RefinementConfig:
    """Configuration for quad refinement."""
    # Warp size constraints
    min_warp_size: int = 400
    max_warp_size: int = 1200
    warp_margin: int = 20  # pixels of margin around document in warp

    # Edge detection
    canny_low: int = 50
    canny_high: int = 150

    # Line detection
    use_lsd: bool = True  # Use LSD if available, else HoughLinesP
    hough_threshold: int = 50
    hough_min_length: int = 50
    hough_max_gap: int = 10

    # Line classification (angle thresholds in degrees)
    horizontal_angle_threshold: float = 20.0  # degrees from horizontal
    vertical_angle_threshold: float = 20.0  # degrees from vertical

    # Line selection
    min_lines_per_direction: int = 2  # Need at least 2 H and 2 V lines
    border_distance_threshold: float = 0.15  # fraction of warp size

    # Validation
    min_edge_strength: float = 10.0  # minimum mean edge magnitude
    area_ratio_min: float = 0.85
    area_ratio_max: float = 1.15
    max_corner_shift: float = 50.0  # max shift in pixels before fallback

    # Fallback
    fallback_on_any_failure: bool = True


def compute_quad_area(quad: np.ndarray) -> float:
    """Compute area of quadrilateral using Shoelace formula."""
    x = quad[:, 0]
    y = quad[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    return 0.5 * np.abs(np.sum(x * y_next - x_next * y))


def is_convex(quad: np.ndarray) -> bool:
    """Check if quadrilateral is convex using cross product signs."""
    n = len(quad)
    signs = []
    for i in range(n):
        p1 = quad[i]
        p2 = quad[(i + 1) % n]
        p3 = quad[(i + 2) % n]
        # Cross product of (p2-p1) x (p3-p2)
        cross = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])
        signs.append(cross)
    # All same sign means convex
    return all(s >= 0 for s in signs) or all(s <= 0 for s in signs)


def get_line_angle(line: np.ndarray) -> float:
    """Get angle of line segment in degrees [0, 180)."""
    x1, y1, x2, y2 = line
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    # Normalize to [0, 180)
    if angle < 0:
        angle += 180
    return angle


def classify_line(line: np.ndarray, config: RefinementConfig) -> str:
    """Classify line as 'horizontal', 'vertical', or 'other'."""
    angle = get_line_angle(line)

    # Horizontal: close to 0 or 180 degrees
    if angle < config.horizontal_angle_threshold or angle > (180 - config.horizontal_angle_threshold):
        return 'horizontal'

    # Vertical: close to 90 degrees
    if abs(angle - 90) < config.vertical_angle_threshold:
        return 'vertical'

    return 'other'


def line_to_params(line: np.ndarray) -> Tuple[float, float, float]:
    """Convert line segment to ax + by + c = 0 form, normalized."""
    x1, y1, x2, y2 = line
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    # Normalize
    norm = np.sqrt(a * a + b * b)
    if norm > 1e-8:
        a, b, c = a / norm, b / norm, c / norm
    return a, b, c


def line_intersection(line1_params: Tuple[float, float, float],
                      line2_params: Tuple[float, float, float]) -> Optional[np.ndarray]:
    """Find intersection of two lines in ax + by + c = 0 form."""
    a1, b1, c1 = line1_params
    a2, b2, c2 = line2_params

    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-8:
        return None  # Parallel lines

    x = (b1 * c2 - b2 * c1) / det
    y = (a2 * c1 - a1 * c2) / det
    return np.array([x, y])


def point_to_line_distance(point: np.ndarray, line_params: Tuple[float, float, float]) -> float:
    """Compute signed distance from point to line."""
    a, b, c = line_params
    return abs(a * point[0] + b * point[1] + c)


def select_border_lines(lines: List[np.ndarray],
                        warp_size: Tuple[int, int],
                        position: str,
                        config: RefinementConfig) -> List[np.ndarray]:
    """
    Select lines that are close to a specific border of the warp.

    Args:
        lines: List of line segments
        warp_size: (width, height) of warped image
        position: 'top', 'bottom', 'left', 'right'
        config: Refinement configuration

    Returns:
        Lines sorted by distance to border (closest first)
    """
    w, h = warp_size
    threshold = config.border_distance_threshold * max(w, h)

    scored_lines = []
    for line in lines:
        x1, y1, x2, y2 = line
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        if position == 'top':
            dist = mid_y
        elif position == 'bottom':
            dist = h - mid_y
        elif position == 'left':
            dist = mid_x
        elif position == 'right':
            dist = w - mid_x
        else:
            continue

        if dist < threshold:
            scored_lines.append((dist, line))

    # Sort by distance (closest to border first)
    scored_lines.sort(key=lambda x: x[0])
    return [line for _, line in scored_lines]


def fit_line_robust(lines: List[np.ndarray]) -> Optional[Tuple[float, float, float]]:
    """
    Fit a single line through multiple line segments using weighted average.

    Uses the midpoint and direction of each segment, weighted by length.
    """
    if not lines:
        return None

    # Collect all points from line segments
    points = []
    weights = []
    for line in lines:
        x1, y1, x2, y2 = line
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        points.append([x1, y1])
        points.append([x2, y2])
        weights.extend([length, length])

    points = np.array(points)
    weights = np.array(weights)
    weights = weights / weights.sum()

    # Weighted centroid
    cx = np.sum(points[:, 0] * weights)
    cy = np.sum(points[:, 1] * weights)

    # Weighted covariance for direction
    centered = points - np.array([cx, cy])
    cov = np.zeros((2, 2))
    for i, (p, w) in enumerate(zip(centered, weights)):
        cov += w * np.outer(p, p)

    # Principal direction
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    direction = eigenvectors[:, 1]  # Largest eigenvalue

    # Line params: direction is (dx, dy), normal is (-dy, dx)
    a = -direction[1]
    b = direction[0]
    c = -(a * cx + b * cy)

    # Normalize
    norm = np.sqrt(a * a + b * b)
    if norm > 1e-8:
        a, b, c = a / norm, b / norm, c / norm

    return a, b, c


def detect_lines_lsd(edges: np.ndarray) -> np.ndarray:
    """Detect lines using LSD (Line Segment Detector)."""
    # LSD works on grayscale, edges is already edge image
    # We need to create LSD detector
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines, _, _, _ = lsd.detect(edges)

    if lines is None:
        return np.array([])

    # Reshape from (N, 1, 4) to (N, 4)
    return lines.reshape(-1, 4)


def detect_lines_hough(edges: np.ndarray, config: RefinementConfig) -> np.ndarray:
    """Detect lines using Probabilistic Hough Transform."""
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=config.hough_threshold,
        minLineLength=config.hough_min_length,
        maxLineGap=config.hough_max_gap
    )

    if lines is None:
        return np.array([])

    return lines.reshape(-1, 4)


def refine_quad(
    image: np.ndarray,
    quad: np.ndarray,
    config: Optional[RefinementConfig] = None
) -> RefinementResult:
    """
    Refine document quad corners using edge detection and line fitting.

    Args:
        image: Original image (H, W, 3) RGB or BGR
        quad: Predicted corners [4, 2] in pixel coords, order: TL, TR, BR, BL
        config: Refinement configuration (uses defaults if None)

    Returns:
        RefinementResult with refined quad and diagnostics
    """
    if config is None:
        config = RefinementConfig()

    quad = np.array(quad, dtype=np.float32)
    original_quad = quad.copy()

    # Validate input
    if quad.shape != (4, 2):
        return RefinementResult(
            refined_quad=quad,
            original_quad=original_quad,
            refine_ok=False,
            reason="Invalid quad shape"
        )

    if not is_convex(quad):
        return RefinementResult(
            refined_quad=quad,
            original_quad=original_quad,
            refine_ok=False,
            reason="Input quad is not convex"
        )

    h_img, w_img = image.shape[:2]

    # Compute warp output size based on quad dimensions
    # Use max edge lengths for width/height
    top_edge = np.linalg.norm(quad[1] - quad[0])
    bottom_edge = np.linalg.norm(quad[2] - quad[3])
    left_edge = np.linalg.norm(quad[3] - quad[0])
    right_edge = np.linalg.norm(quad[2] - quad[1])

    warp_w = int(max(top_edge, bottom_edge))
    warp_h = int(max(left_edge, right_edge))

    # Clamp to config limits
    scale = 1.0
    if max(warp_w, warp_h) > config.max_warp_size:
        scale = config.max_warp_size / max(warp_w, warp_h)
    elif max(warp_w, warp_h) < config.min_warp_size:
        scale = config.min_warp_size / max(warp_w, warp_h)

    warp_w = int(warp_w * scale)
    warp_h = int(warp_h * scale)

    # Add margin
    margin = config.warp_margin
    warp_w_full = warp_w + 2 * margin
    warp_h_full = warp_h + 2 * margin

    # Destination points (rectangle with margin)
    dst_pts = np.array([
        [margin, margin],  # TL
        [margin + warp_w, margin],  # TR
        [margin + warp_w, margin + warp_h],  # BR
        [margin, margin + warp_h],  # BL
    ], dtype=np.float32)

    # Compute perspective transform
    M = cv2.getPerspectiveTransform(quad, dst_pts)
    M_inv = cv2.getPerspectiveTransform(dst_pts, quad)

    # Warp image
    warped = cv2.warpPerspective(image, M, (warp_w_full, warp_h_full))

    # Convert to grayscale if needed
    if len(warped.shape) == 3:
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    else:
        gray = warped

    # Edge detection
    edges = cv2.Canny(gray, config.canny_low, config.canny_high)

    # Compute edge strength (diagnostic)
    edge_strength = float(np.mean(edges))

    if edge_strength < config.min_edge_strength:
        return RefinementResult(
            refined_quad=quad,
            original_quad=original_quad,
            refine_ok=False,
            reason=f"Edge strength too low: {edge_strength:.1f} < {config.min_edge_strength}",
            edge_strength=edge_strength
        )

    # Detect lines
    if config.use_lsd:
        try:
            lines = detect_lines_lsd(gray)
        except Exception:
            lines = detect_lines_hough(edges, config)
    else:
        lines = detect_lines_hough(edges, config)

    if len(lines) == 0:
        return RefinementResult(
            refined_quad=quad,
            original_quad=original_quad,
            refine_ok=False,
            reason="No lines detected",
            edge_strength=edge_strength
        )

    # Classify lines
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        classification = classify_line(line, config)
        if classification == 'horizontal':
            horizontal_lines.append(line)
        elif classification == 'vertical':
            vertical_lines.append(line)

    num_horizontal = len(horizontal_lines)
    num_vertical = len(vertical_lines)

    if num_horizontal < config.min_lines_per_direction or num_vertical < config.min_lines_per_direction:
        return RefinementResult(
            refined_quad=quad,
            original_quad=original_quad,
            refine_ok=False,
            reason=f"Insufficient lines: {num_horizontal}H, {num_vertical}V (need {config.min_lines_per_direction} each)",
            num_lines_found=len(lines),
            num_horizontal=num_horizontal,
            num_vertical=num_vertical,
            edge_strength=edge_strength
        )

    # Select border lines
    warp_size = (warp_w_full, warp_h_full)
    top_lines = select_border_lines(horizontal_lines, warp_size, 'top', config)
    bottom_lines = select_border_lines(horizontal_lines, warp_size, 'bottom', config)
    left_lines = select_border_lines(vertical_lines, warp_size, 'left', config)
    right_lines = select_border_lines(vertical_lines, warp_size, 'right', config)

    if not all([top_lines, bottom_lines, left_lines, right_lines]):
        return RefinementResult(
            refined_quad=quad,
            original_quad=original_quad,
            refine_ok=False,
            reason="Could not find lines near all borders",
            num_lines_found=len(lines),
            num_horizontal=num_horizontal,
            num_vertical=num_vertical,
            edge_strength=edge_strength
        )

    # Fit robust lines for each border
    top_params = fit_line_robust(top_lines[:3])  # Use top 3 closest
    bottom_params = fit_line_robust(bottom_lines[:3])
    left_params = fit_line_robust(left_lines[:3])
    right_params = fit_line_robust(right_lines[:3])

    if None in [top_params, bottom_params, left_params, right_params]:
        return RefinementResult(
            refined_quad=quad,
            original_quad=original_quad,
            refine_ok=False,
            reason="Failed to fit lines for some borders",
            num_lines_found=len(lines),
            num_horizontal=num_horizontal,
            num_vertical=num_vertical,
            edge_strength=edge_strength
        )

    # Compute intersections (corners in warp space)
    tl_warp = line_intersection(top_params, left_params)
    tr_warp = line_intersection(top_params, right_params)
    br_warp = line_intersection(bottom_params, right_params)
    bl_warp = line_intersection(bottom_params, left_params)

    if None in [tl_warp, tr_warp, br_warp, bl_warp]:
        return RefinementResult(
            refined_quad=quad,
            original_quad=original_quad,
            refine_ok=False,
            reason="Line intersections failed (parallel lines)",
            num_lines_found=len(lines),
            num_horizontal=num_horizontal,
            num_vertical=num_vertical,
            edge_strength=edge_strength
        )

    refined_warp = np.array([tl_warp, tr_warp, br_warp, bl_warp], dtype=np.float32)

    # Check if refined corners are within warp bounds (with some tolerance)
    tolerance = max(warp_w_full, warp_h_full) * 0.3
    for corner in refined_warp:
        if corner[0] < -tolerance or corner[0] > warp_w_full + tolerance:
            return RefinementResult(
                refined_quad=quad,
                original_quad=original_quad,
                refine_ok=False,
                reason="Refined corners outside valid range",
                num_lines_found=len(lines),
                num_horizontal=num_horizontal,
                num_vertical=num_vertical,
                edge_strength=edge_strength
            )
        if corner[1] < -tolerance or corner[1] > warp_h_full + tolerance:
            return RefinementResult(
                refined_quad=quad,
                original_quad=original_quad,
                refine_ok=False,
                reason="Refined corners outside valid range",
                num_lines_found=len(lines),
                num_horizontal=num_horizontal,
                num_vertical=num_vertical,
                edge_strength=edge_strength
            )

    # Unwarp corners back to original image space
    refined_warp_homog = np.hstack([refined_warp, np.ones((4, 1))])
    refined_orig_homog = (M_inv @ refined_warp_homog.T).T
    refined_quad = refined_orig_homog[:, :2] / refined_orig_homog[:, 2:3]

    # Validate refined quad
    if not is_convex(refined_quad):
        return RefinementResult(
            refined_quad=original_quad,
            original_quad=original_quad,
            refine_ok=False,
            reason="Refined quad is not convex",
            num_lines_found=len(lines),
            num_horizontal=num_horizontal,
            num_vertical=num_vertical,
            edge_strength=edge_strength
        )

    # Check area ratio
    original_area = compute_quad_area(original_quad)
    refined_area = compute_quad_area(refined_quad)
    area_ratio = refined_area / (original_area + 1e-8)

    if area_ratio < config.area_ratio_min or area_ratio > config.area_ratio_max:
        return RefinementResult(
            refined_quad=original_quad,
            original_quad=original_quad,
            refine_ok=False,
            reason=f"Area ratio {area_ratio:.3f} outside [{config.area_ratio_min}, {config.area_ratio_max}]",
            num_lines_found=len(lines),
            num_horizontal=num_horizontal,
            num_vertical=num_vertical,
            edge_strength=edge_strength,
            area_ratio=area_ratio
        )

    # Check max corner shift
    corner_shifts = np.linalg.norm(refined_quad - original_quad, axis=1)
    max_shift = np.max(corner_shifts)

    if max_shift > config.max_corner_shift:
        return RefinementResult(
            refined_quad=original_quad,
            original_quad=original_quad,
            refine_ok=False,
            reason=f"Max corner shift {max_shift:.1f}px exceeds limit {config.max_corner_shift}px",
            num_lines_found=len(lines),
            num_horizontal=num_horizontal,
            num_vertical=num_vertical,
            edge_strength=edge_strength,
            area_ratio=area_ratio
        )

    # Clip to image bounds
    refined_quad[:, 0] = np.clip(refined_quad[:, 0], 0, w_img - 1)
    refined_quad[:, 1] = np.clip(refined_quad[:, 1], 0, h_img - 1)

    # Compute fit residual (mean distance from detected lines to fitted lines)
    fit_residual = 0.0
    # Simplified: use corner shift as proxy
    fit_residual = float(np.mean(corner_shifts))

    return RefinementResult(
        refined_quad=refined_quad,
        original_quad=original_quad,
        refine_ok=True,
        reason="Refinement successful",
        num_lines_found=len(lines),
        num_horizontal=num_horizontal,
        num_vertical=num_vertical,
        edge_strength=edge_strength,
        fit_residual=fit_residual,
        area_ratio=area_ratio
    )


def refine_quad_batch(
    images: List[np.ndarray],
    quads: np.ndarray,
    config: Optional[RefinementConfig] = None
) -> List[RefinementResult]:
    """
    Refine multiple quads in batch.

    Args:
        images: List of images (H, W, 3)
        quads: Array of quads [N, 4, 2]
        config: Refinement configuration

    Returns:
        List of RefinementResult
    """
    results = []
    for image, quad in zip(images, quads):
        result = refine_quad(image, quad, config)
        results.append(result)
    return results


if __name__ == "__main__":
    # Quick test
    import sys

    print("Refine Quad Module Test")
    print("=" * 50)

    # Create a simple test image with a white rectangle
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 80), (540, 400), (255, 255, 255), -1)
    cv2.rectangle(img, (100, 80), (540, 400), (0, 0, 0), 2)

    # Simulate predicted quad (slightly off from perfect)
    pred_quad = np.array([
        [105, 85],   # TL - slightly off
        [535, 82],   # TR
        [542, 398],  # BR
        [98, 402],   # BL
    ], dtype=np.float32)

    # Run refinement
    result = refine_quad(img, pred_quad)

    print(f"Refine OK: {result.refine_ok}")
    print(f"Reason: {result.reason}")
    print(f"Original quad:\n{result.original_quad}")
    print(f"Refined quad:\n{result.refined_quad}")
    print(f"Lines found: {result.num_lines_found} ({result.num_horizontal}H, {result.num_vertical}V)")
    print(f"Edge strength: {result.edge_strength:.2f}")
    print(f"Area ratio: {result.area_ratio:.3f}")

    if result.refine_ok:
        shifts = np.linalg.norm(result.refined_quad - result.original_quad, axis=1)
        print(f"Corner shifts (px): {shifts}")
        print(f"Mean shift: {np.mean(shifts):.2f}px")
