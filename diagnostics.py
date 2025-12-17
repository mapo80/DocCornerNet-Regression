"""
Diagnostics Module for Document Corner Detection Error Attribution.

This module provides tools to analyze model predictions and identify
failure modes by computing per-sample metrics and correlations.

Key analyses:
- Per-sample metrics: IoU, corner error, area ratios
- Image quality indicators: sharpness, boundary contrast
- Error correlations with image properties
- Domain-based breakdown
- Worst case identification
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import json

try:
    from shapely.geometry import Polygon
    from shapely.validation import make_valid
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False


@dataclass
class SampleDiagnostics:
    """Diagnostic metrics for a single sample."""
    # Identification
    image_name: str
    domain: str  # extracted from filename prefix

    # Model outputs
    score: float  # model confidence (after sigmoid)
    has_gt: bool

    # Localization metrics (only for samples with GT)
    iou: float = 0.0
    corner_err_mean_px: float = 0.0
    corner_err_tl_px: float = 0.0
    corner_err_tr_px: float = 0.0
    corner_err_br_px: float = 0.0
    corner_err_bl_px: float = 0.0

    # Area analysis
    gt_area: float = 0.0  # normalized [0,1]
    pred_area: float = 0.0
    area_ratio: float = 1.0  # pred/gt

    # Perspective/geometry
    perspective_score: float = 0.0  # proxy for perspective distortion

    # Image quality
    sharpness: float = 0.0  # variance of Laplacian
    boundary_contrast: float = 0.0  # gradient along GT edges

    # Flags
    is_small_doc: bool = False  # area < 10%
    is_high_perspective: bool = False
    is_low_contrast: bool = False
    is_blurry: bool = False

    # Coordinates (for visualization)
    pred_coords: np.ndarray = field(default_factory=lambda: np.zeros(8))
    gt_coords: np.ndarray = field(default_factory=lambda: np.zeros(8))


def compute_sharpness(image: np.ndarray) -> float:
    """Compute image sharpness using variance of Laplacian."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.var(laplacian))


def compute_boundary_contrast(image: np.ndarray, quad: np.ndarray, num_samples: int = 50) -> float:
    """
    Compute mean gradient magnitude along the document boundaries.

    Args:
        image: Input image (H, W, 3) or (H, W)
        quad: Corner coordinates [4, 2] in pixel space (TL, TR, BR, BL)
        num_samples: Number of sample points per edge

    Returns:
        Mean gradient magnitude along boundaries
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    h, w = gray.shape

    # Compute gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Sample points along each edge
    gradients = []
    for i in range(4):
        p1 = quad[i]
        p2 = quad[(i + 1) % 4]

        for t in np.linspace(0, 1, num_samples):
            pt = p1 + t * (p2 - p1)
            x, y = int(np.clip(pt[0], 0, w - 1)), int(np.clip(pt[1], 0, h - 1))
            gradients.append(grad_mag[y, x])

    return float(np.mean(gradients)) if gradients else 0.0


def compute_perspective_score(quad: np.ndarray) -> float:
    """
    Compute a proxy score for perspective distortion.

    Uses ratio of opposite edge lengths. Perfect rectangle = 1.0.
    """
    # Edge lengths
    top = np.linalg.norm(quad[1] - quad[0])
    bottom = np.linalg.norm(quad[2] - quad[3])
    left = np.linalg.norm(quad[3] - quad[0])
    right = np.linalg.norm(quad[2] - quad[1])

    # Ratios (should be 1.0 for rectangle)
    h_ratio = min(top, bottom) / (max(top, bottom) + 1e-8)
    v_ratio = min(left, right) / (max(left, right) + 1e-8)

    # Combined score (1.0 = perfect rectangle, 0.0 = highly distorted)
    return float(h_ratio * v_ratio)


def compute_quad_area_normalized(coords: np.ndarray) -> float:
    """Compute normalized area [0, 1] of quad using Shoelace formula."""
    if coords.shape == (8,):
        x = coords[0::2]
        y = coords[1::2]
    else:
        x = coords[:, 0]
        y = coords[:, 1]

    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    return 0.5 * np.abs(np.sum(x * y_next - x_next * y))


def coords_to_polygon(coords: np.ndarray) -> "Polygon":
    """Convert 8-value coordinate array to Shapely Polygon."""
    if coords.shape == (8,):
        points = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]
    else:
        points = [(coords[i, 0], coords[i, 1]) for i in range(4)]

    poly = Polygon(points)
    if not poly.is_valid:
        poly = make_valid(poly)
        if poly.geom_type == 'GeometryCollection':
            for geom in poly.geoms:
                if geom.geom_type == 'Polygon':
                    return geom
            return Polygon(points).convex_hull
        elif poly.geom_type == 'MultiPolygon':
            return max(poly.geoms, key=lambda p: p.area)
    return poly


def compute_iou(pred_coords: np.ndarray, gt_coords: np.ndarray) -> float:
    """Compute IoU between predicted and ground truth quads."""
    if not SHAPELY_AVAILABLE:
        return 0.0

    try:
        pred_poly = coords_to_polygon(pred_coords)
        gt_poly = coords_to_polygon(gt_coords)

        if pred_poly.is_empty or gt_poly.is_empty:
            return 0.0

        intersection = pred_poly.intersection(gt_poly).area
        union = pred_poly.union(gt_poly).area

        return float(intersection / union) if union > 0 else 0.0
    except Exception:
        return 0.0


def compute_corner_errors(pred_coords: np.ndarray, gt_coords: np.ndarray, img_size: int) -> Tuple[float, List[float]]:
    """
    Compute per-corner Euclidean errors in pixels.

    Args:
        pred_coords: Predicted coordinates [8] normalized
        gt_coords: Ground truth coordinates [8] normalized
        img_size: Image size for pixel conversion

    Returns:
        (mean_error, [tl_err, tr_err, br_err, bl_err])
    """
    errors = []
    for i in range(4):
        px = pred_coords[2*i] * img_size
        py = pred_coords[2*i + 1] * img_size
        gx = gt_coords[2*i] * img_size
        gy = gt_coords[2*i + 1] * img_size
        err = np.sqrt((px - gx)**2 + (py - gy)**2)
        errors.append(err)

    return float(np.mean(errors)), errors


def extract_domain(image_name: str) -> str:
    """Extract domain from image filename (prefix before first underscore)."""
    stem = Path(image_name).stem
    if '_' in stem:
        return stem.split('_')[0]
    return "other"


def diagnose_sample(
    image: np.ndarray,
    image_name: str,
    pred_coords: np.ndarray,
    pred_score: float,
    gt_coords: Optional[np.ndarray],
    has_gt: bool,
    img_size: int = 320,
    thresholds: Optional[Dict] = None
) -> SampleDiagnostics:
    """
    Compute full diagnostics for a single sample.

    Args:
        image: Original image (H, W, 3)
        image_name: Image filename
        pred_coords: Predicted coordinates [8] normalized
        pred_score: Predicted score (after sigmoid)
        gt_coords: Ground truth coordinates [8] normalized (or None)
        has_gt: Whether sample has valid ground truth
        img_size: Model input size for pixel conversion
        thresholds: Custom thresholds for flags

    Returns:
        SampleDiagnostics with all computed metrics
    """
    if thresholds is None:
        thresholds = {
            'small_doc_area': 0.10,
            'high_perspective': 0.7,
            'low_contrast': 30.0,
            'blurry': 100.0,
        }

    domain = extract_domain(image_name)

    # Initialize diagnostics
    diag = SampleDiagnostics(
        image_name=image_name,
        domain=domain,
        score=pred_score,
        has_gt=has_gt,
        pred_coords=pred_coords.copy(),
    )

    # Image quality metrics (always computed)
    diag.sharpness = compute_sharpness(image)
    diag.is_blurry = diag.sharpness < thresholds['blurry']

    # Predicted area
    diag.pred_area = compute_quad_area_normalized(pred_coords)

    if has_gt and gt_coords is not None:
        diag.gt_coords = gt_coords.copy()

        # Localization metrics
        diag.iou = compute_iou(pred_coords, gt_coords)

        mean_err, per_corner_err = compute_corner_errors(pred_coords, gt_coords, img_size)
        diag.corner_err_mean_px = mean_err
        diag.corner_err_tl_px = per_corner_err[0]
        diag.corner_err_tr_px = per_corner_err[1]
        diag.corner_err_br_px = per_corner_err[2]
        diag.corner_err_bl_px = per_corner_err[3]

        # Area analysis
        diag.gt_area = compute_quad_area_normalized(gt_coords)
        diag.area_ratio = diag.pred_area / (diag.gt_area + 1e-8)

        # Convert GT to pixel coords for boundary contrast
        h, w = image.shape[:2]
        gt_quad_px = np.array([
            [gt_coords[0] * w, gt_coords[1] * h],
            [gt_coords[2] * w, gt_coords[3] * h],
            [gt_coords[4] * w, gt_coords[5] * h],
            [gt_coords[6] * w, gt_coords[7] * h],
        ])

        diag.boundary_contrast = compute_boundary_contrast(image, gt_quad_px)
        diag.is_low_contrast = diag.boundary_contrast < thresholds['low_contrast']

        # Perspective
        diag.perspective_score = compute_perspective_score(gt_quad_px)
        diag.is_high_perspective = diag.perspective_score < thresholds['high_perspective']

        # Flags
        diag.is_small_doc = diag.gt_area < thresholds['small_doc_area']

    return diag


def diagnostics_to_dataframe(diagnostics: List[SampleDiagnostics]) -> pd.DataFrame:
    """Convert list of diagnostics to pandas DataFrame."""
    records = []
    for d in diagnostics:
        record = {
            'image_name': d.image_name,
            'domain': d.domain,
            'score': d.score,
            'has_gt': d.has_gt,
            'iou': d.iou,
            'corner_err_mean_px': d.corner_err_mean_px,
            'corner_err_tl_px': d.corner_err_tl_px,
            'corner_err_tr_px': d.corner_err_tr_px,
            'corner_err_br_px': d.corner_err_br_px,
            'corner_err_bl_px': d.corner_err_bl_px,
            'gt_area': d.gt_area,
            'pred_area': d.pred_area,
            'area_ratio': d.area_ratio,
            'perspective_score': d.perspective_score,
            'sharpness': d.sharpness,
            'boundary_contrast': d.boundary_contrast,
            'is_small_doc': d.is_small_doc,
            'is_high_perspective': d.is_high_perspective,
            'is_low_contrast': d.is_low_contrast,
            'is_blurry': d.is_blurry,
        }
        records.append(record)

    return pd.DataFrame(records)


def compute_correlations(df: pd.DataFrame) -> Dict[str, float]:
    """Compute correlations between error and various factors."""
    # Filter to samples with GT
    df_gt = df[df['has_gt'] == True].copy()

    if len(df_gt) < 10:
        return {}

    correlations = {}

    # Error vs various factors
    for col in ['sharpness', 'boundary_contrast', 'gt_area', 'score', 'perspective_score']:
        if col in df_gt.columns:
            corr = df_gt['corner_err_mean_px'].corr(df_gt[col])
            correlations[f'err_vs_{col}'] = float(corr) if not np.isnan(corr) else 0.0

    return correlations


def compute_domain_breakdown(df: pd.DataFrame) -> Dict[str, Dict]:
    """Compute metrics breakdown by domain."""
    df_gt = df[df['has_gt'] == True].copy()

    breakdown = {}
    for domain in df_gt['domain'].unique():
        domain_df = df_gt[df_gt['domain'] == domain]
        breakdown[domain] = {
            'count': len(domain_df),
            'mean_iou': float(domain_df['iou'].mean()),
            'median_iou': float(domain_df['iou'].median()),
            'mean_corner_err_px': float(domain_df['corner_err_mean_px'].mean()),
            'r90': float((domain_df['iou'] >= 0.90).sum() / len(domain_df) * 100),
            'r95': float((domain_df['iou'] >= 0.95).sum() / len(domain_df) * 100),
            'pct_small_doc': float(domain_df['is_small_doc'].sum() / len(domain_df) * 100),
            'pct_blurry': float(domain_df['is_blurry'].sum() / len(domain_df) * 100),
            'pct_low_contrast': float(domain_df['is_low_contrast'].sum() / len(domain_df) * 100),
        }

    return breakdown


def identify_worst_cases(df: pd.DataFrame, n: int = 50) -> Dict[str, List[str]]:
    """Identify worst performing samples."""
    df_gt = df[df['has_gt'] == True].copy()

    results = {}

    # Worst by IoU
    worst_iou = df_gt.nsmallest(n, 'iou')['image_name'].tolist()
    results['worst_by_iou'] = worst_iou

    # Worst by corner error
    worst_err = df_gt.nlargest(n, 'corner_err_mean_px')['image_name'].tolist()
    results['worst_by_corner_err'] = worst_err

    # Overconfident bad localization
    overconfident = df_gt[(df_gt['score'] >= 0.8) & (df_gt['corner_err_mean_px'] >= 15)]
    results['overconfident_bad_loc'] = overconfident['image_name'].tolist()

    return results


def generate_summary(df: pd.DataFrame) -> Dict:
    """Generate comprehensive summary statistics."""
    df_gt = df[df['has_gt'] == True].copy()
    df_neg = df[df['has_gt'] == False].copy()

    summary = {
        'total_samples': len(df),
        'positive_samples': len(df_gt),
        'negative_samples': len(df_neg),
    }

    if len(df_gt) > 0:
        summary['localization'] = {
            'mean_iou': float(df_gt['iou'].mean()),
            'median_iou': float(df_gt['iou'].median()),
            'std_iou': float(df_gt['iou'].std()),
            'mean_corner_err_px': float(df_gt['corner_err_mean_px'].mean()),
            'median_corner_err_px': float(df_gt['corner_err_mean_px'].median()),
            'p90_corner_err_px': float(df_gt['corner_err_mean_px'].quantile(0.90)),
            'p95_corner_err_px': float(df_gt['corner_err_mean_px'].quantile(0.95)),
            'recall_50': float((df_gt['iou'] >= 0.50).sum() / len(df_gt) * 100),
            'recall_75': float((df_gt['iou'] >= 0.75).sum() / len(df_gt) * 100),
            'recall_90': float((df_gt['iou'] >= 0.90).sum() / len(df_gt) * 100),
            'recall_95': float((df_gt['iou'] >= 0.95).sum() / len(df_gt) * 100),
        }

        # Per-corner analysis
        summary['per_corner'] = {
            'TL': {
                'mean_px': float(df_gt['corner_err_tl_px'].mean()),
                'median_px': float(df_gt['corner_err_tl_px'].median()),
                'p90_px': float(df_gt['corner_err_tl_px'].quantile(0.90)),
            },
            'TR': {
                'mean_px': float(df_gt['corner_err_tr_px'].mean()),
                'median_px': float(df_gt['corner_err_tr_px'].median()),
                'p90_px': float(df_gt['corner_err_tr_px'].quantile(0.90)),
            },
            'BR': {
                'mean_px': float(df_gt['corner_err_br_px'].mean()),
                'median_px': float(df_gt['corner_err_br_px'].median()),
                'p90_px': float(df_gt['corner_err_br_px'].quantile(0.90)),
            },
            'BL': {
                'mean_px': float(df_gt['corner_err_bl_px'].mean()),
                'median_px': float(df_gt['corner_err_bl_px'].median()),
                'p90_px': float(df_gt['corner_err_bl_px'].quantile(0.90)),
            },
        }

        # Flag percentages
        summary['quality_flags'] = {
            'pct_small_doc': float(df_gt['is_small_doc'].sum() / len(df_gt) * 100),
            'pct_high_perspective': float(df_gt['is_high_perspective'].sum() / len(df_gt) * 100),
            'pct_low_contrast': float(df_gt['is_low_contrast'].sum() / len(df_gt) * 100),
            'pct_blurry': float(df_gt['is_blurry'].sum() / len(df_gt) * 100),
        }

        # Correlations
        summary['correlations'] = compute_correlations(df)

    # Score classification
    if len(df) > 0:
        pred_binary = (df['score'] >= 0.5).astype(int)
        gt_binary = df['has_gt'].astype(int)

        tp = ((pred_binary == 1) & (gt_binary == 1)).sum()
        fp = ((pred_binary == 1) & (gt_binary == 0)).sum()
        tn = ((pred_binary == 0) & (gt_binary == 0)).sum()
        fn = ((pred_binary == 0) & (gt_binary == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        summary['score_classification'] = {
            'accuracy': float((tp + tn) / len(df)),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn),
        }

    return summary


if __name__ == "__main__":
    # Quick test
    print("Diagnostics Module Test")
    print("=" * 50)

    # Create synthetic test data
    np.random.seed(42)

    diagnostics = []
    for i in range(20):
        # Simulate predictions
        has_gt = i < 15
        pred_coords = np.random.rand(8) * 0.6 + 0.2
        gt_coords = pred_coords + np.random.randn(8) * 0.02 if has_gt else None
        score = np.random.rand() * 0.3 + 0.7 if has_gt else np.random.rand() * 0.3

        # Create dummy image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        diag = diagnose_sample(
            image=image,
            image_name=f"test_{i:04d}.jpg",
            pred_coords=pred_coords,
            pred_score=score,
            gt_coords=gt_coords,
            has_gt=has_gt,
        )
        diagnostics.append(diag)

    # Convert to DataFrame
    df = diagnostics_to_dataframe(diagnostics)
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")

    # Generate summary
    summary = generate_summary(df)
    print(f"\nSummary:")
    print(json.dumps(summary, indent=2, default=str))

    # Domain breakdown
    breakdown = compute_domain_breakdown(df)
    print(f"\nDomain Breakdown:")
    for domain, stats in breakdown.items():
        print(f"  {domain}: {stats['count']} samples, IoU={stats['mean_iou']:.3f}")

    # Worst cases
    worst = identify_worst_cases(df, n=5)
    print(f"\nWorst by IoU: {worst['worst_by_iou']}")
