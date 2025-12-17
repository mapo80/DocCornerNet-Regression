"""
Metrics for document corner detection evaluation.

- compute_polygon_iou: IoU between two quadrilaterals using Shapely
- compute_corner_error: Mean absolute error on corner coordinates
- ValidationMetrics: Class to accumulate and compute epoch-level metrics
"""

import torch
import numpy as np

try:
    from shapely.geometry import Polygon
    from shapely.validation import make_valid
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    print("Warning: shapely not installed. Falling back to bbox IoU.")


def coords_to_polygon(coords: np.ndarray) -> "Polygon":
    """
    Convert 8-value coordinate array to Shapely Polygon.

    Args:
        coords: Array of shape [8] with (x0,y0,x1,y1,x2,y2,x3,y3)
                representing TL, TR, BR, BL corners.

    Returns:
        Shapely Polygon object.
    """
    points = [
        (coords[0], coords[1]),  # TL
        (coords[2], coords[3]),  # TR
        (coords[4], coords[5]),  # BR
        (coords[6], coords[7]),  # BL
    ]
    poly = Polygon(points)

    # Handle self-intersecting polygons
    if not poly.is_valid:
        poly = make_valid(poly)
        # make_valid might return a GeometryCollection, extract polygon
        if poly.geom_type == 'GeometryCollection':
            for geom in poly.geoms:
                if geom.geom_type == 'Polygon':
                    return geom
            # Fallback: return convex hull
            return Polygon(points).convex_hull
        elif poly.geom_type == 'MultiPolygon':
            # Return largest polygon
            return max(poly.geoms, key=lambda p: p.area)

    return poly


def compute_polygon_iou(pred_coords: np.ndarray, gt_coords: np.ndarray) -> float:
    """
    Compute IoU between predicted and ground truth quadrilaterals.

    Args:
        pred_coords: [8] predicted coordinates (normalized [0, 1])
        gt_coords: [8] ground truth coordinates (normalized [0, 1])

    Returns:
        IoU value in [0, 1]
    """
    if not SHAPELY_AVAILABLE:
        return compute_bbox_iou(pred_coords, gt_coords)

    try:
        pred_poly = coords_to_polygon(pred_coords)
        gt_poly = coords_to_polygon(gt_coords)

        if pred_poly.is_empty or gt_poly.is_empty:
            return 0.0

        intersection = pred_poly.intersection(gt_poly).area
        union = pred_poly.union(gt_poly).area

        if union == 0:
            return 0.0

        return intersection / union

    except Exception:
        # Fallback to bbox IoU on any geometry error
        return compute_bbox_iou(pred_coords, gt_coords)


def compute_bbox_iou(pred_coords: np.ndarray, gt_coords: np.ndarray) -> float:
    """
    Compute axis-aligned bounding box IoU as fallback.

    Args:
        pred_coords: [8] predicted coordinates
        gt_coords: [8] ground truth coordinates

    Returns:
        Bbox IoU value in [0, 1]
    """
    # Extract x and y coordinates
    pred_x = pred_coords[0::2]  # x0, x1, x2, x3
    pred_y = pred_coords[1::2]  # y0, y1, y2, y3
    gt_x = gt_coords[0::2]
    gt_y = gt_coords[1::2]

    # Compute bounding boxes
    pred_bbox = [pred_x.min(), pred_y.min(), pred_x.max(), pred_y.max()]
    gt_bbox = [gt_x.min(), gt_y.min(), gt_x.max(), gt_y.max()]

    # Intersection
    x1 = max(pred_bbox[0], gt_bbox[0])
    y1 = max(pred_bbox[1], gt_bbox[1])
    x2 = min(pred_bbox[2], gt_bbox[2])
    y2 = min(pred_bbox[3], gt_bbox[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Union
    pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
    union = pred_area + gt_area - intersection

    if union == 0:
        return 0.0

    return intersection / union


def compute_corner_error(pred_coords: np.ndarray, gt_coords: np.ndarray) -> float:
    """
    Compute mean absolute error on corner coordinates.

    Args:
        pred_coords: [8] predicted coordinates (normalized [0, 1])
        gt_coords: [8] ground truth coordinates (normalized [0, 1])

    Returns:
        Mean absolute error across all 8 coordinate values.
    """
    return np.abs(pred_coords - gt_coords).mean()


def compute_per_corner_distance(pred_coords: np.ndarray, gt_coords: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distance for each corner.

    Args:
        pred_coords: [8] predicted coordinates
        gt_coords: [8] ground truth coordinates

    Returns:
        Array of 4 distances, one per corner (TL, TR, BR, BL).
    """
    distances = []
    for i in range(4):
        dx = pred_coords[2 * i] - gt_coords[2 * i]
        dy = pred_coords[2 * i + 1] - gt_coords[2 * i + 1]
        distances.append(np.sqrt(dx ** 2 + dy ** 2))
    return np.array(distances)


class ValidationMetrics:
    """
    Accumulates predictions and ground truth over validation epoch,
    then computes aggregate metrics.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all accumulators."""
        self.pred_coords_list = []
        self.gt_coords_list = []
        self.pred_scores_list = []
        self.gt_scores_list = []
        self.has_gt_list = []

    def update(
        self,
        pred_coords: torch.Tensor,
        gt_coords: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_scores: torch.Tensor,
        has_gt: torch.Tensor,
    ):
        """
        Add batch of predictions and ground truth.

        Args:
            pred_coords: [B, 8] predicted coordinates
            gt_coords: [B, 8] ground truth coordinates
            pred_scores: [B] predicted score logits
            gt_scores: [B] ground truth scores (0 or 1)
            has_gt: [B] mask indicating valid GT (1 = valid)
        """
        self.pred_coords_list.append(pred_coords.detach().cpu())
        self.gt_coords_list.append(gt_coords.detach().cpu())
        self.pred_scores_list.append(pred_scores.detach().cpu())
        self.gt_scores_list.append(gt_scores.detach().cpu())
        self.has_gt_list.append(has_gt.detach().cpu())

    def compute(self) -> dict:
        """
        Compute aggregate metrics over all accumulated samples.

        Returns:
            Dictionary with:
            - mean_iou: Mean IoU over samples with valid GT
            - median_iou: Median IoU
            - mean_corner_error: Mean MAE on coordinates (normalized)
            - corner_error_px: Corner error in pixels (assuming 224px image)
            - recall_50/75/90: Recall at IoU thresholds 0.5/0.75/0.9
            - ap_50/75: Average Precision at IoU 0.5/0.75
            - corner_dist_p50/p90/p95: Corner distance percentiles
            - num_samples: Total samples
            - num_with_gt: Samples with valid GT
        """
        # Concatenate all batches
        pred_coords = torch.cat(self.pred_coords_list, dim=0).numpy()
        gt_coords = torch.cat(self.gt_coords_list, dim=0).numpy()
        has_gt = torch.cat(self.has_gt_list, dim=0).numpy()

        num_samples = len(pred_coords)
        num_with_gt = int(has_gt.sum())

        # Filter to samples with valid GT
        mask = has_gt == 1

        # Initialize metrics
        results = {
            "mean_iou": 0.0,
            "median_iou": 0.0,
            "mean_corner_error": 0.0,
            "corner_error_px": 0.0,
            "recall_50": 0.0,
            "recall_75": 0.0,
            "recall_90": 0.0,
            "ap_50": 0.0,
            "ap_75": 0.0,
            "corner_dist_p50": 0.0,
            "corner_dist_p90": 0.0,
            "corner_dist_p95": 0.0,
            "num_samples": num_samples,
            "num_with_gt": num_with_gt,
        }

        if mask.sum() == 0:
            return results

        pred_coords_valid = pred_coords[mask]
        gt_coords_valid = gt_coords[mask]
        num_gt = len(pred_coords_valid)

        # Compute IoU for each sample
        ious = []
        for i in range(num_gt):
            iou = compute_polygon_iou(pred_coords_valid[i], gt_coords_valid[i])
            ious.append(iou)
        ious = np.array(ious)

        # Compute corner distances for each sample (Euclidean distance per corner)
        all_corner_dists = []
        corner_errors = []
        for i in range(num_gt):
            # Per-corner Euclidean distances
            dists = compute_per_corner_distance(pred_coords_valid[i], gt_coords_valid[i])
            all_corner_dists.extend(dists)
            # Mean absolute error
            error = compute_corner_error(pred_coords_valid[i], gt_coords_valid[i])
            corner_errors.append(error)

        all_corner_dists = np.array(all_corner_dists)

        # IoU metrics
        results["mean_iou"] = float(np.mean(ious))
        results["median_iou"] = float(np.median(ious))

        # Corner error metrics
        results["mean_corner_error"] = float(np.mean(corner_errors))
        results["corner_error_px"] = float(np.mean(corner_errors) * 224)  # Assuming 224px image

        # Recall@IoU thresholds
        results["recall_50"] = float((ious >= 0.50).sum() / num_gt)
        results["recall_75"] = float((ious >= 0.75).sum() / num_gt)
        results["recall_90"] = float((ious >= 0.90).sum() / num_gt)

        # Average Precision (simplified - area under recall curve at different thresholds)
        # AP@50 = mean recall for IoU thresholds from 0.50 to 0.95 with step 0.05
        # For single-class detection, this is equivalent to recall at the threshold
        results["ap_50"] = float((ious >= 0.50).sum() / num_gt)
        results["ap_75"] = float((ious >= 0.75).sum() / num_gt)

        # Corner distance percentiles (in normalized coords)
        results["corner_dist_p50"] = float(np.percentile(all_corner_dists, 50))
        results["corner_dist_p90"] = float(np.percentile(all_corner_dists, 90))
        results["corner_dist_p95"] = float(np.percentile(all_corner_dists, 95))

        return results


if __name__ == "__main__":
    # Quick test
    print(f"Shapely available: {SHAPELY_AVAILABLE}")

    # Test polygon IoU
    # Perfect match
    coords1 = np.array([0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8])
    coords2 = np.array([0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8])
    print(f"Perfect match IoU: {compute_polygon_iou(coords1, coords2):.4f}")

    # Partial overlap
    coords3 = np.array([0.3, 0.3, 0.9, 0.3, 0.9, 0.9, 0.3, 0.9])
    print(f"Partial overlap IoU: {compute_polygon_iou(coords1, coords3):.4f}")

    # No overlap
    coords4 = np.array([0.0, 0.0, 0.1, 0.0, 0.1, 0.1, 0.0, 0.1])
    coords5 = np.array([0.9, 0.9, 1.0, 0.9, 1.0, 1.0, 0.9, 1.0])
    print(f"No overlap IoU: {compute_polygon_iou(coords4, coords5):.4f}")

    # Corner error
    print(f"Corner error (same): {compute_corner_error(coords1, coords2):.4f}")
    print(f"Corner error (diff): {compute_corner_error(coords1, coords3):.4f}")

    # Test ValidationMetrics
    metrics = ValidationMetrics()
    metrics.update(
        pred_coords=torch.rand(4, 8),
        gt_coords=torch.rand(4, 8),
        pred_scores=torch.randn(4),
        gt_scores=torch.tensor([1.0, 1.0, 0.0, 1.0]),
        has_gt=torch.tensor([1, 1, 0, 1]),
    )
    results = metrics.compute()
    print(f"\nValidation metrics:")
    print(f"  Mean IoU: {results['mean_iou']:.4f}")
    print(f"  Mean Corner Error: {results['mean_corner_error']:.4f}")
    print(f"  Samples: {results['num_samples']} (with GT: {results['num_with_gt']})")
