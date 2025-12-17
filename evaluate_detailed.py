"""
Detailed Evaluation of DocCornerNet model.

Comprehensive metrics including:
- IoU statistics with histogram and percentiles
- Per-corner analysis (TL, TR, BR, BL)
- Recall at multiple IoU thresholds (50, 75, 80, 85, 90, 95)
- Corner distance analysis in pixels
- Coordinate range analysis (border detection check)
- Error breakdown by difficulty (small/medium/large documents)
- Score classification metrics (precision, recall, F1)
- Per-sample analysis with worst cases identification

Usage:
    python evaluate_detailed.py \
        --checkpoint /workspace/checkpoints/doccornernet_v5_wing/best.pth \
        --data_root /workspace/doc-scanner-dataset-labeled \
        --split val_with_negative_v2 \
        --output_dir eval_results
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from model import create_model
from dataset import DocDataset

try:
    from shapely.geometry import Polygon
    from shapely.validation import make_valid
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    print("Warning: shapely not installed. IoU calculation will be approximate.")


# ============================================================================
# Geometry Utilities
# ============================================================================

def coords_to_polygon(coords: np.ndarray) -> "Polygon":
    """Convert 8-value coordinate array to Shapely Polygon."""
    points = [
        (coords[0], coords[1]),  # TL
        (coords[2], coords[3]),  # TR
        (coords[4], coords[5]),  # BR
        (coords[6], coords[7]),  # BL
    ]
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


def compute_polygon_iou(pred_coords: np.ndarray, gt_coords: np.ndarray) -> float:
    """Compute IoU between predicted and ground truth quadrilaterals."""
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
        return compute_bbox_iou(pred_coords, gt_coords)


def compute_bbox_iou(pred_coords: np.ndarray, gt_coords: np.ndarray) -> float:
    """Compute axis-aligned bounding box IoU as fallback."""
    pred_x = pred_coords[0::2]
    pred_y = pred_coords[1::2]
    gt_x = gt_coords[0::2]
    gt_y = gt_coords[1::2]
    pred_bbox = [pred_x.min(), pred_y.min(), pred_x.max(), pred_y.max()]
    gt_bbox = [gt_x.min(), gt_y.min(), gt_x.max(), gt_y.max()]
    x1 = max(pred_bbox[0], gt_bbox[0])
    y1 = max(pred_bbox[1], gt_bbox[1])
    x2 = min(pred_bbox[2], gt_bbox[2])
    y2 = min(pred_bbox[3], gt_bbox[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
    union = pred_area + gt_area - intersection
    if union == 0:
        return 0.0
    return intersection / union


def compute_quad_area(coords: np.ndarray) -> float:
    """Compute quadrilateral area using Shoelace formula (normalized)."""
    x = coords[0::2]
    y = coords[1::2]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    area = 0.5 * np.abs((x * y_next - x_next * y).sum())
    return area


def compute_per_corner_distance(pred_coords: np.ndarray, gt_coords: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance for each corner (normalized)."""
    distances = []
    for i in range(4):
        dx = pred_coords[2 * i] - gt_coords[2 * i]
        dy = pred_coords[2 * i + 1] - gt_coords[2 * i + 1]
        distances.append(np.sqrt(dx ** 2 + dy ** 2))
    return np.array(distances)


# ============================================================================
# Detailed Metrics Calculator
# ============================================================================

class DetailedMetrics:
    """Compute comprehensive evaluation metrics."""

    CORNER_NAMES = ["TL", "TR", "BR", "BL"]
    IOU_THRESHOLDS = [0.50, 0.75, 0.80, 0.85, 0.90, 0.95]

    def __init__(self, img_size: int = 320):
        self.img_size = img_size
        self.reset()

    def reset(self):
        self.pred_coords_list = []
        self.gt_coords_list = []
        self.pred_scores_list = []
        self.gt_scores_list = []
        self.has_gt_list = []
        self.image_names_list = []

    def update(
        self,
        pred_coords: torch.Tensor,
        gt_coords: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_scores: torch.Tensor,
        has_gt: torch.Tensor,
        image_names: list = None,
    ):
        self.pred_coords_list.append(pred_coords.detach().cpu())
        self.gt_coords_list.append(gt_coords.detach().cpu())
        self.pred_scores_list.append(pred_scores.detach().cpu())
        self.gt_scores_list.append(gt_scores.detach().cpu())
        self.has_gt_list.append(has_gt.detach().cpu())
        if image_names:
            self.image_names_list.extend(image_names)

    def compute(self) -> dict:
        """Compute all detailed metrics."""
        # Concatenate
        pred_coords = torch.cat(self.pred_coords_list, dim=0).numpy()
        gt_coords = torch.cat(self.gt_coords_list, dim=0).numpy()
        pred_scores = torch.cat(self.pred_scores_list, dim=0).numpy()
        gt_scores = torch.cat(self.gt_scores_list, dim=0).numpy()
        has_gt = torch.cat(self.has_gt_list, dim=0).numpy()

        num_samples = len(pred_coords)
        mask = has_gt == 1
        num_with_gt = int(mask.sum())

        results = {
            "summary": {},
            "iou": {},
            "recall_at_thresholds": {},
            "per_corner": {},
            "coordinate_range": {},
            "by_difficulty": {},
            "score_classification": {},
            "worst_samples": [],
            "best_samples": [],
        }

        if num_with_gt == 0:
            results["summary"]["error"] = "No samples with ground truth"
            return results

        pred_coords_valid = pred_coords[mask]
        gt_coords_valid = gt_coords[mask]

        # ===============================
        # 1. IoU Analysis
        # ===============================
        ious = np.array([
            compute_polygon_iou(pred_coords_valid[i], gt_coords_valid[i])
            for i in range(num_with_gt)
        ])

        results["iou"] = {
            "mean": float(np.mean(ious)),
            "median": float(np.median(ious)),
            "std": float(np.std(ious)),
            "min": float(np.min(ious)),
            "max": float(np.max(ious)),
            "percentiles": {
                "p5": float(np.percentile(ious, 5)),
                "p10": float(np.percentile(ious, 10)),
                "p25": float(np.percentile(ious, 25)),
                "p50": float(np.percentile(ious, 50)),
                "p75": float(np.percentile(ious, 75)),
                "p90": float(np.percentile(ious, 90)),
                "p95": float(np.percentile(ious, 95)),
                "p99": float(np.percentile(ious, 99)),
            },
            "histogram": {
                f"{int(t*100)}-{int((t+0.1)*100)}": float(((ious >= t) & (ious < t + 0.1)).sum() / num_with_gt * 100)
                for t in np.arange(0.0, 1.0, 0.1)
            }
        }

        # ===============================
        # 2. Recall at IoU Thresholds
        # ===============================
        for thresh in self.IOU_THRESHOLDS:
            key = f"R@{int(thresh*100)}"
            count = (ious >= thresh).sum()
            results["recall_at_thresholds"][key] = {
                "recall": float(count / num_with_gt),
                "count": int(count),
                "total": num_with_gt,
            }

        # ===============================
        # 3. Per-Corner Analysis
        # ===============================
        all_corner_dists = []  # [num_samples, 4]
        for i in range(num_with_gt):
            dists = compute_per_corner_distance(pred_coords_valid[i], gt_coords_valid[i])
            all_corner_dists.append(dists)
        all_corner_dists = np.array(all_corner_dists)  # [N, 4]

        for c, name in enumerate(self.CORNER_NAMES):
            corner_dists = all_corner_dists[:, c]
            corner_dists_px = corner_dists * self.img_size
            results["per_corner"][name] = {
                "mean_dist_normalized": float(np.mean(corner_dists)),
                "mean_dist_px": float(np.mean(corner_dists_px)),
                "median_dist_px": float(np.median(corner_dists_px)),
                "std_dist_px": float(np.std(corner_dists_px)),
                "max_dist_px": float(np.max(corner_dists_px)),
                "p90_dist_px": float(np.percentile(corner_dists_px, 90)),
                "p95_dist_px": float(np.percentile(corner_dists_px, 95)),
                "within_5px": float((corner_dists_px <= 5).sum() / num_with_gt * 100),
                "within_10px": float((corner_dists_px <= 10).sum() / num_with_gt * 100),
                "within_20px": float((corner_dists_px <= 20).sum() / num_with_gt * 100),
            }

        # Mean corner distance (all corners combined)
        mean_corner_dist = all_corner_dists.mean(axis=1)  # [N]
        mean_corner_dist_px = mean_corner_dist * self.img_size
        results["per_corner"]["MEAN"] = {
            "mean_dist_px": float(np.mean(mean_corner_dist_px)),
            "median_dist_px": float(np.median(mean_corner_dist_px)),
            "p90_dist_px": float(np.percentile(mean_corner_dist_px, 90)),
            "p95_dist_px": float(np.percentile(mean_corner_dist_px, 95)),
        }

        # ===============================
        # 4. Coordinate Range Analysis (Border Detection)
        # ===============================
        pred_min = pred_coords_valid.min()
        pred_max = pred_coords_valid.max()
        gt_min = gt_coords_valid.min()
        gt_max = gt_coords_valid.max()

        # Check for border shrinkage (sigmoid issue)
        results["coordinate_range"] = {
            "pred_min": float(pred_min),
            "pred_max": float(pred_max),
            "gt_min": float(gt_min),
            "gt_max": float(gt_max),
            "pred_range": float(pred_max - pred_min),
            "gt_range": float(gt_max - gt_min),
            "pred_near_0_count": int((pred_coords_valid < 0.05).sum()),
            "pred_near_1_count": int((pred_coords_valid > 0.95).sum()),
            "gt_near_0_count": int((gt_coords_valid < 0.05).sum()),
            "gt_near_1_count": int((gt_coords_valid > 0.95).sum()),
            "border_shrinkage_detected": bool(pred_min > 0.05 or pred_max < 0.95),
        }

        # Per-coordinate statistics
        coord_names = ["x_TL", "y_TL", "x_TR", "y_TR", "x_BR", "y_BR", "x_BL", "y_BL"]
        results["coordinate_range"]["per_coord"] = {}
        for i, name in enumerate(coord_names):
            pred_vals = pred_coords_valid[:, i]
            gt_vals = gt_coords_valid[:, i]
            results["coordinate_range"]["per_coord"][name] = {
                "pred_mean": float(np.mean(pred_vals)),
                "pred_std": float(np.std(pred_vals)),
                "pred_min": float(np.min(pred_vals)),
                "pred_max": float(np.max(pred_vals)),
                "gt_mean": float(np.mean(gt_vals)),
                "gt_std": float(np.std(gt_vals)),
                "bias": float(np.mean(pred_vals) - np.mean(gt_vals)),
            }

        # ===============================
        # 5. Difficulty Breakdown (by document area)
        # ===============================
        gt_areas = np.array([compute_quad_area(gt_coords_valid[i]) for i in range(num_with_gt)])

        # Define difficulty thresholds based on area
        area_thresholds = {
            "small": (0.0, 0.15),      # < 15% of image
            "medium": (0.15, 0.40),    # 15-40% of image
            "large": (0.40, 1.0),      # > 40% of image
        }

        for diff_name, (area_min, area_max) in area_thresholds.items():
            diff_mask = (gt_areas >= area_min) & (gt_areas < area_max)
            if diff_mask.sum() == 0:
                results["by_difficulty"][diff_name] = {"count": 0}
                continue

            diff_ious = ious[diff_mask]
            diff_dists = mean_corner_dist_px[diff_mask]

            results["by_difficulty"][diff_name] = {
                "count": int(diff_mask.sum()),
                "percentage": float(diff_mask.sum() / num_with_gt * 100),
                "mean_iou": float(np.mean(diff_ious)),
                "median_iou": float(np.median(diff_ious)),
                "R@90": float((diff_ious >= 0.90).sum() / diff_mask.sum() * 100),
                "R@95": float((diff_ious >= 0.95).sum() / diff_mask.sum() * 100),
                "mean_corner_error_px": float(np.mean(diff_dists)),
                "mean_area": float(np.mean(gt_areas[diff_mask])),
            }

        # ===============================
        # 6. Score Classification Metrics
        # ===============================
        pred_probs = 1 / (1 + np.exp(-pred_scores))  # Sigmoid
        pred_binary = (pred_probs >= 0.5).astype(int)
        gt_binary = gt_scores.astype(int)

        # Confusion matrix
        tp = ((pred_binary == 1) & (gt_binary == 1)).sum()
        fp = ((pred_binary == 1) & (gt_binary == 0)).sum()
        tn = ((pred_binary == 0) & (gt_binary == 0)).sum()
        fn = ((pred_binary == 0) & (gt_binary == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / num_samples if num_samples > 0 else 0.0

        results["score_classification"] = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix": {
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            },
            "num_positive_gt": int(gt_binary.sum()),
            "num_negative_gt": int((1 - gt_binary).sum()),
            "num_positive_pred": int(pred_binary.sum()),
        }

        # ===============================
        # 7. Worst and Best Samples
        # ===============================
        sorted_indices = np.argsort(ious)

        # Worst 10 samples
        worst_indices = sorted_indices[:10]
        for idx in worst_indices:
            sample = {
                "index": int(idx),
                "iou": float(ious[idx]),
                "corner_error_px": float(mean_corner_dist_px[idx]),
                "gt_area": float(compute_quad_area(gt_coords_valid[idx])),
            }
            if self.image_names_list:
                # Find original index in full dataset
                valid_indices = np.where(mask)[0]
                orig_idx = valid_indices[idx]
                if orig_idx < len(self.image_names_list):
                    sample["image_name"] = self.image_names_list[orig_idx]
            results["worst_samples"].append(sample)

        # Best 10 samples
        best_indices = sorted_indices[-10:][::-1]
        for idx in best_indices:
            sample = {
                "index": int(idx),
                "iou": float(ious[idx]),
                "corner_error_px": float(mean_corner_dist_px[idx]),
                "gt_area": float(compute_quad_area(gt_coords_valid[idx])),
            }
            results["best_samples"].append(sample)

        # ===============================
        # Summary
        # ===============================
        results["summary"] = {
            "num_samples": num_samples,
            "num_with_gt": num_with_gt,
            "img_size": self.img_size,
            "mean_iou": float(np.mean(ious)),
            "median_iou": float(np.median(ious)),
            "R@90": float((ious >= 0.90).sum() / num_with_gt * 100),
            "R@95": float((ious >= 0.95).sum() / num_with_gt * 100),
            "mean_corner_error_px": float(np.mean(mean_corner_dist_px)),
            "score_accuracy": float(accuracy),
            "score_f1": float(f1),
        }

        return results


# ============================================================================
# Main Evaluation
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Detailed DocCornerNet Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--data_root", type=str, default="/workspace/doc-scanner-dataset-labeled")
    parser.add_argument("--split", type=str, default="val_with_negative_v2", help="Split name (without .txt)")
    parser.add_argument("--negative_image_dir", type=str, default="images-negative",
                        help="Directory for negative images (relative to data_root)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save results JSON")
    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def print_section(title: str):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_results(results: dict, img_size: int):
    """Pretty print evaluation results."""

    # Summary
    print_section("SUMMARY")
    s = results["summary"]
    print(f"  Samples:              {s['num_samples']:,} total, {s['num_with_gt']:,} with GT")
    print(f"  Image size:           {s['img_size']}px")
    print()
    print(f"  Mean IoU:             {s['mean_iou']:.4f} ({s['mean_iou']*100:.2f}%)")
    print(f"  Median IoU:           {s['median_iou']:.4f} ({s['median_iou']*100:.2f}%)")
    print(f"  R@90:                 {s['R@90']:.2f}%")
    print(f"  R@95:                 {s['R@95']:.2f}%")
    print(f"  Mean Corner Error:    {s['mean_corner_error_px']:.2f}px")
    print(f"  Score Accuracy:       {s['score_accuracy']*100:.2f}%")
    print(f"  Score F1:             {s['score_f1']:.4f}")

    # IoU Details
    print_section("IoU ANALYSIS")
    iou = results["iou"]
    print(f"  Mean:    {iou['mean']:.4f}   Std:    {iou['std']:.4f}")
    print(f"  Median:  {iou['median']:.4f}   Min:    {iou['min']:.4f}   Max: {iou['max']:.4f}")
    print()
    print("  Percentiles:")
    p = iou["percentiles"]
    print(f"    P5={p['p5']:.3f}  P10={p['p10']:.3f}  P25={p['p25']:.3f}  P50={p['p50']:.3f}")
    print(f"    P75={p['p75']:.3f}  P90={p['p90']:.3f}  P95={p['p95']:.3f}  P99={p['p99']:.3f}")
    print()
    print("  IoU Histogram (% of samples):")
    for bucket, pct in iou["histogram"].items():
        bar = "█" * int(pct / 2)
        print(f"    {bucket}%: {bar} {pct:.1f}%")

    # Recall at Thresholds
    print_section("RECALL @ IoU THRESHOLDS")
    for key, data in results["recall_at_thresholds"].items():
        bar = "█" * int(data['recall'] * 40)
        print(f"  {key}:  {bar} {data['recall']*100:.2f}% ({data['count']}/{data['total']})")

    # Per-Corner Analysis
    print_section("PER-CORNER ANALYSIS (in pixels)")
    print(f"  {'Corner':<8} {'Mean':>8} {'Median':>8} {'Std':>8} {'P90':>8} {'P95':>8} {'<5px':>8} {'<10px':>8}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for corner in ["TL", "TR", "BR", "BL"]:
        c = results["per_corner"][corner]
        print(f"  {corner:<8} {c['mean_dist_px']:>8.2f} {c['median_dist_px']:>8.2f} {c['std_dist_px']:>8.2f} "
              f"{c['p90_dist_px']:>8.2f} {c['p95_dist_px']:>8.2f} {c['within_5px']:>7.1f}% {c['within_10px']:>7.1f}%")
    m = results["per_corner"]["MEAN"]
    print(f"  {'MEAN':<8} {m['mean_dist_px']:>8.2f} {m['median_dist_px']:>8.2f} {'---':>8} "
          f"{m['p90_dist_px']:>8.2f} {m['p95_dist_px']:>8.2f}")

    # Coordinate Range Analysis
    print_section("COORDINATE RANGE ANALYSIS")
    cr = results["coordinate_range"]
    print(f"  Prediction range:     [{cr['pred_min']:.4f}, {cr['pred_max']:.4f}]")
    print(f"  Ground truth range:   [{cr['gt_min']:.4f}, {cr['gt_max']:.4f}]")
    print(f"  Predictions near 0 (<0.05): {cr['pred_near_0_count']:,}")
    print(f"  Predictions near 1 (>0.95): {cr['pred_near_1_count']:,}")
    if cr['border_shrinkage_detected']:
        print(f"  ⚠️  BORDER SHRINKAGE DETECTED (predictions avoid 0 and 1)")
    else:
        print(f"  ✓  No border shrinkage detected")

    # Difficulty Breakdown
    print_section("BREAKDOWN BY DOCUMENT SIZE")
    print(f"  {'Size':<10} {'Count':>8} {'%':>8} {'Mean IoU':>10} {'R@90':>8} {'R@95':>8} {'Err(px)':>10}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*10}")
    for diff_name in ["small", "medium", "large"]:
        d = results["by_difficulty"][diff_name]
        if d["count"] == 0:
            print(f"  {diff_name:<10} {'0':>8} {'0.0%':>8} {'---':>10} {'---':>8} {'---':>8} {'---':>10}")
        else:
            print(f"  {diff_name:<10} {d['count']:>8} {d['percentage']:>7.1f}% {d['mean_iou']:>10.4f} "
                  f"{d['R@90']:>7.1f}% {d['R@95']:>7.1f}% {d['mean_corner_error_px']:>10.2f}")

    # Score Classification
    print_section("SCORE CLASSIFICATION (Document Detection)")
    sc = results["score_classification"]
    print(f"  Accuracy:   {sc['accuracy']*100:.2f}%")
    print(f"  Precision:  {sc['precision']*100:.2f}%")
    print(f"  Recall:     {sc['recall']*100:.2f}%")
    print(f"  F1 Score:   {sc['f1']:.4f}")
    print()
    print("  Confusion Matrix:")
    cm = sc["confusion_matrix"]
    print(f"                 Predicted")
    print(f"                 Pos    Neg")
    print(f"    Actual Pos   {cm['tp']:>5}  {cm['fn']:>5}")
    print(f"    Actual Neg   {cm['fp']:>5}  {cm['tn']:>5}")

    # Worst Samples
    print_section("WORST 10 SAMPLES (Lowest IoU)")
    print(f"  {'#':<4} {'IoU':>8} {'Error(px)':>10} {'GT Area':>10} {'Image':<30}")
    print(f"  {'-'*4} {'-'*8} {'-'*10} {'-'*10} {'-'*30}")
    for i, sample in enumerate(results["worst_samples"]):
        img_name = sample.get("image_name", f"sample_{sample['index']}")
        print(f"  {i+1:<4} {sample['iou']:>8.4f} {sample['corner_error_px']:>10.2f} "
              f"{sample['gt_area']:>10.4f} {img_name:<30}")

    print()
    print("=" * 70)


def main():
    args = parse_args()
    device = get_device(args.device)

    print(f"\n{'='*70}")
    print(f"  DETAILED DOCCORNERNET EVALUATION")
    print(f"{'='*70}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Dataset:    {args.data_root}")
    print(f"  Split:      {args.split}")
    print(f"  Device:     {device}")

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    img_size = config.get("img_size", 320)
    width_mult = config.get("width_mult", 1.0)
    coord_activation = config.get("coord_activation", "sigmoid")

    print(f"\n  Model Config:")
    print(f"    img_size:          {img_size}")
    print(f"    width_mult:        {width_mult}")
    print(f"    coord_activation:  {coord_activation}")

    # Create model
    model = create_model(
        img_size=img_size,
        width_mult=width_mult,
        pretrained=False,
        coord_activation=coord_activation,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"    parameters:        {model.get_num_params():,}")

    # Load dataset
    data_root = Path(args.data_root)
    split_file = data_root / f"{args.split}.txt"

    if not split_file.exists():
        print(f"\nError: Split file not found: {split_file}")
        return

    # Check for negative images directory
    negative_image_root = data_root / args.negative_image_dir
    neg_root_str = str(negative_image_root) if negative_image_root.exists() else None
    if neg_root_str:
        print(f"    negative_images:   {negative_image_root}")

    dataset = DocDataset(
        image_root=str(data_root / "images"),
        label_root=str(data_root / "labels"),
        split_file=str(split_file),
        img_size=img_size,
        augment=False,
        negative_image_root=neg_root_str,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"\n  Dataset: {len(dataset):,} samples")

    # Evaluate
    metrics = DetailedMetrics(img_size=img_size)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["image"].to(device)
            coords_gt = batch["coords"]
            score_gt = batch["score"]
            has_label = batch["has_label"]
            image_names = batch.get("image_name", None)

            coords_pred, score_pred = model(images)

            metrics.update(
                coords_pred.cpu(),
                coords_gt,
                score_pred.cpu(),
                score_gt,
                has_label,
                image_names,
            )

    results = metrics.compute()

    # Print results
    print_results(results, img_size)

    # Save results
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"eval_{args.split}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to: {output_file}")


if __name__ == "__main__":
    main()
