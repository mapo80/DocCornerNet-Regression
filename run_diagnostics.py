"""
Run Diagnostics Script for DocCornerNet Model.

This script runs the full diagnostic analysis on the validation set,
computing per-sample metrics, correlations, and identifying failure modes.

Usage:
    python run_diagnostics.py \
        --checkpoint /workspace/checkpoints/doccornernet_v5_wing/best.pth \
        --data_root /workspace/doc-scanner-dataset-labeled \
        --split val_with_negative_v2 \
        --output_dir /workspace/checkpoints/doccornernet_v5_wing/diagnostics
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from model import create_model
from dataset import DocDataset
from diagnostics import (
    diagnose_sample,
    diagnostics_to_dataframe,
    generate_summary,
    compute_domain_breakdown,
    identify_worst_cases,
    SampleDiagnostics,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run DocCornerNet Diagnostics")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--data_root", type=str, default="/workspace/doc-scanner-dataset-labeled")
    parser.add_argument("--split", type=str, default="val_with_negative_v2", help="Split name (without .txt)")
    parser.add_argument("--negative_image_dir", type=str, default="images-negative")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (1 for diagnostics)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for diagnostics")
    parser.add_argument("--save_worst_images", action="store_true", help="Save visualization of worst cases")
    parser.add_argument("--num_worst", type=int, default=50, help="Number of worst cases to identify")
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


def load_image_original(image_name: str, data_root: Path, negative_image_dir: str) -> np.ndarray:
    """Load original image (not resized) for diagnostics."""
    if image_name.startswith("negative_"):
        image_path = data_root / negative_image_dir / image_name
    else:
        image_path = data_root / "images" / image_name

    image = cv2.imread(str(image_path))
    if image is None:
        # Try with different extension
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            alt_path = image_path.with_suffix(ext)
            if alt_path.exists():
                image = cv2.imread(str(alt_path))
                if image is not None:
                    break

    return image


def draw_quad(image: np.ndarray, quad: np.ndarray, color: tuple, thickness: int = 2, label: str = None):
    """Draw quadrilateral on image."""
    pts = quad.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(image, [pts], True, color, thickness)

    # Draw corner circles
    for i, pt in enumerate(quad):
        cv2.circle(image, tuple(pt.astype(int)), 5, color, -1)

    # Add label
    if label:
        cv2.putText(image, label, (int(quad[0, 0]), int(quad[0, 1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def save_worst_case_visualization(
    sample_diag: SampleDiagnostics,
    data_root: Path,
    negative_image_dir: str,
    output_path: Path,
    img_size: int
):
    """Save visualization of a worst case sample."""
    image = load_image_original(sample_diag.image_name, data_root, negative_image_dir)
    if image is None:
        return

    h, w = image.shape[:2]

    # Convert normalized coords to pixel coords
    pred_quad = np.array([
        [sample_diag.pred_coords[0] * w, sample_diag.pred_coords[1] * h],
        [sample_diag.pred_coords[2] * w, sample_diag.pred_coords[3] * h],
        [sample_diag.pred_coords[4] * w, sample_diag.pred_coords[5] * h],
        [sample_diag.pred_coords[6] * w, sample_diag.pred_coords[7] * h],
    ])

    gt_quad = np.array([
        [sample_diag.gt_coords[0] * w, sample_diag.gt_coords[1] * h],
        [sample_diag.gt_coords[2] * w, sample_diag.gt_coords[3] * h],
        [sample_diag.gt_coords[4] * w, sample_diag.gt_coords[5] * h],
        [sample_diag.gt_coords[6] * w, sample_diag.gt_coords[7] * h],
    ])

    # Draw on image
    vis = image.copy()
    draw_quad(vis, gt_quad, (0, 255, 0), 3, "GT")  # Green
    draw_quad(vis, pred_quad, (0, 0, 255), 2, "Pred")  # Red

    # Add metrics text
    info_text = [
        f"IoU: {sample_diag.iou:.4f}",
        f"Err: {sample_diag.corner_err_mean_px:.2f}px",
        f"Score: {sample_diag.score:.3f}",
        f"Sharp: {sample_diag.sharpness:.1f}",
        f"Contrast: {sample_diag.boundary_contrast:.1f}",
    ]

    y_offset = 30
    for text in info_text:
        cv2.putText(vis, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        y_offset += 25

    # Flags
    flags = []
    if sample_diag.is_small_doc:
        flags.append("SMALL")
    if sample_diag.is_blurry:
        flags.append("BLURRY")
    if sample_diag.is_low_contrast:
        flags.append("LOW_CONTRAST")
    if sample_diag.is_high_perspective:
        flags.append("HIGH_PERSP")

    if flags:
        cv2.putText(vis, " | ".join(flags), (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    cv2.imwrite(str(output_path), vis)


def main():
    args = parse_args()
    device = get_device(args.device)

    print(f"\n{'='*70}")
    print(f"  DOCCORNERNET DIAGNOSTICS")
    print(f"{'='*70}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Dataset:    {args.data_root}")
    print(f"  Split:      {args.split}")
    print(f"  Device:     {device}")
    print(f"  Output:     {args.output_dir}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    img_size = config.get("img_size", 320)
    width_mult = config.get("width_mult", 1.0)
    coord_activation = config.get("coord_activation", "clamp")

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

    # Check for negative images
    negative_image_root = data_root / args.negative_image_dir
    neg_root_str = str(negative_image_root) if negative_image_root.exists() else None

    # Load split file to get image names
    with open(split_file, "r") as f:
        image_list = [line.strip() for line in f if line.strip()]

    print(f"\n  Dataset: {len(image_list)} samples")

    # Create dataset for inference
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
        batch_size=1,
        shuffle=False,
        num_workers=0,  # Single worker for consistent ordering
        pin_memory=True,
    )

    # Run inference and diagnostics
    print(f"\n  Running diagnostics...")
    all_diagnostics = []

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Analyzing")):
            image_name = image_list[idx]

            # Model inference
            images = batch["image"].to(device)
            coords_gt = batch["coords"].numpy()[0]
            score_gt = batch["score"].numpy()[0]
            has_label = batch["has_label"].numpy()[0]

            coords_pred, score_pred = model(images)
            coords_pred = coords_pred.cpu().numpy()[0]
            score_pred = torch.sigmoid(score_pred).cpu().numpy()[0]

            # Load original image for quality analysis
            original_image = load_image_original(image_name, data_root, args.negative_image_dir)

            if original_image is None:
                print(f"  Warning: Could not load {image_name}")
                continue

            # Run diagnostics
            diag = diagnose_sample(
                image=original_image,
                image_name=image_name,
                pred_coords=coords_pred,
                pred_score=float(score_pred),
                gt_coords=coords_gt if has_label else None,
                has_gt=bool(has_label),
                img_size=img_size,
            )
            all_diagnostics.append(diag)

    print(f"\n  Processed {len(all_diagnostics)} samples")

    # Convert to DataFrame
    df = diagnostics_to_dataframe(all_diagnostics)

    # Save per-sample CSV
    csv_path = output_dir / "per_sample.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Generate summary
    summary = generate_summary(df)

    # Domain breakdown
    summary['domain_breakdown'] = compute_domain_breakdown(df)

    # Worst cases
    worst_cases = identify_worst_cases(df, n=args.num_worst)
    summary['worst_cases'] = worst_cases

    # Save summary JSON
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved: {summary_path}")

    # Save worst case visualizations
    if args.save_worst_images:
        worst_dir = output_dir / "worst_cases"
        worst_dir.mkdir(exist_ok=True)

        # Create lookup for diagnostics
        diag_lookup = {d.image_name: d for d in all_diagnostics}

        # Save worst by IoU
        print(f"\n  Saving worst case visualizations...")
        for i, img_name in enumerate(tqdm(worst_cases['worst_by_iou'][:20], desc="Worst IoU")):
            if img_name in diag_lookup:
                save_worst_case_visualization(
                    diag_lookup[img_name],
                    data_root,
                    args.negative_image_dir,
                    worst_dir / f"worst_iou_{i:02d}_{Path(img_name).stem}.jpg",
                    img_size
                )

        # Save overconfident bad localization
        for i, img_name in enumerate(tqdm(worst_cases['overconfident_bad_loc'][:20], desc="Overconfident")):
            if img_name in diag_lookup:
                save_worst_case_visualization(
                    diag_lookup[img_name],
                    data_root,
                    args.negative_image_dir,
                    worst_dir / f"overconfident_{i:02d}_{Path(img_name).stem}.jpg",
                    img_size
                )

    # Print summary
    print(f"\n{'='*70}")
    print(f"  DIAGNOSTICS SUMMARY")
    print(f"{'='*70}")

    if 'localization' in summary:
        loc = summary['localization']
        print(f"\n  Localization (positives only):")
        print(f"    Mean IoU:           {loc['mean_iou']:.4f}")
        print(f"    Median IoU:         {loc['median_iou']:.4f}")
        print(f"    Mean Corner Err:    {loc['mean_corner_err_px']:.2f}px")
        print(f"    P90 Corner Err:     {loc['p90_corner_err_px']:.2f}px")
        print(f"    R@90:               {loc['recall_90']:.2f}%")
        print(f"    R@95:               {loc['recall_95']:.2f}%")

    if 'per_corner' in summary:
        print(f"\n  Per-Corner Error (px):")
        print(f"    {'Corner':<6} {'Mean':>8} {'Median':>8} {'P90':>8}")
        for corner, stats in summary['per_corner'].items():
            print(f"    {corner:<6} {stats['mean_px']:>8.2f} {stats['median_px']:>8.2f} {stats['p90_px']:>8.2f}")

    if 'quality_flags' in summary:
        flags = summary['quality_flags']
        print(f"\n  Quality Flags (% of positives):")
        print(f"    Small doc (<10%):   {flags['pct_small_doc']:.1f}%")
        print(f"    High perspective:   {flags['pct_high_perspective']:.1f}%")
        print(f"    Low contrast:       {flags['pct_low_contrast']:.1f}%")
        print(f"    Blurry:             {flags['pct_blurry']:.1f}%")

    if 'correlations' in summary:
        print(f"\n  Error Correlations:")
        for key, val in summary['correlations'].items():
            direction = "↑" if val > 0 else "↓"
            print(f"    {key}: {val:+.3f} {direction}")

    if 'domain_breakdown' in summary:
        print(f"\n  Domain Breakdown:")
        print(f"    {'Domain':<15} {'Count':>6} {'IoU':>8} {'Err(px)':>8} {'R@90':>6} {'R@95':>6}")
        for domain, stats in sorted(summary['domain_breakdown'].items(), key=lambda x: -x[1]['count']):
            print(f"    {domain:<15} {stats['count']:>6} {stats['mean_iou']:>8.4f} "
                  f"{stats['mean_corner_err_px']:>8.2f} {stats['r90']:>5.1f}% {stats['r95']:>5.1f}%")

    print(f"\n  Worst Cases:")
    print(f"    Overconfident bad loc: {len(worst_cases['overconfident_bad_loc'])} samples")

    print(f"\n{'='*70}")
    print(f"  Output saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
