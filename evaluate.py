"""
Evaluate DocCornerNet model on test set.

Usage:
    python evaluate.py --checkpoint checkpoints/doccornernet_v2/best.pth --data_root ../doc-scanner-dataset-labeled
"""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from model import create_model
from dataset import DocDataset, IMAGENET_MEAN, IMAGENET_STD
from metrics import ValidationMetrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DocCornerNet on test set")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="../doc-scanner-dataset-labeled",
        help="Path to dataset root",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto, cpu, cuda, mps)",
    )
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


def main():
    args = parse_args()
    device = get_device(args.device)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    img_size = config.get("img_size", 224)
    width_mult = config.get("width_mult", 1.0)

    print(f"Model config: img_size={img_size}, width_mult={width_mult}")

    # Create model
    model = create_model(
        img_size=img_size,
        width_mult=width_mult,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Model parameters: {model.get_num_params():,}")

    # Load dataset
    data_root = Path(args.data_root)
    split_file = data_root / f"{args.split}.txt"

    if not split_file.exists():
        print(f"Error: Split file not found: {split_file}")
        return

    dataset = DocDataset(
        image_root=str(data_root / "images"),
        label_root=str(data_root / "labels"),
        split_file=str(split_file),
        img_size=img_size,
        augment=False,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print(f"Dataset size: {len(dataset)} samples")
    print()

    # Evaluate
    metrics = ValidationMetrics()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["image"].to(device)
            coords_gt = batch["coords"]
            score_gt = batch["score"]
            has_label = batch["has_label"]

            coords_pred, score_pred = model(images)

            metrics.update(
                coords_pred.cpu(),
                coords_gt,
                score_pred.cpu(),
                score_gt,
                has_label,
            )

    results = metrics.compute()

    # Print results
    print()
    print("=" * 60)
    print(f"  EVALUATION RESULTS ({args.split.upper()} SET)")
    print("=" * 60)
    print()
    print(f"  Samples:           {results['num_samples']:,}")
    print(f"  With GT:           {results['num_with_gt']:,}")
    print()
    print("  IoU Metrics:")
    print(f"    Mean IoU:        {results['mean_iou']:.4f} ({results['mean_iou']*100:.2f}%)")
    print(f"    Median IoU:      {results['median_iou']:.4f} ({results['median_iou']*100:.2f}%)")
    print()
    print("  Corner Error:")
    print(f"    Mean Error:      {results['mean_corner_error']:.4f} (normalized)")
    print(f"    Error (px):      {results['corner_error_px']:.2f} px (at {img_size}px)")
    print()
    print("  Recall @ IoU Threshold:")
    print(f"    Recall@50:       {results['recall_50']*100:.2f}%")
    print(f"    Recall@75:       {results['recall_75']*100:.2f}%")
    print(f"    Recall@90:       {results['recall_90']*100:.2f}%")
    print()
    print("  Corner Distance Percentiles (normalized):")
    print(f"    P50 (median):    {results['corner_dist_p50']:.4f}")
    print(f"    P90:             {results['corner_dist_p90']:.4f}")
    print(f"    P95:             {results['corner_dist_p95']:.4f}")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
