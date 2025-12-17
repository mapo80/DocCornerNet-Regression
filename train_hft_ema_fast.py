"""
HFT-EMA FAST - Hard-Mining Fine-Tune with EMA (Optimized Version)

Key optimizations:
1. Pre-cache all images in RAM as tensors (one-time cost)
2. No Shapely IoU - use corner error only (vectorized numpy)
3. Update weights every 5 epochs (not 2)
4. Larger batch sizes for inference passes

Usage:
    python train_hft_ema_fast.py \
        --checkpoint /workspace/checkpoints/doccornernet_v5_wing/best.pth \
        --data_root /workspace/doc-scanner-dataset-labeled \
        --experiment_name hft_ema_v1
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm

from model import create_model
from dataset import DocDataset
from train import DetectionLoss


# ============================================================================
# EMA Model
# ============================================================================

class EMAModel:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {name: param.data.clone() for name, param in model.named_parameters() if param.requires_grad}
        self.backup = {}

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


# ============================================================================
# Fast Dataset Cache - Load ALL images into RAM
# ============================================================================

class CachedDataset:
    """
    Pre-loads ALL images and labels into RAM for maximum speed.
    ~7GB for 23k images at 320x320.
    """

    def __init__(self, dataset: DocDataset, device: torch.device):
        self.device = device
        self.N = len(dataset)

        print(f"  Caching {self.N} samples into RAM...")

        # Pre-allocate tensors
        img_size = dataset.img_size
        self.images = torch.zeros((self.N, 3, img_size, img_size), dtype=torch.float32)
        self.coords = torch.zeros((self.N, 8), dtype=torch.float32)
        self.scores = torch.zeros((self.N,), dtype=torch.float32)
        self.has_label = torch.zeros((self.N,), dtype=torch.bool)

        # Load with multiple workers
        loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=8,
            pin_memory=False,
        )

        idx = 0
        for batch in tqdm(loader, desc="Caching to RAM", leave=False):
            bs = batch["image"].shape[0]
            self.images[idx:idx+bs] = batch["image"]
            self.coords[idx:idx+bs] = batch["coords"]
            self.scores[idx:idx+bs] = batch["score"]
            self.has_label[idx:idx+bs] = batch["has_label"].bool()
            idx += bs

        print(f"  Cached {self.N} samples ({self.images.element_size() * self.images.nelement() / 1e9:.2f} GB)")

    def get_batches(self, batch_size: int, indices: torch.Tensor = None, shuffle: bool = True):
        """Generator that yields batches from RAM."""
        if indices is None:
            indices = torch.arange(self.N)

        if shuffle:
            indices = indices[torch.randperm(len(indices))]

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            yield {
                "image": self.images[batch_idx].to(self.device, non_blocking=True),
                "coords": self.coords[batch_idx],
                "score": self.scores[batch_idx],
                "has_label": self.has_label[batch_idx],
            }

    def get_weighted_batches(self, batch_size: int, weights: np.ndarray, num_samples: int):
        """Generator with weighted sampling."""
        sampler = WeightedRandomSampler(
            torch.from_numpy(weights),
            num_samples=num_samples,
            replacement=True,
        )
        indices = torch.tensor(list(sampler))

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            yield {
                "image": self.images[batch_idx].to(self.device, non_blocking=True),
                "coords": self.coords[batch_idx],
                "score": self.scores[batch_idx],
                "has_label": self.has_label[batch_idx],
            }


# ============================================================================
# Fast Hardness Computation (NO Shapely - corner error only)
# ============================================================================

def compute_corner_errors_fast(model: nn.Module, cache: CachedDataset, img_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute corner errors for all samples. Pure numpy, no Shapely.
    Returns (corner_errs, scores) arrays.
    """
    model.eval()
    device = cache.device
    N = cache.N

    all_errs = np.zeros(N, dtype=np.float32)
    all_scores = np.zeros(N, dtype=np.float32)

    idx = 0
    batch_size = 256  # Large batch for inference

    with torch.no_grad():
        for start in tqdm(range(0, N, batch_size), desc="Computing errors", leave=False):
            end = min(start + batch_size, N)
            images = cache.images[start:end].to(device, non_blocking=True)

            coords_pred, score_pred = model(images)
            coords_pred = coords_pred.cpu().numpy()
            scores = torch.sigmoid(score_pred).cpu().numpy().flatten()

            # Ground truth
            coords_gt = cache.coords[start:end].numpy()
            has_label = cache.has_label[start:end].numpy()

            # Vectorized corner error
            bs = end - start
            pred_pts = coords_pred.reshape(bs, 4, 2) * img_size
            gt_pts = coords_gt.reshape(bs, 4, 2) * img_size
            dists = np.sqrt(np.sum((pred_pts - gt_pts) ** 2, axis=2))
            mean_errs = np.mean(dists, axis=1)

            # Store
            all_errs[start:end] = np.where(has_label, mean_errs, 0.0)
            all_scores[start:end] = scores

    return all_errs, all_scores


def compute_hardness_weights_fast(
    corner_errs: np.ndarray,
    scores: np.ndarray,
    has_gt: np.ndarray,
    max_weight: float = 12.0,
) -> np.ndarray:
    """
    Compute weights based on corner error only (no IoU).

    Corner error to IoU approximation:
    - err < 3px  → ~IoU 0.98+ → weight 1 (easy)
    - err 3-5px  → ~IoU 0.95  → weight 4
    - err 5-8px  → ~IoU 0.90  → weight 7
    - err 8-12px → ~IoU 0.85  → weight 10
    - err > 12px → bad        → weight 12
    """
    N = len(corner_errs)
    weights = np.ones(N, dtype=np.float32)

    pos = has_gt

    # Tiered weights based on corner error
    weights[pos & (corner_errs >= 3.0)] = 4.0
    weights[pos & (corner_errs >= 5.0)] = 7.0
    weights[pos & (corner_errs >= 8.0)] = 10.0
    weights[pos & (corner_errs >= 12.0)] = 12.0

    # Overconfident penalty
    overconfident = pos & (scores >= 0.8) & (corner_errs >= 10.0)
    weights[overconfident] = max_weight

    return np.clip(weights, 1.0, max_weight)


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(
    model: nn.Module,
    cache: CachedDataset,
    weights: np.ndarray,
    optimizer: optim.Optimizer,
    loss_fn: DetectionLoss,
    ema: EMAModel,
    batch_size: int,
    grad_clip: float = 1.0,
) -> Dict:
    """Train one epoch with weighted sampling from RAM cache."""
    model.train()

    total_loss = 0.0
    num_batches = 0

    num_samples = cache.N  # Full epoch

    pbar = tqdm(
        cache.get_weighted_batches(batch_size, weights, num_samples),
        total=num_samples // batch_size,
        desc="Training",
        leave=False,
    )

    for batch in pbar:
        optimizer.zero_grad()

        coords_pred, score_pred = model(batch["image"])

        losses = loss_fn(
            coords_pred, score_pred,
            batch["coords"], batch["score"], batch["has_label"]
        )

        losses["total"].backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        ema.update(model)

        total_loss += losses["total"].item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{losses["total"].item():.4f}'})

    return {"loss": total_loss / max(num_batches, 1)}


def validate(
    model: nn.Module,
    cache: CachedDataset,
    loss_fn: DetectionLoss,
    img_size: int,
    batch_size: int = 128,
) -> Dict:
    """Validate and compute metrics."""
    model.eval()

    total_loss = 0.0
    num_batches = 0

    all_errs = []
    all_has_gt = []

    with torch.no_grad():
        for batch in cache.get_batches(batch_size, shuffle=False):
            coords_pred, score_pred = model(batch["image"])

            # Clamp for metrics
            coords_clamped = torch.clamp(coords_pred, 0.0, 1.0)

            losses = loss_fn(
                coords_clamped, score_pred,
                batch["coords"], batch["score"], batch["has_label"]
            )
            total_loss += losses["total"].item()
            num_batches += 1

            # Corner errors
            coords_np = coords_clamped.cpu().numpy()
            gt_np = batch["coords"].numpy()
            has_label = batch["has_label"].numpy()

            bs = len(coords_np)
            pred_pts = coords_np.reshape(bs, 4, 2) * img_size
            gt_pts = gt_np.reshape(bs, 4, 2) * img_size
            dists = np.sqrt(np.sum((pred_pts - gt_pts) ** 2, axis=2))
            mean_errs = np.mean(dists, axis=1)

            for i in range(bs):
                if has_label[i]:
                    all_errs.append(mean_errs[i])
                all_has_gt.append(has_label[i])

    all_errs = np.array(all_errs)
    num_pos = len(all_errs)

    # Approximate IoU from corner error (inverse relationship)
    # err=0 → IoU~1.0, err=3 → IoU~0.95, err=6 → IoU~0.90, err=10 → IoU~0.85
    approx_iou = np.clip(1.0 - all_errs / 50.0, 0.0, 1.0)

    return {
        "loss": total_loss / max(num_batches, 1),
        "mean_err_px": float(np.mean(all_errs)) if num_pos > 0 else 0,
        "median_err_px": float(np.median(all_errs)) if num_pos > 0 else 0,
        "p90_err_px": float(np.percentile(all_errs, 90)) if num_pos > 0 else 0,
        "p95_err_px": float(np.percentile(all_errs, 95)) if num_pos > 0 else 0,
        "pct_under_3px": float((all_errs < 3.0).sum() / num_pos * 100) if num_pos > 0 else 0,
        "pct_under_5px": float((all_errs < 5.0).sum() / num_pos * 100) if num_pos > 0 else 0,
        "r90_approx": float((approx_iou >= 0.90).sum() / num_pos * 100) if num_pos > 0 else 0,
        "r95_approx": float((approx_iou >= 0.95).sum() / num_pos * 100) if num_pos > 0 else 0,
        "num_samples": num_pos,
    }


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="HFT-EMA Fast Training")

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--train_split", type=str, default="train_with_negative_v2.txt")
    parser.add_argument("--val_split", type=str, default="val_with_negative_v2.txt")
    parser.add_argument("--negative_image_dir", type=str, default="images-negative")

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--experiment_name", type=str, default="hft_ema_fast")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=50)

    parser.add_argument("--lr_backbone", type=float, default=2e-5)
    parser.add_argument("--lr_head", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--update_weights_every", type=int, default=5)
    parser.add_argument("--max_weight", type=float, default=12.0)

    parser.add_argument("--scheduler_patience", type=int, default=5)
    parser.add_argument("--early_stopping_patience", type=int, default=15)

    parser.add_argument("--lambda_geometry", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--device", type=str, default="auto")

    return parser.parse_args()


class Colors:
    BOLD = '\033[1m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    DIM = '\033[2m'
    END = '\033[0m'


def main():
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"  {Colors.BOLD}HFT-EMA FAST{Colors.END}")
    print(f"{'='*60}")
    print(f"  Device: {device}")

    # Load checkpoint
    print(f"\n{Colors.DIM}Loading checkpoint...{Colors.END}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    img_size = config.get("img_size", 320)
    width_mult = config.get("width_mult", 1.0)

    # Create model - use clamp to match pretrained weights
    model = create_model(
        img_size=img_size,
        width_mult=width_mult,
        pretrained=False,
        coord_activation="clamp",
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print(f"  img_size: {img_size}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # EMA
    ema = EMAModel(model, decay=args.ema_decay)

    # Output dir
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    data_root = Path(args.data_root)
    neg_root = data_root / args.negative_image_dir
    neg_root_str = str(neg_root) if neg_root.exists() else None

    print(f"\n{Colors.CYAN}Loading datasets...{Colors.END}")

    train_dataset = DocDataset(
        image_root=str(data_root / "images"),
        label_root=str(data_root / "labels"),
        split_file=str(data_root / args.train_split),
        img_size=img_size,
        augment=True,
        negative_image_root=neg_root_str,
    )

    val_dataset = DocDataset(
        image_root=str(data_root / "images"),
        label_root=str(data_root / "labels"),
        split_file=str(data_root / args.val_split),
        img_size=img_size,
        augment=False,
        negative_image_root=neg_root_str,
    )

    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Cache everything to RAM
    print(f"\n{Colors.CYAN}Caching to RAM...{Colors.END}")
    train_cache = CachedDataset(train_dataset, device)
    val_cache = CachedDataset(val_dataset, device)

    # Optimizer with differential LR
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.head.parameters())

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr_backbone},
        {'params': head_params, 'lr': args.lr_head},
    ], weight_decay=args.weight_decay)

    print(f"\n  LR backbone: {args.lr_backbone}, head: {args.lr_head}")

    # Scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=args.scheduler_patience, min_lr=1e-6,
    )

    # Loss
    loss_fn = DetectionLoss(
        lambda_coords=1.0,
        lambda_score=2.0,
        lambda_geometry=args.lambda_geometry,
        use_wing_loss=True,
        img_size=img_size,
    )

    # Initial weights (uniform)
    current_weights = np.ones(train_cache.N, dtype=np.float32)
    has_gt = train_cache.has_label.numpy()

    # Training
    best_err = float('inf')
    epochs_no_improve = 0

    print(f"\n{'='*60}")
    print(f"  {Colors.BOLD}Training{Colors.END}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()

        # Update hardness weights periodically
        if epoch == 1 or (epoch - 1) % args.update_weights_every == 0:
            print(f"{Colors.CYAN}Updating hardness weights...{Colors.END}")
            errs, scores = compute_corner_errors_fast(model, train_cache, img_size)
            current_weights = compute_hardness_weights_fast(errs, scores, has_gt, args.max_weight)

            hard_pct = (current_weights >= 4.0).sum() / has_gt.sum() * 100
            print(f"  Hard samples: {hard_pct:.1f}%")

        # Train
        train_metrics = train_epoch(
            model, train_cache, current_weights,
            optimizer, loss_fn, ema,
            args.batch_size, args.grad_clip
        )

        # Validate with EMA
        ema.apply_shadow(model)
        val_metrics = validate(model, val_cache, loss_fn, img_size)
        ema.restore(model)

        scheduler.step(val_metrics["mean_err_px"])

        # Check improvement
        if val_metrics["mean_err_px"] < best_err:
            best_err = val_metrics["mean_err_px"]
            epochs_no_improve = 0

            # Save best
            ema.apply_shadow(model)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': {'img_size': img_size, 'width_mult': width_mult, 'coord_activation': 'clamp'},
                'metrics': val_metrics,
            }, output_dir / 'best_ema.pth')
            ema.restore(model)

            print(f"  {Colors.GREEN}★ New best: {best_err:.2f}px{Colors.END}")
        else:
            epochs_no_improve += 1

        elapsed = time.time() - t0
        lr = optimizer.param_groups[1]['lr']

        print(f"Epoch {epoch}/{args.num_epochs} ({elapsed:.1f}s) | "
              f"Loss: {train_metrics['loss']:.4f} | "
              f"Err: {val_metrics['mean_err_px']:.2f}px | "
              f"<3px: {val_metrics['pct_under_3px']:.1f}% | "
              f"<5px: {val_metrics['pct_under_5px']:.1f}%")

        if epochs_no_improve >= args.early_stopping_patience:
            print(f"\n{Colors.YELLOW}Early stopping.{Colors.END}")
            break

    print(f"\n{'='*60}")
    print(f"  {Colors.BOLD}Done{Colors.END} - Best error: {Colors.GREEN}{best_err:.2f}px{Colors.END}")
    print(f"  Saved to: {output_dir / 'best_ema.pth'}")
    print()


if __name__ == "__main__":
    main()
