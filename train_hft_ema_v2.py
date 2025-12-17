"""
HFT-EMA v2 - Hard-Mining Fine-Tune with EMA (Fixed Version)

Fixes from v1:
1. NO RAM caching for train (augment on-the-fly)
2. RAM cache ONLY for val (no augment)
3. TOP-K% hard mining instead of absolute thresholds
4. Initial eval to verify checkpoint loads correctly
5. Clear separation: TRAIN metrics vs VAL_FULL metrics

Usage:
    python train_hft_ema_v2.py \
        --checkpoint checkpoints/best.pth \
        --data_root /path/to/dataset \
        --experiment_name hft_ema_v2
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
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
# Validation Cache (NO augment, static)
# ============================================================================

class ValCache:
    """Cache validation images in RAM (no augment, deterministic)."""

    def __init__(self, dataset: DocDataset, device: torch.device):
        self.device = device
        self.N = len(dataset)

        print(f"  Caching {self.N} val samples...")

        img_size = dataset.img_size
        self.images = torch.zeros((self.N, 3, img_size, img_size), dtype=torch.float32)
        self.coords = torch.zeros((self.N, 8), dtype=torch.float32)
        self.scores = torch.zeros((self.N,), dtype=torch.float32)
        self.has_label = torch.zeros((self.N,), dtype=torch.bool)

        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

        idx = 0
        for batch in tqdm(loader, desc="Caching val", leave=False):
            bs = batch["image"].shape[0]
            self.images[idx:idx+bs] = batch["image"]
            self.coords[idx:idx+bs] = batch["coords"]
            self.scores[idx:idx+bs] = batch["score"]
            self.has_label[idx:idx+bs] = batch["has_label"].bool()
            idx += bs

        print(f"  Val cached: {self.N} samples")


# ============================================================================
# Hardness Computation (TOP-K% based)
# ============================================================================

def compute_errors_on_train(
    model: nn.Module,
    dataset: DocDataset,
    device: torch.device,
    img_size: int,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute corner errors on train set.
    Uses the dataset as-is (should have augment=False for this call).
    Returns (corner_errs, scores, has_gt).
    """
    model.eval()
    N = len(dataset)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    all_errs = np.zeros(N, dtype=np.float32)
    all_scores = np.zeros(N, dtype=np.float32)
    all_has_gt = np.zeros(N, dtype=bool)

    idx = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing train errors", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            coords_gt = batch["coords"].numpy()
            has_label = batch["has_label"].numpy()
            bs = len(images)

            coords_pred, score_pred = model(images)
            coords_pred = coords_pred.cpu().numpy()
            scores = torch.sigmoid(score_pred).cpu().numpy().flatten()

            # Vectorized corner error
            pred_pts = coords_pred.reshape(bs, 4, 2) * img_size
            gt_pts = coords_gt.reshape(bs, 4, 2) * img_size
            dists = np.sqrt(np.sum((pred_pts - gt_pts) ** 2, axis=2))
            mean_errs = np.mean(dists, axis=1)

            all_errs[idx:idx+bs] = np.where(has_label, mean_errs, 0.0)
            all_scores[idx:idx+bs] = scores
            all_has_gt[idx:idx+bs] = has_label.astype(bool)
            idx += bs

    return all_errs, all_scores, all_has_gt


def compute_hardness_weights_topk(
    corner_errs: np.ndarray,
    has_gt: np.ndarray,
    hard_pct: float = 0.25,
    medium_pct: float = 0.25,
    hard_weight: float = 4.0,
    medium_weight: float = 2.0,
) -> np.ndarray:
    """
    Compute weights using TOP-K% strategy.

    - Top hard_pct% errors -> hard_weight
    - Next medium_pct% errors -> medium_weight
    - Rest -> 1.0
    - Negatives -> 1.0
    """
    N = len(corner_errs)
    weights = np.ones(N, dtype=np.float32)

    # Get positive indices and their errors
    pos_indices = np.where(has_gt)[0]
    pos_errs = corner_errs[pos_indices]

    if len(pos_errs) == 0:
        return weights

    # Sort by error (descending)
    sorted_idx = np.argsort(pos_errs)[::-1]

    # Top K% are hard
    n_hard = int(len(pos_errs) * hard_pct)
    n_medium = int(len(pos_errs) * medium_pct)

    hard_indices = pos_indices[sorted_idx[:n_hard]]
    medium_indices = pos_indices[sorted_idx[n_hard:n_hard + n_medium]]

    weights[hard_indices] = hard_weight
    weights[medium_indices] = medium_weight

    return weights


# ============================================================================
# Training & Validation
# ============================================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: DetectionLoss,
    device: torch.device,
    ema: EMAModel,
    grad_clip: float = 1.0,
) -> Dict:
    """Train one epoch."""
    model.train()

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        coords_gt = batch["coords"]
        score_gt = batch["score"]
        has_label = batch["has_label"]

        optimizer.zero_grad()

        coords_pred, score_pred = model(images)

        losses = loss_fn(coords_pred, score_pred, coords_gt, score_gt, has_label)
        losses["total"].backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        ema.update(model)

        total_loss += losses["total"].item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{losses["total"].item():.4f}'})

    return {"loss": total_loss / max(num_batches, 1)}


def validate_full(
    model: nn.Module,
    val_cache: ValCache,
    loss_fn: DetectionLoss,
    device: torch.device,
    img_size: int,
    batch_size: int = 64,
) -> Dict:
    """
    Full validation on cached val set.
    Returns comprehensive metrics.
    """
    model.eval()

    total_loss = 0.0
    num_batches = 0

    all_errs = []

    N = val_cache.N

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)

            images = val_cache.images[start:end].to(device, non_blocking=True)
            coords_gt = val_cache.coords[start:end]
            score_gt = val_cache.scores[start:end]
            has_label = val_cache.has_label[start:end]

            coords_pred, score_pred = model(images)

            losses = loss_fn(coords_pred, score_pred, coords_gt, score_gt, has_label)
            total_loss += losses["total"].item()
            num_batches += 1

            # Corner errors (only for positives)
            coords_np = coords_pred.cpu().numpy()
            gt_np = coords_gt.numpy()
            has_label_np = has_label.numpy()

            bs = len(coords_np)
            pred_pts = coords_np.reshape(bs, 4, 2) * img_size
            gt_pts = gt_np.reshape(bs, 4, 2) * img_size
            dists = np.sqrt(np.sum((pred_pts - gt_pts) ** 2, axis=2))
            mean_errs = np.mean(dists, axis=1)

            for i in range(bs):
                if has_label_np[i]:
                    all_errs.append(mean_errs[i])

    all_errs = np.array(all_errs)
    num_pos = len(all_errs)

    if num_pos == 0:
        return {"loss": total_loss / max(num_batches, 1), "mean_err_px": 0, "num_pos": 0}

    return {
        "loss": total_loss / max(num_batches, 1),
        "mean_err_px": float(np.mean(all_errs)),
        "median_err_px": float(np.median(all_errs)),
        "p90_err_px": float(np.percentile(all_errs, 90)),
        "p95_err_px": float(np.percentile(all_errs, 95)),
        "pct_under_3px": float((all_errs < 3.0).sum() / num_pos * 100),
        "pct_under_5px": float((all_errs < 5.0).sum() / num_pos * 100),
        "pct_under_10px": float((all_errs < 10.0).sum() / num_pos * 100),
        "num_pos": num_pos,
    }


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="HFT-EMA v2 Training")

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--train_split", type=str, default="train_with_negative_v2.txt")
    parser.add_argument("--val_split", type=str, default="val_with_negative_v2.txt")
    parser.add_argument("--negative_image_dir", type=str, default="images-negative")

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--experiment_name", type=str, default="hft_ema_v2")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=4)

    # Learning rates (conservative for fine-tuning)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--lr_head", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    parser.add_argument("--ema_decay", type=float, default=0.999)

    # Hard mining (TOP-K% based)
    parser.add_argument("--update_weights_every", type=int, default=5)
    parser.add_argument("--hard_pct", type=float, default=0.25, help="Top K% considered hard")
    parser.add_argument("--medium_pct", type=float, default=0.25, help="Next K% considered medium")
    parser.add_argument("--hard_weight", type=float, default=3.0)
    parser.add_argument("--medium_weight", type=float, default=1.5)

    parser.add_argument("--scheduler_patience", type=int, default=5)
    parser.add_argument("--early_stopping_patience", type=int, default=10)

    parser.add_argument("--lambda_geometry", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--device", type=str, default="auto")

    return parser.parse_args()


class Colors:
    BOLD = '\033[1m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    RED = '\033[91m'
    DIM = '\033[2m'
    END = '\033[0m'


def main():
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"  {Colors.BOLD}HFT-EMA v2 - Fixed Hard Mining{Colors.END}")
    print(f"{'='*60}")
    print(f"  Device: {device}")

    # Load checkpoint
    print(f"\n{Colors.DIM}Loading checkpoint: {args.checkpoint}{Colors.END}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    img_size = config.get("img_size", 320)
    width_mult = config.get("width_mult", 1.0)

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

    # Train dataset WITH augment (for actual training)
    train_dataset = DocDataset(
        image_root=str(data_root / "images"),
        label_root=str(data_root / "labels"),
        split_file=str(data_root / args.train_split),
        img_size=img_size,
        augment=True,
        negative_image_root=neg_root_str,
    )

    # Train dataset WITHOUT augment (for error measurement / hardness computation)
    train_dataset_no_aug = DocDataset(
        image_root=str(data_root / "images"),
        label_root=str(data_root / "labels"),
        split_file=str(data_root / args.train_split),
        img_size=img_size,
        augment=False,
        negative_image_root=neg_root_str,
    )

    # Val dataset WITHOUT augment
    val_dataset = DocDataset(
        image_root=str(data_root / "images"),
        label_root=str(data_root / "labels"),
        split_file=str(data_root / args.val_split),
        img_size=img_size,
        augment=False,
        negative_image_root=neg_root_str,
    )

    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Cache ONLY validation
    print(f"\n{Colors.CYAN}Caching validation set...{Colors.END}")
    val_cache = ValCache(val_dataset, device)

    # Loss
    loss_fn = DetectionLoss(
        lambda_coords=1.0,
        lambda_score=2.0,
        lambda_geometry=args.lambda_geometry,
        use_wing_loss=True,
        img_size=img_size,
    )

    # =========================================================================
    # INITIAL EVALUATION - Verify checkpoint loaded correctly
    # =========================================================================
    print(f"\n{Colors.CYAN}Initial evaluation (checkpoint verification)...{Colors.END}")
    initial_metrics = validate_full(model, val_cache, loss_fn, device, img_size)

    print(f"\n  {Colors.BOLD}Checkpoint baseline:{Colors.END}")
    print(f"    Mean err: {initial_metrics['mean_err_px']:.2f}px")
    print(f"    Median err: {initial_metrics['median_err_px']:.2f}px")
    print(f"    <3px: {initial_metrics['pct_under_3px']:.1f}%")
    print(f"    <5px: {initial_metrics['pct_under_5px']:.1f}%")
    print(f"    P90 err: {initial_metrics['p90_err_px']:.2f}px")

    if initial_metrics['mean_err_px'] > 10:
        print(f"\n  {Colors.RED}WARNING: Initial error > 10px! Checkpoint may not have loaded correctly.{Colors.END}")

    # Optimizer with differential LR (conservative)
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
        patience=args.scheduler_patience, min_lr=1e-7,
    )

    # Initial weights (uniform)
    current_weights = np.ones(len(train_dataset), dtype=np.float32)

    # Training state
    best_err = initial_metrics['mean_err_px']
    epochs_no_improve = 0

    print(f"\n{'='*60}")
    print(f"  {Colors.BOLD}Training (target: beat {best_err:.2f}px){Colors.END}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()

        # Update hardness weights periodically
        if epoch == 1 or (epoch - 1) % args.update_weights_every == 0:
            print(f"{Colors.CYAN}Updating hardness weights (TOP-K%)...{Colors.END}")

            errs, scores, has_gt = compute_errors_on_train(
                model, train_dataset_no_aug, device, img_size, args.batch_size * 2
            )

            current_weights = compute_hardness_weights_topk(
                errs, has_gt,
                hard_pct=args.hard_pct,
                medium_pct=args.medium_pct,
                hard_weight=args.hard_weight,
                medium_weight=args.medium_weight,
            )

            # Stats
            pos_mask = has_gt
            pos_errs = errs[pos_mask]
            hard_mask = current_weights >= args.hard_weight
            hard_err_mean = pos_errs[hard_mask[pos_mask]].mean() if hard_mask[pos_mask].sum() > 0 else 0

            print(f"  Hard ({args.hard_pct*100:.0f}%): {hard_mask.sum()} samples, mean err: {hard_err_mean:.2f}px")
            print(f"  Train mean err: {pos_errs.mean():.2f}px")

        # Create weighted sampler
        sampler = WeightedRandomSampler(
            torch.from_numpy(current_weights),
            num_samples=len(train_dataset),
            replacement=True,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device, ema, args.grad_clip
        )

        # Validate with EMA on FULL val set
        ema.apply_shadow(model)
        val_metrics = validate_full(model, val_cache, loss_fn, device, img_size)
        ema.restore(model)

        scheduler.step(val_metrics["mean_err_px"])

        # Check improvement
        improved = val_metrics["mean_err_px"] < best_err
        if improved:
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
        else:
            epochs_no_improve += 1

        elapsed = time.time() - t0
        lr = optimizer.param_groups[1]['lr']

        status = f"{Colors.GREEN}★ NEW BEST{Colors.END}" if improved else ""

        print(f"Epoch {epoch}/{args.num_epochs} ({elapsed:.1f}s) | "
              f"Train loss: {train_metrics['loss']:.4f} | "
              f"VAL: {val_metrics['mean_err_px']:.2f}px (med:{val_metrics['median_err_px']:.2f}) | "
              f"<3px: {val_metrics['pct_under_3px']:.1f}% | "
              f"<5px: {val_metrics['pct_under_5px']:.1f}% {status}")

        if epochs_no_improve >= args.early_stopping_patience:
            print(f"\n{Colors.YELLOW}Early stopping after {epochs_no_improve} epochs without improvement.{Colors.END}")
            break

    print(f"\n{'='*60}")
    print(f"  {Colors.BOLD}Done{Colors.END}")
    print(f"  Initial: {initial_metrics['mean_err_px']:.2f}px")
    print(f"  Best:    {Colors.GREEN}{best_err:.2f}px{Colors.END}")
    print(f"  Saved:   {output_dir / 'best_ema.pth'}")
    print()


if __name__ == "__main__":
    main()
