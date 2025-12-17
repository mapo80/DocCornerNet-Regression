"""
Hard-Mining Fine-Tune with EMA (HFT-EMA) Training Script.

This script implements aggressive hard example mining to push R@95 higher
by focusing training on the tail distribution (hardest samples).

Key features:
1. Dynamic hardness weights updated every N epochs
2. EMA model for stable evaluation and checkpointing
3. Differential learning rates (backbone vs head)
4. Linear output during training (no clamp), clamp only for metrics
5. Monitors R@95 instead of mean IoU for best model selection

Usage:
    python train_hft_ema.py \
        --checkpoint checkpoints/best.pth \
        --data_root /path/to/dataset \
        --experiment_name hft_ema_v1
"""

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

from model import create_model, DocCornerNet
from dataset import DocDataset, create_dataloaders
from train import focal_bce_with_logits, wing_loss, geometry_loss, DetectionLoss

try:
    from shapely.geometry import Polygon
    from shapely.validation import make_valid
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False


# ============================================================================
# EMA Model Wrapper
# ============================================================================

class EMAModel:
    """
    Exponential Moving Average of model weights.

    Maintains a shadow copy of model weights that are updated with EMA.
    Use the EMA model for evaluation and checkpointing.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        """
        Args:
            model: The model to track
            decay: EMA decay rate (0.999 or 0.9995 recommended)
        """
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        """Update shadow weights with current model weights."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1.0 - self.decay) * param.data
                )

    def apply_shadow(self, model: nn.Module):
        """Apply shadow weights to model (for evaluation)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model: nn.Module):
        """Restore original weights after evaluation."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self) -> Dict:
        """Get EMA state dict for saving."""
        return {
            'decay': self.decay,
            'shadow': self.shadow,
        }

    def load_state_dict(self, state_dict: Dict):
        """Load EMA state dict."""
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']


# ============================================================================
# Hardness Weight Computation with Caching
# ============================================================================

class HardnessCache:
    """
    Cache GT coordinates and has_label to avoid reloading from dataset.
    Only predictions need to be recomputed each epoch.
    """

    def __init__(self, dataset: DocDataset, device: torch.device, batch_size: int = 128, num_workers: int = 8):
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.N = len(dataset)

        # Pre-extract all GT data (one-time cost)
        print("  Caching GT coordinates...")
        self.coords_gt = np.zeros((self.N, 8), dtype=np.float32)
        self.has_gt = np.zeros(self.N, dtype=bool)

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        idx = 0
        for batch in tqdm(loader, desc="Caching GT", leave=False):
            bs = batch["coords"].shape[0]
            self.coords_gt[idx:idx+bs] = batch["coords"].numpy()
            self.has_gt[idx:idx+bs] = batch["has_label"].numpy().astype(bool)
            idx += bs

        # Store dataset reference for image loading
        self.dataset = dataset

        # Pre-allocate prediction arrays
        self.coords_pred = np.zeros((self.N, 8), dtype=np.float32)
        self.scores = np.zeros(self.N, dtype=np.float32)

    def update_predictions(self, model: nn.Module, img_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Update predictions using current model weights.
        Returns (ious, corner_errs, scores, has_gt).
        """
        model.eval()

        loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

        idx = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc="Updating predictions", leave=False):
                images = batch["image"].to(self.device, non_blocking=True)
                bs = images.shape[0]

                coords_pred, score_pred = model(images)

                self.coords_pred[idx:idx+bs] = coords_pred.cpu().numpy()
                self.scores[idx:idx+bs] = torch.sigmoid(score_pred).cpu().numpy().flatten()
                idx += bs

        # Vectorized IoU and corner error computation
        ious = compute_iou_batch(self.coords_pred, self.coords_gt, self.has_gt)
        corner_errs = compute_corner_error_batch(self.coords_pred, self.coords_gt, self.has_gt, img_size)

        return ious, corner_errs, self.scores, self.has_gt


def compute_iou_batch(pred: np.ndarray, gt: np.ndarray, has_gt: np.ndarray) -> np.ndarray:
    """Vectorized IoU computation for all samples."""
    N = len(pred)
    ious = np.zeros(N, dtype=np.float32)

    if not SHAPELY_AVAILABLE:
        ious[has_gt] = 0.5
        return ious

    for i in range(N):
        if has_gt[i]:
            ious[i] = compute_iou_single(pred[i], gt[i])

    return ious


def compute_corner_error_batch(pred: np.ndarray, gt: np.ndarray, has_gt: np.ndarray, img_size: int) -> np.ndarray:
    """Vectorized corner error computation."""
    N = len(pred)
    errs = np.zeros(N, dtype=np.float32)

    # Reshape to [N, 4, 2]
    pred_pts = pred.reshape(N, 4, 2) * img_size
    gt_pts = gt.reshape(N, 4, 2) * img_size

    # Compute per-corner distances
    dists = np.sqrt(np.sum((pred_pts - gt_pts) ** 2, axis=2))  # [N, 4]
    mean_errs = np.mean(dists, axis=1)  # [N]

    errs[has_gt] = mean_errs[has_gt]
    return errs


def compute_sample_metrics(
    model: nn.Module,
    dataset: DocDataset,
    device: torch.device,
    img_size: int,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-sample metrics for hardness weighting.
    Legacy function - use HardnessCache for better performance.
    """
    model.eval()

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    all_ious = []
    all_errs = []
    all_scores = []
    all_has_gt = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing hardness", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            coords_gt = batch["coords"].numpy()
            has_label = batch["has_label"].numpy()

            coords_pred, score_pred = model(images)
            coords_pred = coords_pred.cpu().numpy()
            score_prob = torch.sigmoid(score_pred).cpu().numpy()

            for i in range(len(images)):
                all_scores.append(score_prob[i])
                all_has_gt.append(bool(has_label[i]))

                if has_label[i]:
                    iou = compute_iou_single(coords_pred[i], coords_gt[i])
                    all_ious.append(iou)
                    err = compute_corner_error(coords_pred[i], coords_gt[i], img_size)
                    all_errs.append(err)
                else:
                    all_ious.append(0.0)
                    all_errs.append(0.0)

    return (
        np.array(all_ious),
        np.array(all_errs),
        np.array(all_scores),
        np.array(all_has_gt),
    )


def compute_iou_single(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute IoU between single pred and gt (8-element arrays)."""
    if not SHAPELY_AVAILABLE:
        return 0.5  # Fallback

    try:
        pred_pts = [(pred[i], pred[i+1]) for i in range(0, 8, 2)]
        gt_pts = [(gt[i], gt[i+1]) for i in range(0, 8, 2)]

        pred_poly = Polygon(pred_pts)
        gt_poly = Polygon(gt_pts)

        if not pred_poly.is_valid:
            pred_poly = make_valid(pred_poly)
        if not gt_poly.is_valid:
            gt_poly = make_valid(gt_poly)

        if pred_poly.is_empty or gt_poly.is_empty:
            return 0.0

        intersection = pred_poly.intersection(gt_poly).area
        union = pred_poly.union(gt_poly).area

        return intersection / union if union > 0 else 0.0
    except Exception:
        return 0.0


def compute_corner_error(pred: np.ndarray, gt: np.ndarray, img_size: int) -> float:
    """Compute mean corner error in pixels."""
    errors = []
    for i in range(4):
        px = pred[2*i] * img_size
        py = pred[2*i + 1] * img_size
        gx = gt[2*i] * img_size
        gy = gt[2*i + 1] * img_size
        errors.append(np.sqrt((px - gx)**2 + (py - gy)**2))
    return np.mean(errors)


def compute_hardness_weights(
    ious: np.ndarray,
    corner_errs: np.ndarray,
    scores: np.ndarray,
    has_gt: np.ndarray,
    max_weight: float = 12.0,
) -> np.ndarray:
    """
    Compute per-sample hardness weights for weighted sampling.

    Weighting scheme:
    - Negatives: w = 1.0
    - Positives base: w = 1.0
    - IoU < 0.95: w += 3.0
    - IoU < 0.90 OR err > 8px: w += 6.0
    - Overconfident bad loc (score >= 0.8 AND err >= 15px): w += 8.0

    Returns:
        weights: [N] sample weights
    """
    N = len(ious)
    weights = np.ones(N, dtype=np.float32)

    # Positives
    pos_mask = has_gt

    # Medium hard: IoU < 0.95
    medium_hard = pos_mask & (ious < 0.95)
    weights[medium_hard] += 3.0

    # Hard: IoU < 0.90 OR corner_err > 8px
    hard = pos_mask & ((ious < 0.90) | (corner_errs > 8.0))
    weights[hard] += 6.0

    # Very hard: overconfident with bad localization
    very_hard = pos_mask & (scores >= 0.8) & (corner_errs >= 15.0)
    weights[very_hard] += 8.0

    # Clamp
    weights = np.clip(weights, 1.0, max_weight)

    return weights


def get_weight_stats(weights: np.ndarray, has_gt: np.ndarray) -> Dict:
    """Get statistics about hardness weights."""
    pos_mask = has_gt
    neg_mask = ~has_gt

    pos_weights = weights[pos_mask]

    stats = {
        'total_samples': len(weights),
        'num_positives': pos_mask.sum(),
        'num_negatives': neg_mask.sum(),
        'mean_pos_weight': float(pos_weights.mean()) if len(pos_weights) > 0 else 0,
        'max_pos_weight': float(pos_weights.max()) if len(pos_weights) > 0 else 0,
        'pct_hard': float((pos_weights >= 4.0).sum() / len(pos_weights) * 100) if len(pos_weights) > 0 else 0,
        'pct_very_hard': float((pos_weights >= 10.0).sum() / len(pos_weights) * 100) if len(pos_weights) > 0 else 0,
    }
    return stats


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch_hft(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: DetectionLoss,
    device: torch.device,
    ema: EMAModel,
    epoch: int,
    num_epochs: int,
    grad_clip: float = 1.0,
) -> Dict:
    """Train for one epoch with hard mining."""
    model.train()

    total_loss = 0.0
    total_coords = 0.0
    total_geometry = 0.0
    total_score = 0.0
    num_batches = 0

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}/{num_epochs} [Train]",
        leave=False,
    )

    for batch in pbar:
        images = batch["image"].to(device)
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

        # Update EMA
        ema.update(model)

        total_loss += losses["total"].item()
        total_coords += losses["coords"].item()
        total_geometry += losses.get("geometry", torch.tensor(0.0)).item() if isinstance(losses.get("geometry"), torch.Tensor) else losses.get("geometry", 0.0)
        total_score += losses["score"].item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{losses["total"].item():.4f}'})

    return {
        "loss": total_loss / num_batches,
        "coords": total_coords / num_batches,
        "geometry": total_geometry / num_batches,
        "score": total_score / num_batches,
    }


def validate_with_recall(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: DetectionLoss,
    device: torch.device,
    img_size: int,
) -> Dict:
    """Validate and compute R@90, R@95, corner error stats."""
    model.eval()

    total_loss = 0.0
    num_batches = 0

    all_ious = []
    all_errs = []
    all_has_gt = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            images = batch["image"].to(device)
            coords_gt = batch["coords"]
            score_gt = batch["score"]
            has_label = batch["has_label"]

            coords_pred, score_pred = model(images)

            # Clamp for metrics (training uses linear output)
            coords_pred_clamped = torch.clamp(coords_pred, 0.0, 1.0)

            losses = loss_fn(coords_pred_clamped, score_pred, coords_gt, score_gt, has_label)
            total_loss += losses["total"].item()
            num_batches += 1

            # Compute per-sample metrics
            coords_pred_np = coords_pred_clamped.cpu().numpy()
            coords_gt_np = coords_gt.numpy()
            has_label_np = has_label.numpy()

            for i in range(len(images)):
                all_has_gt.append(bool(has_label_np[i]))
                if has_label_np[i]:
                    iou = compute_iou_single(coords_pred_np[i], coords_gt_np[i])
                    err = compute_corner_error(coords_pred_np[i], coords_gt_np[i], img_size)
                    all_ious.append(iou)
                    all_errs.append(err)

    all_ious = np.array(all_ious)
    all_errs = np.array(all_errs)

    num_pos = len(all_ious)

    results = {
        "loss": total_loss / num_batches,
        "mean_iou": float(np.mean(all_ious)) if num_pos > 0 else 0,
        "median_iou": float(np.median(all_ious)) if num_pos > 0 else 0,
        "r50": float((all_ious >= 0.50).sum() / num_pos * 100) if num_pos > 0 else 0,
        "r75": float((all_ious >= 0.75).sum() / num_pos * 100) if num_pos > 0 else 0,
        "r90": float((all_ious >= 0.90).sum() / num_pos * 100) if num_pos > 0 else 0,
        "r95": float((all_ious >= 0.95).sum() / num_pos * 100) if num_pos > 0 else 0,
        "mean_err_px": float(np.mean(all_errs)) if num_pos > 0 else 0,
        "median_err_px": float(np.median(all_errs)) if num_pos > 0 else 0,
        "p90_err_px": float(np.percentile(all_errs, 90)) if num_pos > 0 else 0,
        "p95_err_px": float(np.percentile(all_errs, 95)) if num_pos > 0 else 0,
        "num_samples": num_pos,
    }

    return results


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="HFT-EMA Training")

    # Data
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--train_split", type=str, default="train_with_negative_v2.txt")
    parser.add_argument("--val_split", type=str, default="val_with_negative_v2.txt")
    parser.add_argument("--negative_image_dir", type=str, default="images-negative")

    # Checkpoint
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pth to fine-tune")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--experiment_name", type=str, default="hft_ema")

    # Model (loaded from checkpoint, but can override)
    parser.add_argument("--img_size", type=int, default=None, help="Override img_size from checkpoint")

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)

    # Learning rates (differential)
    parser.add_argument("--lr_backbone", type=float, default=2e-5)
    parser.add_argument("--lr_head", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    # EMA
    parser.add_argument("--ema_decay", type=float, default=0.999)

    # Hard mining
    parser.add_argument("--update_weights_every", type=int, default=2, help="Epochs between weight updates")
    parser.add_argument("--max_weight", type=float, default=12.0)

    # Scheduler
    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    parser.add_argument("--scheduler_patience", type=int, default=5)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    # Early stopping
    parser.add_argument("--early_stopping_patience", type=int, default=20)

    # Loss
    parser.add_argument("--use_wing_loss", action="store_true", default=True)
    parser.add_argument("--lambda_geometry", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Device
    parser.add_argument("--device", type=str, default="auto")

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


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'


def main():
    args = parse_args()
    device = get_device(args.device)

    print(f"\n{'='*70}")
    print(f"  {Colors.BOLD}HFT-EMA: Hard-Mining Fine-Tune with EMA{Colors.ENDC}")
    print(f"{'='*70}")

    # Load checkpoint
    print(f"\n{Colors.DIM}Loading checkpoint: {args.checkpoint}{Colors.ENDC}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    img_size = args.img_size or config.get("img_size", 320)
    width_mult = config.get("width_mult", 1.0)

    print(f"  img_size: {img_size}")
    print(f"  width_mult: {width_mult}")

    # Create model with LINEAR output (no clamp in forward for training)
    # We'll clamp manually during validation
    model = create_model(
        img_size=img_size,
        width_mult=width_mult,
        pretrained=False,
        coord_activation="none",  # Linear output for training!
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print(f"  Parameters: {model.get_num_params():,}")
    print(f"  {Colors.YELLOW}coord_activation: none (linear for training){Colors.ENDC}")

    # Initialize EMA
    ema = EMAModel(model, decay=args.ema_decay)
    print(f"  EMA decay: {args.ema_decay}")

    # Create output directory
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data root
    data_root = Path(args.data_root)
    negative_image_root = data_root / args.negative_image_dir
    neg_root_str = str(negative_image_root) if negative_image_root.exists() else None

    # Create train dataset (we'll rebuild loader with weights)
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

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"\n  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # Optimizer with differential LR
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.head.parameters())

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr_backbone},
        {'params': head_params, 'lr': args.lr_head},
    ], weight_decay=args.weight_decay)

    print(f"\n  Optimizer: AdamW")
    print(f"    Backbone LR: {args.lr_backbone}")
    print(f"    Head LR: {args.lr_head}")
    print(f"    Weight decay: {args.weight_decay}")

    # Scheduler monitors R@95
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',  # Maximize R@95
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
        verbose=True,
    )

    # Loss function
    loss_fn = DetectionLoss(
        lambda_coords=1.0,
        lambda_score=2.0,
        lambda_geometry=args.lambda_geometry,
        use_wing_loss=args.use_wing_loss,
        img_size=img_size,
    )

    print(f"\n  Loss: {'Wing' if args.use_wing_loss else 'SmoothL1'} + Geometry({args.lambda_geometry})")

    # Initialize hardness cache (one-time GT caching)
    print(f"\n{Colors.CYAN}Initializing hardness cache...{Colors.ENDC}")
    hardness_cache = HardnessCache(
        train_dataset,
        device,
        batch_size=args.batch_size * 2,  # Larger batch for inference
        num_workers=args.num_workers,
    )

    # Training state
    best_r95 = 0.0
    epochs_without_improvement = 0
    current_weights = None

    # Training loop
    print(f"\n{'='*70}")
    print(f"  {Colors.BOLD}Starting HFT-EMA Training{Colors.ENDC}")
    print(f"{'='*70}\n")

    for epoch in range(1, args.num_epochs + 1):
        epoch_start = time.time()

        # Update hardness weights periodically
        if current_weights is None or (epoch - 1) % args.update_weights_every == 0:
            print(f"\n{Colors.CYAN}Updating hardness weights...{Colors.ENDC}")

            # Use cached GT, only recompute predictions
            ious, errs, scores, has_gt = hardness_cache.update_predictions(model, img_size)

            # Compute weights
            current_weights = compute_hardness_weights(
                ious, errs, scores, has_gt, args.max_weight
            )

            stats = get_weight_stats(current_weights, has_gt)
            print(f"  Hard samples (w>=4): {stats['pct_hard']:.1f}%")
            print(f"  Very hard (w>=10): {stats['pct_very_hard']:.1f}%")
            print(f"  Mean pos weight: {stats['mean_pos_weight']:.2f}")

        # Create weighted sampler
        sampler = WeightedRandomSampler(
            torch.from_numpy(current_weights),
            num_samples=len(train_dataset),
            replacement=True,
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if args.num_workers > 0 else False,
        )

        # Train epoch
        train_metrics = train_epoch_hft(
            model, train_loader, optimizer, loss_fn, device,
            ema, epoch, args.num_epochs, args.grad_clip
        )

        # Validate with EMA model
        ema.apply_shadow(model)
        val_metrics = validate_with_recall(model, val_loader, loss_fn, device, img_size)
        ema.restore(model)

        # Update scheduler based on R@95
        scheduler.step(val_metrics['r95'])

        # Check for improvement
        if val_metrics['r95'] > best_r95:
            best_r95 = val_metrics['r95']
            epochs_without_improvement = 0

            # Save best EMA checkpoint
            ema.apply_shadow(model)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ema_state_dict': ema.state_dict(),
                'config': {
                    'img_size': img_size,
                    'width_mult': width_mult,
                    'coord_activation': 'clamp',  # Use clamp for inference
                },
                'metrics': val_metrics,
            }, output_dir / 'best_ema.pth')
            ema.restore(model)

            print(f"\n  {Colors.GREEN}★ New best R@95: {best_r95:.2f}%{Colors.ENDC}")
        else:
            epochs_without_improvement += 1

        # Print epoch summary
        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[1]['lr']  # Head LR

        print(f"\nEpoch {epoch}/{args.num_epochs} ({elapsed:.1f}s) | LR: {current_lr:.2e}")
        print(f"  Train: loss={train_metrics['loss']:.4f}")
        print(f"  Val:   IoU={val_metrics['mean_iou']:.4f} | "
              f"R@90={val_metrics['r90']:.1f}% | "
              f"{Colors.BOLD}R@95={val_metrics['r95']:.1f}%{Colors.ENDC} | "
              f"Err={val_metrics['mean_err_px']:.2f}px")

        if epochs_without_improvement > 0:
            print(f"  {Colors.DIM}No improvement for {epochs_without_improvement} epochs{Colors.ENDC}")

        # Early stopping
        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"\n{Colors.YELLOW}Early stopping triggered.{Colors.ENDC}")
            break

        # Save periodic checkpoint
        if epoch % 10 == 0:
            ema.apply_shadow(model)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': {
                    'img_size': img_size,
                    'width_mult': width_mult,
                    'coord_activation': 'clamp',
                },
            }, output_dir / f'epoch_{epoch:03d}.pth')
            ema.restore(model)

    # Final summary
    print(f"\n{'='*70}")
    print(f"  {Colors.BOLD}Training Complete{Colors.ENDC}")
    print(f"{'='*70}")
    print(f"  Best R@95: {Colors.GREEN}{best_r95:.2f}%{Colors.ENDC}")
    print(f"  Checkpoints: {output_dir}")
    print()


if __name__ == "__main__":
    main()
