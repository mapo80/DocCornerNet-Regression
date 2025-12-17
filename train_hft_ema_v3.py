"""
HFT-EMA v3 - Optimized for MPS (Apple Silicon)

Key optimizations:
1. Cache ALL images in RAM (train NO augment, val NO augment)
2. Apply augmentation on-the-fly from RAM cache (GPU-friendly transforms)
3. TOP-K% hard mining
4. Proper train/val separation

Usage:
    python train_hft_ema_v3.py \
        --checkpoint checkpoints/best.pth \
        --data_root /path/to/dataset \
        --experiment_name hft_ema_v3
"""

import argparse
import random
import math
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import torchvision.transforms.functional as TF

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
# RAM Cache with On-the-fly Augmentation
# ============================================================================

class CachedDatasetWithAugment:
    """
    Cache all images in RAM (without augment).
    Apply augmentation on-the-fly when sampling.
    """

    def __init__(self, dataset: DocDataset, device: torch.device, apply_augment: bool = False):
        self.device = device
        self.apply_augment = apply_augment
        self.N = len(dataset)
        self.img_size = dataset.img_size

        print(f"  Caching {self.N} samples to RAM...")

        # Use the dataset directly (it should already have augment=False)
        self.images = torch.zeros((self.N, 3, self.img_size, self.img_size), dtype=torch.float32)
        self.coords = torch.zeros((self.N, 8), dtype=torch.float32)
        self.scores = torch.zeros((self.N,), dtype=torch.float32)
        self.has_label = torch.zeros((self.N,), dtype=torch.bool)

        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

        idx = 0
        for batch in tqdm(loader, desc="Caching", leave=False):
            bs = batch["image"].shape[0]
            self.images[idx:idx+bs] = batch["image"]
            self.coords[idx:idx+bs] = batch["coords"]
            self.scores[idx:idx+bs] = batch["score"]
            self.has_label[idx:idx+bs] = batch["has_label"].bool()
            idx += bs

        mem_gb = self.images.element_size() * self.images.nelement() / 1e9
        print(f"  Cached: {self.N} samples ({mem_gb:.2f} GB)")

        # Augmentation config
        self.aug_config = {
            'rotation_degrees': 10,
            'scale_min': 0.9,
            'scale_max': 1.0,
            'brightness': 0.2,
            'contrast': 0.2,
        }

    def __len__(self):
        return self.N

    def get_batch(self, indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get a batch with optional augmentation."""
        images = self.images[indices].clone()
        coords = self.coords[indices].clone()
        scores = self.scores[indices]
        has_label = self.has_label[indices]

        if self.apply_augment:
            images, coords = self._batch_augment(images, coords, has_label)

        return {
            "image": images.to(self.device, non_blocking=True),
            "coords": coords,
            "score": scores,
            "has_label": has_label,
        }

    def _batch_augment(self, images: torch.Tensor, coords: torch.Tensor, has_label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentation to a batch (CPU, then move to device)."""
        B = images.shape[0]

        for i in range(B):
            if has_label[i]:
                # Random rotation
                angle = random.uniform(-self.aug_config['rotation_degrees'], self.aug_config['rotation_degrees'])
                images[i] = TF.rotate(images[i], angle, fill=0.5)
                coords[i] = self._rotate_coords(coords[i], angle)

                # Random scale (zoom out only)
                scale = random.uniform(self.aug_config['scale_min'], self.aug_config['scale_max'])
                if scale < 1.0:
                    coords[i] = 0.5 + (coords[i] - 0.5) * scale
                    # Scale image
                    new_size = int(self.img_size * scale)
                    scaled = F.interpolate(images[i:i+1], size=new_size, mode='bilinear', align_corners=False)
                    # Handle odd dimensions with asymmetric padding
                    total_pad = self.img_size - new_size
                    pad_left = total_pad // 2
                    pad_right = total_pad - pad_left
                    pad_top = total_pad // 2
                    pad_bottom = total_pad - pad_top
                    images[i] = F.pad(scaled, (pad_left, pad_right, pad_top, pad_bottom), value=0.5)[0]

            # Color augmentation for all images
            if random.random() < 0.5:
                brightness = random.uniform(1 - self.aug_config['brightness'], 1 + self.aug_config['brightness'])
                images[i] = torch.clamp(images[i] * brightness, 0, 1)
            if random.random() < 0.5:
                contrast = random.uniform(1 - self.aug_config['contrast'], 1 + self.aug_config['contrast'])
                mean = images[i].mean()
                images[i] = torch.clamp((images[i] - mean) * contrast + mean, 0, 1)

        coords = torch.clamp(coords, 0.0, 1.0)
        return images, coords

    def _rotate_coords(self, coords: torch.Tensor, angle_deg: float) -> torch.Tensor:
        """Rotate normalized coordinates around image center."""
        angle_rad = math.radians(-angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        rotated = coords.clone()
        for i in range(0, 8, 2):
            x = coords[i] - 0.5
            y = coords[i + 1] - 0.5
            rotated[i] = x * cos_a - y * sin_a + 0.5
            rotated[i + 1] = x * sin_a + y * cos_a + 0.5

        return rotated


class FastTrainLoader:
    """Fast training loader from RAM cache with weighted sampling."""

    def __init__(self, cache: CachedDatasetWithAugment, batch_size: int, weights: np.ndarray):
        self.cache = cache
        self.batch_size = batch_size
        self.weights = torch.from_numpy(weights)
        self.num_batches = len(cache) // batch_size

    def __iter__(self):
        # Sample indices with replacement based on weights
        indices = torch.multinomial(self.weights, len(self.cache), replacement=True)

        for i in range(self.num_batches):
            batch_idx = indices[i * self.batch_size:(i + 1) * self.batch_size]
            yield self.cache.get_batch(batch_idx)

    def __len__(self):
        return self.num_batches


# ============================================================================
# Fast Validation (from RAM)
# ============================================================================

def validate_from_cache(
    model: nn.Module,
    cache: CachedDatasetWithAugment,
    loss_fn: DetectionLoss,
    device: torch.device,
    img_size: int,
    batch_size: int = 128,
) -> Dict:
    """Validate from RAM cache (no augment)."""
    model.eval()

    total_loss = 0.0
    num_batches = 0
    all_errs = []

    indices = torch.arange(cache.N)

    with torch.no_grad():
        for start in range(0, cache.N, batch_size):
            end = min(start + batch_size, cache.N)
            batch_idx = indices[start:end]

            batch = cache.get_batch(batch_idx)
            images = batch["image"]
            coords_gt = batch["coords"]
            score_gt = batch["score"]
            has_label = batch["has_label"]

            coords_pred, score_pred = model(images)

            losses = loss_fn(coords_pred, score_pred, coords_gt, score_gt, has_label)
            total_loss += losses["total"].item()
            num_batches += 1

            # Corner errors
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

    return {
        "loss": total_loss / max(num_batches, 1),
        "mean_err_px": float(np.mean(all_errs)) if num_pos > 0 else 0,
        "median_err_px": float(np.median(all_errs)) if num_pos > 0 else 0,
        "p90_err_px": float(np.percentile(all_errs, 90)) if num_pos > 0 else 0,
        "p95_err_px": float(np.percentile(all_errs, 95)) if num_pos > 0 else 0,
        "pct_under_3px": float((all_errs < 3.0).sum() / num_pos * 100) if num_pos > 0 else 0,
        "pct_under_5px": float((all_errs < 5.0).sum() / num_pos * 100) if num_pos > 0 else 0,
        "num_pos": num_pos,
    }


# ============================================================================
# Hardness Computation (from RAM cache)
# ============================================================================

def compute_errors_from_cache(
    model: nn.Module,
    cache: CachedDatasetWithAugment,
    device: torch.device,
    img_size: int,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute corner errors from RAM cache (fast!)."""
    model.eval()
    N = cache.N

    all_errs = np.zeros(N, dtype=np.float32)
    all_scores = np.zeros(N, dtype=np.float32)
    all_has_gt = cache.has_label.numpy().copy()

    indices = torch.arange(N)

    with torch.no_grad():
        for start in tqdm(range(0, N, batch_size), desc="Computing errors", leave=False):
            end = min(start + batch_size, N)
            batch_idx = indices[start:end]

            # Get batch WITHOUT augment
            images = cache.images[batch_idx].to(device, non_blocking=True)
            coords_gt = cache.coords[batch_idx].numpy()
            has_label = cache.has_label[batch_idx].numpy()

            coords_pred, score_pred = model(images)
            coords_pred = coords_pred.cpu().numpy()
            scores = torch.sigmoid(score_pred).cpu().numpy().flatten()

            bs = end - start
            pred_pts = coords_pred.reshape(bs, 4, 2) * img_size
            gt_pts = coords_gt.reshape(bs, 4, 2) * img_size
            dists = np.sqrt(np.sum((pred_pts - gt_pts) ** 2, axis=2))
            mean_errs = np.mean(dists, axis=1)

            all_errs[start:end] = np.where(has_label, mean_errs, 0.0)
            all_scores[start:end] = scores

    return all_errs, all_scores, all_has_gt


def compute_hardness_weights_topk(
    corner_errs: np.ndarray,
    has_gt: np.ndarray,
    hard_pct: float = 0.25,
    medium_pct: float = 0.25,
    hard_weight: float = 3.0,
    medium_weight: float = 1.5,
) -> np.ndarray:
    """TOP-K% based hardness weights."""
    N = len(corner_errs)
    weights = np.ones(N, dtype=np.float32)

    pos_indices = np.where(has_gt)[0]
    pos_errs = corner_errs[pos_indices]

    if len(pos_errs) == 0:
        return weights

    sorted_idx = np.argsort(pos_errs)[::-1]

    n_hard = int(len(pos_errs) * hard_pct)
    n_medium = int(len(pos_errs) * medium_pct)

    hard_indices = pos_indices[sorted_idx[:n_hard]]
    medium_indices = pos_indices[sorted_idx[n_hard:n_hard + n_medium]]

    weights[hard_indices] = hard_weight
    weights[medium_indices] = medium_weight

    return weights


# ============================================================================
# Training
# ============================================================================

def train_epoch_from_cache(
    model: nn.Module,
    train_loader: FastTrainLoader,
    optimizer: optim.Optimizer,
    loss_fn: DetectionLoss,
    ema: EMAModel,
    grad_clip: float = 1.0,
) -> Dict:
    """Train one epoch from RAM cache."""
    model.train()

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)

    for batch in pbar:
        optimizer.zero_grad()

        coords_pred, score_pred = model(batch["image"])

        losses = loss_fn(coords_pred, score_pred, batch["coords"], batch["score"], batch["has_label"])
        losses["total"].backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        ema.update(model)

        total_loss += losses["total"].item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{losses["total"].item():.4f}'})

    return {"loss": total_loss / max(num_batches, 1)}


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="HFT-EMA v3 - MPS Optimized")

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--train_split", type=str, default="train_with_negative_v2.txt")
    parser.add_argument("--val_split", type=str, default="val_with_negative_v2.txt")
    parser.add_argument("--negative_image_dir", type=str, default="images-negative")

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--experiment_name", type=str, default="hft_ema_v3")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=30)

    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--lr_head", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    parser.add_argument("--ema_decay", type=float, default=0.999)

    parser.add_argument("--update_weights_every", type=int, default=5)
    parser.add_argument("--hard_pct", type=float, default=0.25)
    parser.add_argument("--medium_pct", type=float, default=0.25)
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
    DIM = '\033[2m'
    END = '\033[0m'


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"  {Colors.BOLD}HFT-EMA v3 - MPS Optimized{Colors.END}")
    print(f"{'='*60}")
    print(f"  Device: {device}")

    # Load checkpoint
    print(f"\n{Colors.DIM}Loading checkpoint...{Colors.END}")
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

    print(f"  img_size: {img_size}, params: {sum(p.numel() for p in model.parameters()):,}")

    ema = EMAModel(model, decay=args.ema_decay)

    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    data_root = Path(args.data_root)
    neg_root = data_root / args.negative_image_dir
    neg_root_str = str(neg_root) if neg_root.exists() else None

    print(f"\n{Colors.CYAN}Loading and caching datasets...{Colors.END}")

    # Create base datasets (just for structure, we'll cache them)
    train_dataset_base = DocDataset(
        image_root=str(data_root / "images"),
        label_root=str(data_root / "labels"),
        split_file=str(data_root / args.train_split),
        img_size=img_size,
        augment=False,
        negative_image_root=neg_root_str,
    )

    val_dataset_base = DocDataset(
        image_root=str(data_root / "images"),
        label_root=str(data_root / "labels"),
        split_file=str(data_root / args.val_split),
        img_size=img_size,
        augment=False,
        negative_image_root=neg_root_str,
    )

    # Cache to RAM
    print(f"\n{Colors.CYAN}Caching train set...{Colors.END}")
    train_cache = CachedDatasetWithAugment(train_dataset_base, device, apply_augment=True)

    print(f"\n{Colors.CYAN}Caching val set...{Colors.END}")
    val_cache = CachedDatasetWithAugment(val_dataset_base, device, apply_augment=False)

    # Loss
    loss_fn = DetectionLoss(
        lambda_coords=1.0,
        lambda_score=2.0,
        lambda_geometry=args.lambda_geometry,
        use_wing_loss=True,
        img_size=img_size,
    )

    # Initial eval
    print(f"\n{Colors.CYAN}Initial evaluation...{Colors.END}")
    initial_metrics = validate_from_cache(model, val_cache, loss_fn, device, img_size)

    print(f"\n  {Colors.BOLD}Checkpoint baseline:{Colors.END}")
    print(f"    Mean err: {initial_metrics['mean_err_px']:.2f}px")
    print(f"    Median err: {initial_metrics['median_err_px']:.2f}px")
    print(f"    <3px: {initial_metrics['pct_under_3px']:.1f}%")
    print(f"    <5px: {initial_metrics['pct_under_5px']:.1f}%")

    # Optimizer
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.head.parameters())

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr_backbone},
        {'params': head_params, 'lr': args.lr_head},
    ], weight_decay=args.weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.scheduler_patience, min_lr=1e-7)

    # Initial weights
    current_weights = np.ones(train_cache.N, dtype=np.float32)

    best_err = initial_metrics['mean_err_px']
    epochs_no_improve = 0

    print(f"\n{'='*60}")
    print(f"  {Colors.BOLD}Training (target: beat {best_err:.2f}px){Colors.END}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()

        # Update hardness weights
        if epoch == 1 or (epoch - 1) % args.update_weights_every == 0:
            print(f"{Colors.CYAN}Updating hardness weights...{Colors.END}")
            errs, scores, has_gt = compute_errors_from_cache(model, train_cache, device, img_size)
            current_weights = compute_hardness_weights_topk(
                errs, has_gt, args.hard_pct, args.medium_pct, args.hard_weight, args.medium_weight
            )

            pos_mask = has_gt
            pos_errs = errs[pos_mask]
            hard_mask = current_weights >= args.hard_weight
            hard_err = pos_errs[hard_mask[pos_mask]].mean() if hard_mask[pos_mask].sum() > 0 else 0
            print(f"  Hard: {hard_mask.sum()} samples ({hard_err:.2f}px), Train mean: {pos_errs.mean():.2f}px")

        # Train
        train_loader = FastTrainLoader(train_cache, args.batch_size, current_weights)
        train_metrics = train_epoch_from_cache(model, train_loader, optimizer, loss_fn, ema, args.grad_clip)

        # Validate with EMA
        ema.apply_shadow(model)
        val_metrics = validate_from_cache(model, val_cache, loss_fn, device, img_size)
        ema.restore(model)

        scheduler.step(val_metrics["mean_err_px"])

        # Check improvement
        improved = val_metrics["mean_err_px"] < best_err
        if improved:
            best_err = val_metrics["mean_err_px"]
            epochs_no_improve = 0

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
        status = f"{Colors.GREEN}★ NEW BEST{Colors.END}" if improved else ""

        print(f"Epoch {epoch}/{args.num_epochs} ({elapsed:.1f}s) | "
              f"Loss: {train_metrics['loss']:.4f} | "
              f"VAL: {val_metrics['mean_err_px']:.2f}px | "
              f"<3px: {val_metrics['pct_under_3px']:.1f}% | "
              f"<5px: {val_metrics['pct_under_5px']:.1f}% {status}")

        if epochs_no_improve >= args.early_stopping_patience:
            print(f"\n{Colors.YELLOW}Early stopping.{Colors.END}")
            break

    print(f"\n{'='*60}")
    print(f"  {Colors.BOLD}Done{Colors.END}")
    print(f"  Initial: {initial_metrics['mean_err_px']:.2f}px -> Best: {Colors.GREEN}{best_err:.2f}px{Colors.END}")
    print(f"  Saved: {output_dir / 'best_ema.pth'}")
    print()


if __name__ == "__main__":
    main()
