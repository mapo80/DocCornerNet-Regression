"""
Training script for DocCornerNet model.

Features:
- Supervised training with GT labels
- ReduceLROnPlateau scheduler
- Validation metrics: loss, IoU, corner error
- Best model checkpointing
- Progress bars with tqdm
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from tqdm import tqdm

from model import create_model
from dataset import create_dataloaders
from metrics import ValidationMetrics


# ANSI colors for terminal output
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DocCornerNet model for document corner detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data paths
    parser.add_argument(
        "--data_root",
        type=str,
        default="../doc-scanner-dataset-labeled",
        help="Path to dataset root (contains images/, labels/, train.txt, val.txt)",
    )
    parser.add_argument("--train_split", type=str, default="train.txt", help="Train split file name")
    parser.add_argument("--val_split", type=str, default="val.txt", help="Validation split file name")

    # Model parameters
    parser.add_argument("--img_size", type=int, default=224, help="Input image size")
    parser.add_argument("--width_mult", type=float, default=1.0, help="MobileNet width multiplier")
    parser.add_argument("--reduced_tail", action="store_true", default=True, help="Use reduced tail")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use pretrained backbone")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate in head (0 to disable)")
    parser.add_argument("--coord_activation", type=str, default="sigmoid",
                        choices=["sigmoid", "clamp", "none"],
                        help="Coordinate activation: sigmoid (original), clamp (recommended), none")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--val_frequency", type=int, default=1, help="Validate every N epochs")
    parser.add_argument("--cache_images", action="store_true", help="Pre-load images into RAM (faster, uses more memory)")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory for persistent disk cache (e.g., ./cache)")
    parser.add_argument("--force_cache", action="store_true", help="Force regeneration of disk cache")
    parser.add_argument("--outlier_list", type=str, default=None, help="Optional txt with image names to oversample")
    parser.add_argument("--outlier_weight", type=float, default=1.0, help="Weight multiplier for outliers (>=1.0)")

    # Loss weights
    parser.add_argument("--lambda_coords", type=float, default=1.0, help="Weight for coordinate loss")
    parser.add_argument("--lambda_score", type=float, default=2.0, help="Weight for score loss")
    parser.add_argument("--lambda_geometry", type=float, default=0.1, help="Weight for geometry loss (area + edge)")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing for score BCE (0 to disable)")
    parser.add_argument("--focal_alpha", type=float, default=0.75, help="Focal loss alpha (weight for positives)")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma (focusing parameter)")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping max norm (0 to disable)")
    parser.add_argument("--use_wing_loss", action="store_true", help="Use Wing Loss instead of SmoothL1 for coords")
    parser.add_argument("--wing_width", type=float, default=10.0, help="Wing loss width parameter (in pixels)")
    parser.add_argument("--wing_curvature", type=float, default=2.0, help="Wing loss curvature parameter")

    # LR scheduler and early stopping
    parser.add_argument("--warmup_epochs", type=int, default=0, help="Number of warmup epochs (LR ramps from lr/10 to lr)")
    parser.add_argument("--scheduler", type=str, default="plateau", choices=["plateau", "cosine"],
                        help="LR scheduler: 'plateau' (ReduceLROnPlateau) or 'cosine' (CosineAnnealingWarmRestarts)")
    parser.add_argument("--scheduler_factor", type=float, default=0.5, help="LR reduction factor (plateau only)")
    parser.add_argument("--scheduler_patience", type=int, default=3, help="Patience epochs for LR reduction (plateau only)")
    parser.add_argument("--cosine_t0", type=int, default=10, help="Cosine scheduler: epochs for first restart")
    parser.add_argument("--cosine_t_mult", type=int, default=2, help="Cosine scheduler: multiplier for restart period")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Early stopping patience (0 to disable)")
    parser.add_argument("--use_iou_for_best", action="store_true", help="Use IoU instead of val_loss for best model selection")
    parser.add_argument("--freeze_backbone_epochs", type=int, default=0, help="Number of epochs to freeze backbone (0 = never freeze)")
    parser.add_argument("--backbone_lr_mult", type=float, default=0.01, help="LR multiplier for backbone when unfreezing")

    # Augmentation (optional presets; default behaviour unchanged)
    parser.add_argument("--augment_preset", type=str, default="default",
                        choices=["default", "geo_light", "strong"],
                        help="Augmentation preset: default, geo_light, or strong (heavy augmentation)")
    parser.add_argument("--outlier_augment_preset", type=str, default=None,
                        choices=["default", "geo_light", "strong"],
                        help="Optional preset applied only to outlier_list samples")

    # Output
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--experiment_name", type=str, default="doccornernet", help="Experiment name")

    # Device
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cpu, cuda, mps)")

    # Resume training
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from (e.g., checkpoints/doccornernet/latest.pth)")

    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    """Get the appropriate device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def get_warmup_lr(epoch: int, warmup_epochs: int, base_lr: float) -> float:
    """Calculate learning rate during warmup phase.

    LR ramps linearly from base_lr/10 to base_lr over warmup_epochs.
    """
    if warmup_epochs <= 0 or epoch > warmup_epochs:
        return base_lr
    # Linear warmup: start at base_lr/10, end at base_lr
    warmup_factor = 0.1 + 0.9 * (epoch - 1) / max(1, warmup_epochs - 1)
    return base_lr * warmup_factor


def set_lr(optimizer: optim.Optimizer, lr: float, backbone_lr_mult: float = 1.0):
    """Set learning rate for all parameter groups."""
    for i, param_group in enumerate(optimizer.param_groups):
        if i == 0 and len(optimizer.param_groups) > 1:
            # First group is backbone (if using differential LR)
            param_group['lr'] = lr * backbone_lr_mult
        else:
            param_group['lr'] = lr


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {text}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.ENDC}\n")


def print_config(args, model, train_size: int, val_size: int):
    """Print training configuration."""
    print(f"{Colors.BOLD}Configuration:{Colors.ENDC}")
    print(f"  Model:        MobileNetV3-Small (width={args.width_mult})")
    print(f"  Parameters:   {model.get_num_params():,}")
    print(f"  Input size:   {args.img_size}x{args.img_size}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Train images: {train_size}")
    print(f"  Val images:   {val_size}")
    print(f"  Epochs:       {args.num_epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Lambda coords: {args.lambda_coords}")
    print(f"  Lambda score:  {args.lambda_score}")
    print()


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Focal Loss for binary classification.

    Focal Loss down-weights easy examples and focuses on hard ones.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        logits: Raw logits (before sigmoid)
        targets: Binary targets (0 or 1, can be smoothed)
        alpha: Balancing factor for positive class (default 0.25)
        gamma: Focusing parameter (default 2.0, higher = more focus on hard examples)

    Returns:
        Per-sample focal loss
    """
    probs = torch.sigmoid(logits)
    # p_t = p for y=1, (1-p) for y=0
    p_t = probs * targets + (1 - probs) * (1 - targets)
    # alpha_t = alpha for y=1, (1-alpha) for y=0
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    # Focal weight
    focal_weight = alpha_t * (1 - p_t) ** gamma
    # BCE loss
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    return focal_weight * bce


def wing_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    width: float = 10.0,
    curvature: float = 2.0,
) -> torch.Tensor:
    """
    Wing Loss for robust coordinate regression.

    Wing Loss provides stronger gradients for small errors (critical for corner detection)
    while remaining robust to outliers for large errors.

    - For small errors (|x| < width): uses log-based loss (stronger gradients)
    - For large errors (|x| >= width): uses linear loss (reduces outlier sensitivity)

    Reference: "Wing Loss for Robust Facial Landmark Localisation with CNNs" (CVPR 2018)

    Args:
        pred: Predicted coordinates in pixel space [B, 8]
        target: Ground truth coordinates in pixel space [B, 8]
        width: Threshold for log vs linear (default 10 pixels)
        curvature: Affects log curvature, typically 2.0

    Returns:
        Wing loss value (scalar)
    """
    diff = pred - target
    abs_diff = torch.abs(diff)

    # Constant C for continuity at x = width
    c = width / (1 + math.log(1 + width / curvature))

    # Wing loss formula
    loss = torch.where(
        abs_diff < width,
        width * torch.log(1 + abs_diff / curvature),
        abs_diff - c
    )
    return loss


def compute_quad_area(coords: torch.Tensor) -> torch.Tensor:
    """
    Compute area of quadrilateral using Shoelace formula.

    Args:
        coords: [B, 8] coordinates as (x0,y0,x1,y1,x2,y2,x3,y3) for TL,TR,BR,BL

    Returns:
        [B] areas (always positive)
    """
    # Extract x and y coordinates
    x = coords[:, 0::2]  # [B, 4] -> x0, x1, x2, x3
    y = coords[:, 1::2]  # [B, 4] -> y0, y1, y2, y3

    # Shoelace formula: sum(x_i * y_{i+1} - x_{i+1} * y_i) / 2
    # For closed polygon, wrap around
    x_next = torch.roll(x, -1, dims=1)  # x1, x2, x3, x0
    y_next = torch.roll(y, -1, dims=1)  # y1, y2, y3, y0

    area = 0.5 * torch.abs((x * y_next - x_next * y).sum(dim=1))
    return area


def compute_edge_lengths(coords: torch.Tensor) -> torch.Tensor:
    """
    Compute edge lengths of quadrilateral.

    Args:
        coords: [B, 8] coordinates as (x0,y0,x1,y1,x2,y2,x3,y3)

    Returns:
        [B, 4] edge lengths (TL-TR, TR-BR, BR-BL, BL-TL)
    """
    # Reshape to [B, 4, 2] for easier manipulation
    pts = coords.view(-1, 4, 2)

    # Compute edges (each vertex to next vertex)
    pts_next = torch.roll(pts, -1, dims=1)
    edges = pts_next - pts  # [B, 4, 2]

    # Edge lengths
    lengths = torch.sqrt((edges ** 2).sum(dim=2) + 1e-8)  # [B, 4]
    return lengths


def geometry_loss(
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    lambda_area: float = 1.0,
    lambda_edge: float = 1.0,
) -> torch.Tensor:
    """
    Geometry-preserving loss for quadrilateral matching.

    Ensures predicted quad has similar area and edge lengths to ground truth.
    This prevents shrinking/expanding of predictions.

    Args:
        pred_coords: [B, 8] predicted normalized coordinates
        gt_coords: [B, 8] ground truth normalized coordinates
        lambda_area: Weight for area matching loss
        lambda_edge: Weight for edge length matching loss

    Returns:
        Geometry loss (scalar)
    """
    # Area loss: relative difference in areas
    pred_area = compute_quad_area(pred_coords)
    gt_area = compute_quad_area(gt_coords)

    # Use relative area error to be scale-invariant
    area_error = torch.abs(pred_area - gt_area) / (gt_area + 1e-8)
    loss_area = area_error.mean()

    # Edge length loss: match each edge
    pred_edges = compute_edge_lengths(pred_coords)
    gt_edges = compute_edge_lengths(gt_coords)

    # Relative edge error
    edge_error = torch.abs(pred_edges - gt_edges) / (gt_edges + 1e-8)
    loss_edge = edge_error.mean()

    return lambda_area * loss_area + lambda_edge * loss_edge


class DetectionLoss:
    """
    Loss function for document corner detection.

    Components:
    - Coordinate loss: SmoothL1 or Wing Loss (masked by has_label)
    - Geometry loss: Area + edge length matching (prevents shrinkage)
    - Score loss: Focal Loss for classification (handles class imbalance)
    - Label smoothing: prevents overconfident predictions
    """

    def __init__(
        self,
        lambda_coords: float = 1.0,
        lambda_score: float = 1.0,
        lambda_geometry: float = 0.1,
        label_smoothing: float = 0.1,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        use_wing_loss: bool = False,
        wing_width: float = 10.0,
        wing_curvature: float = 2.0,
        img_size: int = 256,
    ):
        self.lambda_coords = lambda_coords
        self.lambda_score = lambda_score
        self.lambda_geometry = lambda_geometry
        self.label_smoothing = label_smoothing
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.use_wing_loss = use_wing_loss
        self.wing_width = wing_width
        self.wing_curvature = wing_curvature
        self.img_size = img_size

        self.smooth_l1 = nn.SmoothL1Loss(reduction="none")

    def __call__(
        self,
        coords_pred: torch.Tensor,
        score_pred: torch.Tensor,
        coords_gt: torch.Tensor,
        score_gt: torch.Tensor,
        has_label: torch.Tensor,
    ) -> dict:
        """Compute detection loss."""
        device = coords_pred.device

        # Convert mask to float
        mask = has_label.float().to(device)

        # Initialize losses
        loss_coords = torch.tensor(0.0, device=device)
        loss_geometry = torch.tensor(0.0, device=device)
        loss_score = torch.tensor(0.0, device=device)

        # Apply label smoothing: 1 -> 1-eps, 0 -> eps
        if self.label_smoothing > 0:
            score_gt_smooth = score_gt * (1 - self.label_smoothing) + (1 - score_gt) * self.label_smoothing
        else:
            score_gt_smooth = score_gt

        # ----- Coordinate Loss (only for positive samples with labels) -----
        num_positive = mask.sum()
        if num_positive > 0:
            if self.use_wing_loss:
                # Wing Loss in pixel space for stronger gradients on small errors
                pred_px = coords_pred * self.img_size
                gt_px = coords_gt.to(device) * self.img_size
                wing_loss_per_coord = wing_loss(
                    pred_px, gt_px,
                    width=self.wing_width,
                    curvature=self.wing_curvature
                )
                # Mean over coordinates, then mask
                coord_loss_per_sample = wing_loss_per_coord.mean(dim=1)
                # Normalize back to be comparable with SmoothL1 scale
                loss_coords = (coord_loss_per_sample * mask).sum() / num_positive / self.img_size
            else:
                # Original SmoothL1 loss
                coord_loss_per_sample = self.smooth_l1(coords_pred, coords_gt.to(device)).mean(dim=1)
                loss_coords = (coord_loss_per_sample * mask).sum() / num_positive

            # ----- Geometry Loss (area + edge matching) -----
            if self.lambda_geometry > 0:
                # Only compute for samples with labels
                pos_mask = mask.bool()
                if pos_mask.any():
                    pred_pos = coords_pred[pos_mask]
                    gt_pos = coords_gt.to(device)[pos_mask]
                    loss_geometry = geometry_loss(pred_pos, gt_pos)

        # ----- Score Loss (Focal Loss for all samples) -----
        score_loss_per_sample = focal_bce_with_logits(
            score_pred, score_gt_smooth.to(device),
            alpha=self.focal_alpha, gamma=self.focal_gamma
        )
        loss_score = score_loss_per_sample.mean()

        # Combine losses
        total_loss = (
            self.lambda_coords * loss_coords +
            self.lambda_geometry * loss_geometry +
            self.lambda_score * loss_score
        )

        return {
            "total": total_loss,
            "coords": loss_coords,
            "geometry": loss_geometry,
            "score": loss_score,
            "num_positive": num_positive.item(),
        }


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: DetectionLoss,
    device: torch.device,
    epoch: int,
    num_epochs: int,
    grad_clip: float = 1.0,
) -> dict:
    """Train for one epoch with progress bar."""
    model.train()

    total_loss = 0.0
    total_coords = 0.0
    total_score = 0.0
    num_batches = 0

    # Progress bar
    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}/{num_epochs} [Train]",
        leave=False,
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )

    for batch in pbar:
        # Move to device
        images = batch["image"].to(device)
        coords_gt = batch["coords"].to(device)
        score_gt = batch["score"].to(device)
        has_label = batch["has_label"].to(device)

        # Forward pass
        coords_pred, score_pred = model(images)

        # Compute loss
        losses = loss_fn(
            coords_pred, score_pred,
            coords_gt, score_gt,
            has_label,
        )

        # Backward pass
        optimizer.zero_grad()
        losses["total"].backward()

        # Gradient clipping
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Accumulate
        total_loss += losses["total"].item()
        total_coords += losses["coords"].item()
        total_score += losses["score"].item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses["total"].item():.4f}',
        })

    return {
        "loss": total_loss / num_batches,
        "coords": total_coords / num_batches,
        "score": total_score / num_batches,
    }


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: DetectionLoss,
    device: torch.device,
    epoch: int,
    num_epochs: int,
) -> dict:
    """Validate model with progress bar."""
    model.eval()
    metrics = ValidationMetrics()

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(
        val_loader,
        desc=f"Epoch {epoch}/{num_epochs} [Val]  ",
        leave=False,
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    )

    with torch.no_grad():
        for batch in pbar:
            images = batch["image"].to(device)
            coords_gt = batch["coords"].to(device)
            score_gt = batch["score"].to(device)
            has_label = batch["has_label"].to(device)

            coords_pred, score_pred = model(images)

            losses = loss_fn(
                coords_pred, score_pred,
                coords_gt, score_gt,
                has_label,
            )

            total_loss += losses["total"].item()
            num_batches += 1

            metrics.update(
                coords_pred, coords_gt,
                score_pred, score_gt,
                has_label,
            )

    metric_results = metrics.compute()

    return {
        "loss": total_loss / num_batches,
        **metric_results,  # Include all metrics from ValidationMetrics
    }


def print_epoch_summary(
    epoch: int,
    num_epochs: int,
    train_metrics: dict,
    val_metrics: dict,
    current_lr: float,
    epoch_time: float,
    best_val_loss: float,
    best_iou: float,
    saved_best: bool,
):
    """Print formatted epoch summary."""
    # Epoch header
    progress = epoch / num_epochs * 100
    bar_width = 20
    filled = int(bar_width * epoch / num_epochs)
    bar = "█" * filled + "░" * (bar_width - filled)

    print(f"\n{Colors.BOLD}Epoch {epoch}/{num_epochs}{Colors.ENDC} [{bar}] {progress:.0f}%")

    # Table dimensions
    col1, col2, col3, col4 = 22, 10, 10, 10
    table_width = col1 + col2 + col3 + col4 + 6

    # Header
    print(f"  {'─' * table_width}")
    print(f"  {'Metric':<{col1}} {'Train':>{col2}} {'Val':>{col3}} {'Best':>{col4}}")
    print(f"  {'─' * table_width}")

    # Loss
    val_loss_color = Colors.GREEN if val_metrics['loss'] <= best_val_loss else ""
    val_loss_end = Colors.ENDC if val_loss_color else ""
    print(f"  {'Loss':<{col1}} {train_metrics['loss']:>{col2}.4f} {val_loss_color}{val_metrics['loss']:>{col3}.4f}{val_loss_end} {best_val_loss:>{col4}.4f}")

    # IoU
    iou_color = Colors.GREEN if val_metrics['mean_iou'] >= best_iou else ""
    iou_end = Colors.ENDC if iou_color else ""
    print(f"  {'IoU':<{col1}} {'-':>{col2}} {iou_color}{val_metrics['mean_iou']:>{col3}.4f}{iou_end} {best_iou:>{col4}.4f}")

    # Corner Error (in pixels)
    corner_px = val_metrics.get('corner_error_px', val_metrics['mean_corner_error'] * 224)
    print(f"  {'Corner Error (px)':<{col1}} {'-':>{col2}} {corner_px:>{col3}.1f}px {'-':>{col4}}")

    print(f"  {'─' * table_width}")

    # Recall at different IoU thresholds
    print(f"  {Colors.BOLD}Recall @ IoU threshold:{Colors.ENDC}")
    r50 = val_metrics.get('recall_50', 0) * 100
    r75 = val_metrics.get('recall_75', 0) * 100
    r90 = val_metrics.get('recall_90', 0) * 100
    print(f"  {'Recall@50 (≈AP@50)':<{col1}} {'-':>{col2}} {r50:>{col3-1}.1f}%")
    print(f"  {'Recall@75 (≈AP@75)':<{col1}} {'-':>{col2}} {r75:>{col3-1}.1f}%")
    print(f"  {'Recall@90':<{col1}} {'-':>{col2}} {r90:>{col3-1}.1f}%")

    print(f"  {'─' * table_width}")

    # Corner distance percentiles
    print(f"  {Colors.BOLD}Corner Distance (normalized):{Colors.ENDC}")
    p50 = val_metrics.get('corner_dist_p50', 0)
    p90 = val_metrics.get('corner_dist_p90', 0)
    p95 = val_metrics.get('corner_dist_p95', 0)
    print(f"  {'P50 (median)':<{col1}} {'-':>{col2}} {p50:>{col3}.4f}")
    print(f"  {'P90':<{col1}} {'-':>{col2}} {p90:>{col3}.4f}")
    print(f"  {'P95':<{col1}} {'-':>{col2}} {p95:>{col3}.4f}")

    print(f"  {'─' * table_width}")

    # Footer info
    print(f"  {Colors.DIM}LR: {current_lr:.2e} | Time: {epoch_time:.1f}s{Colors.ENDC}", end="")

    if saved_best:
        print(f"  {Colors.GREEN}★ New best model saved!{Colors.ENDC}")
    else:
        print()


def main():
    args = parse_args()

    # Setup
    device = get_device(args.device)

    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Header
    print_header("DocCornerNet Training")
    print(f"Device: {Colors.BOLD}{device}{Colors.ENDC}")
    print(f"Output: {Colors.BOLD}{output_dir}{Colors.ENDC}")

    # Save config
    config = vars(args)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Create model
    model = create_model(
        img_size=args.img_size,
        width_mult=args.width_mult,
        reduced_tail=args.reduced_tail,
        pretrained=args.pretrained,
        dropout=args.dropout,
        coord_activation=args.coord_activation,
    )
    model = model.to(device)

    # Print coord_activation mode
    print(f"Coord activation: {Colors.BOLD}{args.coord_activation}{Colors.ENDC}")

    # Create dataloaders
    print(f"\n{Colors.DIM}Loading dataset...{Colors.ENDC}")
    # Augmentation presets (default=None keeps current behaviour)
    def get_aug_config(preset: str):
        if preset == "geo_light":
            return {
                "rotation_degrees": 15,
                "scale_range": (0.7, 1.2),
                "translate": 0.1,
                "perspective": (0.03, 0.08),
                "brightness": 0.15,
                "contrast": 0.15,
                "saturation": 0.1,
                "blur_prob": 0.0,
            }
        elif preset == "strong":
            # Heavy augmentation to reduce overfitting
            return {
                "rotation_degrees": 25,
                "scale_range": (0.6, 1.3),
                "translate": 0.15,
                "perspective": (0.05, 0.12),
                "brightness": 0.3,
                "contrast": 0.3,
                "saturation": 0.2,
                "blur_prob": 0.2,
                "blur_kernel": 5,
            }
        return None

    base_aug = get_aug_config(args.augment_preset)
    outlier_aug = get_aug_config(args.outlier_augment_preset) if args.outlier_augment_preset else None

    train_loader, val_loader = create_dataloaders(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment_config=base_aug,
        augment_config_outlier=outlier_aug,
        outlier_list=args.outlier_list,
        outlier_weight=args.outlier_weight,
        cache_images=args.cache_images,
        cache_dir=args.cache_dir,
        force_cache=args.force_cache,
        train_split=args.train_split,
        val_split=args.val_split,
    )

    # Print configuration
    print_config(args, model, len(train_loader.dataset), len(val_loader.dataset))

    # Loss function
    loss_fn = DetectionLoss(
        lambda_coords=args.lambda_coords,
        lambda_score=args.lambda_score,
        lambda_geometry=args.lambda_geometry,
        label_smoothing=args.label_smoothing,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        use_wing_loss=args.use_wing_loss,
        wing_width=args.wing_width,
        wing_curvature=args.wing_curvature,
        img_size=args.img_size,
    )

    # Print loss config
    if args.use_wing_loss:
        print(f"Loss: {Colors.BOLD}Wing Loss{Colors.ENDC} (width={args.wing_width}px, curvature={args.wing_curvature})")
    else:
        print(f"Loss: {Colors.BOLD}SmoothL1{Colors.ENDC}")
    if args.lambda_geometry > 0:
        print(f"Geometry loss: {Colors.BOLD}enabled{Colors.ENDC} (lambda={args.lambda_geometry})")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # LR Scheduler
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.cosine_t0,
            T_mult=args.cosine_t_mult,
            eta_min=args.min_lr,
        )
        print(f"{Colors.CYAN}Using CosineAnnealingWarmRestarts (T_0={args.cosine_t0}, T_mult={args.cosine_t_mult}){Colors.ENDC}")
    else:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            min_lr=args.min_lr,
            verbose=False,
        )

    # Training loop state
    best_val_loss = float("inf")
    best_iou = 0.0
    epochs_without_improvement = 0
    history = []
    start_epoch = 1

    # Resume from checkpoint if specified
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"\n{Colors.CYAN}Resuming from checkpoint: {resume_path}{Colors.ENDC}")
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)

            # Load model state
            model.load_state_dict(checkpoint["model_state_dict"])

            # Load optimizer state
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Load scheduler state
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            # Load training state
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            best_iou = checkpoint.get("best_iou", 0.0)

            # Load history if available
            history_path = output_dir / "history.json"
            if history_path.exists():
                try:
                    with open(history_path, "r") as f:
                        history = json.load(f)
                    print(f"  Loaded {len(history)} history entries")
                except Exception as e:
                    print(f"  Warning: Could not load history: {e}")

            print(f"  Resuming from epoch {start_epoch}")
            print(f"  Best val_loss: {best_val_loss:.4f}")
            print(f"  Best IoU: {best_iou:.4f}")
        else:
            print(f"{Colors.RED}Warning: Checkpoint not found: {resume_path}{Colors.ENDC}")

    # Determine which metric to use for best model selection
    use_iou_for_best = args.use_iou_for_best
    if use_iou_for_best:
        print(f"\n{Colors.CYAN}Using IoU as primary metric for model selection{Colors.ENDC}")
    else:
        print(f"\n{Colors.CYAN}Using val_loss as primary metric for model selection{Colors.ENDC}")

    print_header("Training Started")

    # Freeze backbone if requested (only if not resuming past freeze epochs)
    freeze_backbone_epochs = args.freeze_backbone_epochs
    if freeze_backbone_epochs > 0 and start_epoch <= freeze_backbone_epochs:
        print(f"{Colors.CYAN}Freezing backbone for epochs 1-{freeze_backbone_epochs}{Colors.ENDC}")
        for param in model.backbone.parameters():
            param.requires_grad = False

    # Warmup info
    warmup_epochs = args.warmup_epochs
    if warmup_epochs > 0:
        print(f"{Colors.CYAN}Warmup: LR will ramp from {args.lr/10:.2e} to {args.lr:.2e} over {warmup_epochs} epochs{Colors.ENDC}")

    try:
        for epoch in range(start_epoch, args.num_epochs + 1):
            epoch_start = time.time()

            # Apply warmup LR if in warmup phase
            if warmup_epochs > 0 and epoch <= warmup_epochs:
                warmup_lr = get_warmup_lr(epoch, warmup_epochs, args.lr)
                backbone_mult = args.backbone_lr_mult if epoch > freeze_backbone_epochs else 1.0
                set_lr(optimizer, warmup_lr, backbone_mult)
                if epoch == 1:
                    print(f"  {Colors.CYAN}↑ Warmup epoch {epoch}/{warmup_epochs}: LR = {warmup_lr:.2e}{Colors.ENDC}")

            # Unfreeze backbone after specified epochs
            if freeze_backbone_epochs > 0 and epoch == freeze_backbone_epochs + 1:
                backbone_lr = args.lr * args.backbone_lr_mult
                print(f"\n{Colors.CYAN}Unfreezing backbone at epoch {epoch}{Colors.ENDC}")
                print(f"{Colors.CYAN}Backbone LR: {backbone_lr:.2e} ({args.backbone_lr_mult}x head LR){Colors.ENDC}")
                for param in model.backbone.parameters():
                    param.requires_grad = True
                optimizer = optim.AdamW([
                    {'params': model.backbone.parameters(), 'lr': backbone_lr},
                    {'params': model.head.parameters(), 'lr': args.lr}
                ], weight_decay=args.weight_decay)

            # Train
            train_metrics = train_epoch(
                model, train_loader, optimizer, loss_fn, device,
                epoch, args.num_epochs, grad_clip=args.grad_clip
            )

            # Step cosine scheduler after each epoch (if not in warmup)
            if args.scheduler == "cosine" and epoch > warmup_epochs:
                scheduler.step()

            # Validate
            if epoch % args.val_frequency == 0 or epoch == args.num_epochs:
                val_metrics = validate(
                    model, val_loader, loss_fn, device,
                    epoch, args.num_epochs
                )

                # Update scheduler (skip during warmup)
                old_lr = optimizer.param_groups[0]["lr"]
                if epoch > warmup_epochs:
                    if args.scheduler == "cosine":
                        # Cosine scheduler steps per epoch (already called after train)
                        pass
                    else:
                        # Plateau scheduler steps based on metric
                        if use_iou_for_best:
                            scheduler.step(-val_metrics["mean_iou"])
                        else:
                            scheduler.step(val_metrics["loss"])
                    new_lr = optimizer.param_groups[0]["lr"]

                    if new_lr < old_lr:
                        print(f"\n  {Colors.YELLOW}↓ Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}{Colors.ENDC}")
                else:
                    new_lr = old_lr
                    if epoch == warmup_epochs:
                        print(f"\n  {Colors.CYAN}✓ Warmup complete. LR = {args.lr:.2e}{Colors.ENDC}")

                current_lr = new_lr
                epoch_time = time.time() - epoch_start

                # Check for best model
                saved_best = False

                # Update best values
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_metrics["loss"],
                        "val_iou": val_metrics["mean_iou"],
                        "config": config,
                    }, output_dir / "best_loss.pth")
                    if not use_iou_for_best:
                        saved_best = True
                        epochs_without_improvement = 0

                if val_metrics["mean_iou"] > best_iou:
                    best_iou = val_metrics["mean_iou"]
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_metrics["loss"],
                        "val_iou": val_metrics["mean_iou"],
                        "config": config,
                    }, output_dir / "best.pth")
                    if use_iou_for_best:
                        saved_best = True
                        epochs_without_improvement = 0

                # Update epochs without improvement if not saved
                if not saved_best:
                    epochs_without_improvement += 1

                # Print summary
                print_epoch_summary(
                    epoch, args.num_epochs,
                    train_metrics, val_metrics,
                    current_lr, epoch_time,
                    best_val_loss, best_iou,
                    saved_best,
                )

                # Save history
                history.append({
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "val_loss": val_metrics["loss"],
                    "mean_iou": val_metrics["mean_iou"],
                    "median_iou": val_metrics.get("median_iou", 0),
                    "corner_error_px": val_metrics.get("corner_error_px", 0),
                    "recall_50": val_metrics.get("recall_50", 0),
                    "recall_75": val_metrics.get("recall_75", 0),
                    "recall_90": val_metrics.get("recall_90", 0),
                    "corner_dist_p50": val_metrics.get("corner_dist_p50", 0),
                    "corner_dist_p90": val_metrics.get("corner_dist_p90", 0),
                    "corner_dist_p95": val_metrics.get("corner_dist_p95", 0),
                    "lr": current_lr,
                })

                # Early stopping check
                if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                    print(f"\n{Colors.YELLOW}Early stopping triggered after {epochs_without_improvement} epochs without improvement.{Colors.ENDC}")
                    break
            else:
                # Training-only epoch (no validation)
                epoch_time = time.time() - epoch_start
                print(f"\n{Colors.DIM}Epoch {epoch}/{args.num_epochs} - Train Loss: {train_metrics['loss']:.4f} ({epoch_time:.1f}s){Colors.ENDC}")

            # Save latest checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "best_iou": best_iou,
                "config": config,
            }, output_dir / "latest.pth")

    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Training interrupted by user.{Colors.ENDC}")
        print(f"Saving checkpoint...")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "best_iou": best_iou,
            "config": config,
        }, output_dir / "interrupted.pth")

    # Save training history
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, float):
            return float(obj)
        return obj

    with open(output_dir / "history.json", "w") as f:
        json.dump(convert_to_serializable(history), f, indent=2)

    # Final summary
    print_header("Training Complete")
    print(f"  Best validation loss: {Colors.GREEN}{best_val_loss:.4f}{Colors.ENDC}")
    print(f"  Best IoU:             {Colors.GREEN}{best_iou:.4f}{Colors.ENDC}")
    print(f"  Checkpoints saved to: {Colors.BOLD}{output_dir}{Colors.ENDC}")
    print()


if __name__ == "__main__":
    main()
