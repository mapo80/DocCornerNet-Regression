"""
DocDataset: PyTorch Dataset for document corner detection training.

Loads:
- Images from image_root
- Ground truth labels from YOLO OBB format (.txt files)
- Filtered by split file (train.txt, val.txt, test.txt)

Supports data augmentation with coordinate transformation.
Supports image caching with shared memory for multi-worker DataLoader.
"""

import hashlib
import math
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, WeightedRandomSampler
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Global shared memory cache (accessible by all workers)
_SHARED_CACHE = {}


def _load_and_resize_image_np(args):
    """Load image and return as numpy array for shared memory."""
    image_name, image_root, negative_image_root, img_size = args
    if image_name.startswith("negative_") and negative_image_root:
        image_path = Path(negative_image_root) / image_name
    else:
        image_path = Path(image_root) / image_name
    image = Image.open(image_path).convert("RGB")
    image = image.resize((img_size, img_size), Image.BILINEAR)
    return image_name, np.array(image, dtype=np.uint8)


class DocDataset(Dataset):
    """
    Dataset for document corner detection.

    Each sample provides:
    - image: normalized tensor [3, img_size, img_size]
    - coords: ground truth corners [8] from YOLO labels (normalized 0-1)
    - score: 1.0 if document present, else 0.0
    - has_label: 1 if label exists and is valid, else 0
    """

    def __init__(
        self,
        image_root: str,
        label_root: str,
        split_file: str,
        img_size: int = 224,
        augment: bool = False,
        augment_config: Optional[dict] = None,
        augment_config_outlier: Optional[dict] = None,
        cache_images: bool = False,
        cache_dir: Optional[str] = None,
        force_cache: bool = False,
        outlier_list: Optional[str] = None,
        negative_image_root: Optional[str] = None,
        shared_cache: Optional[dict] = None,
    ):
        """
        Args:
            image_root: Directory containing images.
            label_root: Directory containing YOLO OBB label files (.txt).
            split_file: Path to split file (train.txt, val.txt, or test.txt).
            img_size: Target image size (square).
            augment: Whether to apply data augmentation (use for training only).
            augment_config: Optional dict with augmentation parameters.
            cache_images: If True, pre-load all images into RAM for faster training.
            cache_dir: Directory for persistent disk cache. If None, uses in-memory only.
            force_cache: If True, regenerate disk cache even if it exists.
            negative_image_root: Directory containing negative images (no document).
            shared_cache: Pre-loaded shared memory cache (numpy arrays).
        """
        self.image_root = Path(image_root)
        self.label_root = Path(label_root)
        self.negative_image_root = Path(negative_image_root) if negative_image_root else None
        self.img_size = img_size
        self.augment = augment
        self.cache_images = cache_images
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.force_cache = force_cache
        self.shared_cache = shared_cache
        self.outlier_names = set()

        # Default augmentation config
        self.aug_config = {
            "rotation_degrees": 5,
            "scale_range": (0.9, 1.0),
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.1,
            "blur_prob": 0.1,
            "blur_kernel": 3,
            "translate": 0.0,
            "perspective": (0.0, 0.03),
        }
        if augment_config:
            self.aug_config.update(augment_config)
        self.aug_config_outlier = augment_config_outlier or self.aug_config

        # Load split file
        with open(split_file, "r") as f:
            self.image_list = [line.strip() for line in f if line.strip()]
        if outlier_list:
            outlier_path = Path(outlier_list)
            if outlier_path.exists():
                with open(outlier_path) as f:
                    self.outlier_names = {line.strip() for line in f if line.strip()}

        print(f"DocDataset: Loaded {len(self.image_list)} images from {split_file}")

        # Build index for fast lookup
        self.image_to_idx = {name: i for i, name in enumerate(self.image_list)}

    def _load_image(self, image_name: str) -> Image.Image:
        """Load image from shared cache or disk."""
        if self.shared_cache is not None and image_name in self.shared_cache:
            # Load from shared numpy array
            np_img = self.shared_cache[image_name]
            return Image.fromarray(np_img)
        else:
            if image_name.startswith("negative_") and self.negative_image_root:
                image_path = self.negative_image_root / image_name
            else:
                image_path = self.image_root / image_name
            return Image.open(image_path).convert("RGB")

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx: int) -> dict:
        image_name = self.image_list[idx]

        # Load image
        image = self._load_image(image_name)

        # Load GT coordinates
        label_file = self.label_root / (Path(image_name).stem + ".txt")

        if label_file.exists():
            with open(label_file, "r") as f:
                line = f.readline().strip()
            if line:
                parts = line.split()
                coords = torch.tensor([float(x) for x in parts[1:9]], dtype=torch.float32)
                has_label = 1
                score = 1.0
            else:
                coords = torch.zeros(8, dtype=torch.float32)
                has_label = 0
                score = 0.0
        else:
            coords = torch.zeros(8, dtype=torch.float32)
            has_label = 0
            score = 0.0

        # Apply augmentation
        if self.augment and has_label:
            is_outlier = image_name in self.outlier_names
            image, coords = self._apply_augmentation(image, coords, is_outlier=is_outlier)
        elif self.augment:
            image = self._apply_color_augmentation(image, self.aug_config)

        # Resize if not cached
        if self.shared_cache is None:
            image = image.resize((self.img_size, self.img_size), Image.BILINEAR)

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, IMAGENET_MEAN, IMAGENET_STD)
        coords = torch.clamp(coords, 0.0, 1.0)

        return {
            "image": image_tensor,
            "coords": coords,
            "score": torch.tensor(score, dtype=torch.float32),
            "has_label": torch.tensor(has_label, dtype=torch.long),
        }

    def _apply_augmentation(
        self,
        image: Image.Image,
        coords: torch.Tensor,
        is_outlier: bool = False,
    ) -> tuple[Image.Image, torch.Tensor]:
        """Apply geometric and color augmentation."""
        w, h = image.size
        cfg = self.aug_config_outlier if (is_outlier and self.aug_config_outlier) else self.aug_config

        # Rotation
        angle = random.uniform(-cfg["rotation_degrees"], cfg["rotation_degrees"])
        image = image.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(128, 128, 128))
        aspect_ratio = w / h
        coords = self._rotate_coords(coords, angle, aspect_ratio)

        # Scale
        scale = random.uniform(*cfg["scale_range"])
        if scale < 1.0:
            coords = 0.5 + (coords - 0.5) * scale
            new_size = int(w * scale)
            scaled = image.resize((new_size, new_size), Image.BILINEAR)
            canvas = Image.new("RGB", (w, h), (128, 128, 128))
            offset = (w - new_size) // 2
            canvas.paste(scaled, (offset, offset))
            image = canvas

        image = self._apply_color_augmentation(image, cfg)
        return image, coords

    def _apply_color_augmentation(self, image: Image.Image, cfg: dict) -> Image.Image:
        """Apply color augmentation."""
        if cfg["brightness"] > 0:
            factor = random.uniform(1 - cfg["brightness"], 1 + cfg["brightness"])
            image = TF.adjust_brightness(image, factor)
        if cfg["contrast"] > 0:
            factor = random.uniform(1 - cfg["contrast"], 1 + cfg["contrast"])
            image = TF.adjust_contrast(image, factor)
        if cfg["saturation"] > 0:
            factor = random.uniform(1 - cfg["saturation"], 1 + cfg["saturation"])
            image = TF.adjust_saturation(image, factor)
        if random.random() < cfg["blur_prob"]:
            image = image.filter(ImageFilter.GaussianBlur(radius=cfg["blur_kernel"] / 2))
        return image

    def _rotate_coords(self, coords: torch.Tensor, angle_deg: float, aspect_ratio: float = 1.0) -> torch.Tensor:
        """Rotate normalized coordinates around image center."""
        angle_rad = math.radians(-angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        rotated = []
        for i in range(0, 8, 2):
            x_norm = coords[i].item()
            y_norm = coords[i + 1].item()
            x_px = x_norm * aspect_ratio
            y_px = y_norm
            cx = 0.5 * aspect_ratio
            cy = 0.5
            x_px -= cx
            y_px -= cy
            x_new_px = x_px * cos_a - y_px * sin_a
            y_new_px = x_px * sin_a + y_px * cos_a
            x_new_px += cx
            y_new_px += cy
            x_new = x_new_px / aspect_ratio
            y_new = y_new_px
            rotated.extend([x_new, y_new])

        return torch.tensor(rotated, dtype=torch.float32)


def preload_images_to_shared_memory(
    image_list: list,
    image_root: str,
    negative_image_root: Optional[str],
    img_size: int,
    cache_dir: Optional[str] = None,
    force_cache: bool = False,
) -> dict:
    """Pre-load images into a dictionary of numpy arrays (shared across workers)."""
    from tqdm import tqdm
    from multiprocessing import Pool, cpu_count
    import pickle

    cache_path = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        image_list_hash = hashlib.md5(",".join(sorted(image_list)).encode()).hexdigest()[:8]
        cache_path = cache_dir / f"shared_cache_{img_size}px_{image_list_hash}.pkl"

        if cache_path.exists() and not force_cache:
            print(f"Loading shared cache from disk: {cache_path}")
            try:
                with open(cache_path, "rb") as f:
                    shared_cache = pickle.load(f)
                if len(shared_cache) == len(image_list):
                    print(f"Loaded {len(shared_cache)} images from disk cache.")
                    return shared_cache
                print(f"Cache incomplete, regenerating...")
            except Exception as e:
                print(f"Failed to load cache: {e}, regenerating...")

    num_workers = cpu_count()
    print(f"Pre-loading {len(image_list)} images with {num_workers} workers...")

    neg_root = str(negative_image_root) if negative_image_root else None
    args = [(name, str(image_root), neg_root, img_size) for name in image_list]

    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(_load_and_resize_image_np, args),
            total=len(args),
            desc="Caching images",
            leave=False
        ))

    shared_cache = dict(results)
    print(f"Cached {len(shared_cache)} images in memory.")

    if cache_path:
        print(f"Saving cache to disk: {cache_path}")
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(shared_cache, f)
            cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
            print(f"Cache saved ({cache_size_mb:.1f} MB)")
        except Exception as e:
            print(f"Failed to save cache: {e}")

    return shared_cache


def create_dataloaders(
    data_root: str,
    img_size: int = 192,
    batch_size: int = 32,
    num_workers: int = 4,
    augment_config: Optional[dict] = None,
    augment_config_outlier: Optional[dict] = None,
    outlier_list: Optional[str] = None,
    outlier_weight: float = 1.0,
    cache_images: bool = False,
    cache_dir: Optional[str] = None,
    force_cache: bool = False,
    train_split: str = "train.txt",
    val_split: str = "val.txt",
    negative_image_dir: str = "images-negative",
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation dataloaders with optional shared memory cache."""
    data_root = Path(data_root)
    negative_image_root = data_root / negative_image_dir
    neg_root_str = str(negative_image_root) if negative_image_root.exists() else None

    # Load split files
    with open(data_root / train_split, "r") as f:
        train_images = [line.strip() for line in f if line.strip()]
    with open(data_root / val_split, "r") as f:
        val_images = [line.strip() for line in f if line.strip()]

    # Pre-load images if caching enabled
    train_cache = None
    val_cache = None

    if cache_images:
        # Combine all images for single cache
        all_images = list(set(train_images + val_images))
        shared_cache = preload_images_to_shared_memory(
            all_images,
            str(data_root / "images"),
            neg_root_str,
            img_size,
            cache_dir,
            force_cache,
        )
        train_cache = shared_cache
        val_cache = shared_cache

    train_dataset = DocDataset(
        image_root=str(data_root / "images"),
        label_root=str(data_root / "labels"),
        split_file=str(data_root / train_split),
        img_size=img_size,
        augment=True,
        augment_config=augment_config,
        augment_config_outlier=augment_config_outlier,
        cache_images=cache_images,
        cache_dir=cache_dir,
        force_cache=force_cache,
        outlier_list=outlier_list,
        negative_image_root=neg_root_str,
        shared_cache=train_cache,
    )

    val_dataset = DocDataset(
        image_root=str(data_root / "images"),
        label_root=str(data_root / "labels"),
        split_file=str(data_root / val_split),
        img_size=img_size,
        augment=False,
        cache_images=cache_images,
        cache_dir=cache_dir,
        force_cache=force_cache,
        negative_image_root=neg_root_str,
        shared_cache=val_cache,
    )

    # Setup sampler for outliers
    sampler = None
    if outlier_list:
        outlier_path = Path(outlier_list)
        if outlier_path.exists():
            with open(outlier_path) as f:
                outlier_names = {line.strip() for line in f if line.strip()}
            weights = [outlier_weight if name in outlier_names else 1.0 for name in train_dataset.image_list]
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            train_dataset.outlier_names = outlier_names

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    import sys

    data_root = Path(__file__).parent.parent / "doc-scanner-dataset-labeled"

    if not data_root.exists():
        print(f"Dataset not found at {data_root}")
        sys.exit(1)

    dataset = DocDataset(
        image_root=str(data_root / "images"),
        label_root=str(data_root / "labels"),
        split_file=str(data_root / "train.txt"),
        img_size=192,
        augment=True,
    )

    print(f"\nDataset size: {len(dataset)}")

    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  coords: {sample['coords']}")
        print(f"  score: {sample['score']:.3f}")
        print(f"  has_label: {sample['has_label']}")
