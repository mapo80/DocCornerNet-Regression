"""
Visualize outliers with their augmentations (default vs geo_light).
"""

import random
import math
from pathlib import Path

from PIL import Image, ImageFilter, ImageDraw
import torchvision.transforms.functional as TF


def rotate_coords(coords: list, angle_deg: float, aspect_ratio: float = 1.0) -> list:
    """Rotate normalized coordinates around image center."""
    angle_rad = math.radians(-angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    rotated = []
    for i in range(0, 8, 2):
        x_norm = coords[i]
        y_norm = coords[i + 1]

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

    return rotated


def apply_augmentation(image: Image.Image, coords: list, config: dict):
    """Apply augmentation from dataset.py"""
    w, h = image.size
    aspect_ratio = w / h

    # Rotation
    angle = random.uniform(-config["rotation_degrees"], config["rotation_degrees"])
    image = image.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(128, 128, 128))
    coords = rotate_coords(coords, angle, aspect_ratio)

    # Scale
    scale = random.uniform(*config["scale_range"])
    if scale < 1.0:
        coords = [0.5 + (c - 0.5) * scale for c in coords]
        new_size = int(w * scale)
        scaled = image.resize((new_size, new_size), Image.BILINEAR)
        canvas = Image.new("RGB", (w, h), (128, 128, 128))
        offset = (w - new_size) // 2
        canvas.paste(scaled, (offset, offset))
        image = canvas

    # Brightness
    if config["brightness"] > 0:
        factor = random.uniform(1 - config["brightness"], 1 + config["brightness"])
        image = TF.adjust_brightness(image, factor)

    # Contrast
    if config["contrast"] > 0:
        factor = random.uniform(1 - config["contrast"], 1 + config["contrast"])
        image = TF.adjust_contrast(image, factor)

    # Saturation
    if config["saturation"] > 0:
        factor = random.uniform(1 - config["saturation"], 1 + config["saturation"])
        image = TF.adjust_saturation(image, factor)

    # Blur
    if random.random() < config["blur_prob"]:
        image = image.filter(ImageFilter.GaussianBlur(radius=config["blur_kernel"] / 2))

    coords = [max(0.0, min(1.0, c)) for c in coords]
    return image, coords


def draw_corners(image: Image.Image, coords: list):
    """Draw document corners on image."""
    w, h = image.size
    draw = ImageDraw.Draw(image)

    points = [(coords[i] * w, coords[i+1] * h) for i in range(0, 8, 2)]
    draw.polygon(points, outline=(0, 255, 0), width=3)

    labels = ["TL", "TR", "BR", "BL"]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    for pt, label, color in zip(points, labels, colors):
        x, y = int(pt[0]), int(pt[1])
        r = 6
        draw.ellipse([x-r, y-r, x+r, y+r], fill=color, outline=(255, 255, 255), width=2)

    return image


def create_comparison_grid(original: Image.Image, coords: list,
                           default_config: dict, geo_light_config: dict,
                           n_aug: int = 3) -> Image.Image:
    """Create grid: original | default augs | geo_light augs"""
    w, h = original.size

    # Grid layout: 1 original + n_aug default + n_aug geo_light
    total_cols = 1 + n_aug + n_aug
    grid = Image.new("RGB", (total_cols * w, h), (40, 40, 40))

    # Original
    orig_vis = draw_corners(original.copy(), coords)
    grid.paste(orig_vis, (0, 0))

    # Default augmentations
    for i in range(n_aug):
        aug_img, aug_coords = apply_augmentation(original.copy(), coords.copy(), default_config)
        aug_vis = draw_corners(aug_img, aug_coords)
        grid.paste(aug_vis, ((1 + i) * w, 0))

    # Geo_light augmentations
    for i in range(n_aug):
        aug_img, aug_coords = apply_augmentation(original.copy(), coords.copy(), geo_light_config)
        aug_vis = draw_corners(aug_img, aug_coords)
        grid.paste(aug_vis, ((1 + n_aug + i) * w, 0))

    # Add labels
    draw = ImageDraw.Draw(grid)
    draw.text((10, 10), "Original", fill=(255, 255, 255))
    draw.text((w + 10, 10), "Default Aug", fill=(255, 255, 0))
    draw.text(((1 + n_aug) * w + 10, 10), "Geo Light Aug", fill=(0, 255, 255))

    return grid


def main():
    project_root = Path(__file__).parent.parent.parent
    dataset_root = project_root / "datasets" / "official" / "doc-scanner-dataset-labeled"
    output_dir = Path(__file__).parent / "visualizations"

    image_root = dataset_root / "images"
    label_root = dataset_root / "labels"
    outlier_file = dataset_root / "outliers.txt"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Configs from dataset.py and train.py
    default_config = {
        "rotation_degrees": 5,
        "scale_range": (0.9, 1.0),
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.1,
        "blur_prob": 0.1,
        "blur_kernel": 3,
    }

    geo_light_config = {
        "rotation_degrees": 15,
        "scale_range": (0.7, 1.2),
        "brightness": 0.15,
        "contrast": 0.15,
        "saturation": 0.1,
        "blur_prob": 0.0,
        "blur_kernel": 3,
    }

    # Load outliers
    with open(outlier_file) as f:
        outliers = [line.strip() for line in f if line.strip()]

    print(f"Found {len(outliers)} outliers")

    # Sample some outliers
    n_samples = min(10, len(outliers))
    sampled = random.sample(outliers, n_samples)

    for image_name in sampled:
        image_path = image_root / image_name
        label_path = label_root / (Path(image_name).stem + ".txt")

        if not image_path.exists() or not label_path.exists():
            print(f"  Skipping {image_name} (not found)")
            continue

        # Load image and coords
        image = Image.open(image_path).convert("RGB")
        with open(label_path) as f:
            line = f.readline().strip()
        parts = line.split()
        coords = [float(x) for x in parts[1:9]]

        # Resize for visualization
        target_size = 300
        image = image.resize((target_size, target_size), Image.BILINEAR)

        # Create comparison grid
        grid = create_comparison_grid(image, coords, default_config, geo_light_config, n_aug=3)

        # Save
        sample_name = Path(image_name).stem
        output_path = output_dir / f"outlier_{sample_name}.png"
        grid.save(output_path)
        print(f"  Saved: {output_path.name}")

    print(f"\nDone! Output: {output_dir}")


if __name__ == "__main__":
    main()
