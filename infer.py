"""
Run inference on images and save visualizations with predicted bounding boxes.

Usage:
    python infer.py --checkpoint checkpoints/doccornernet_v2/best.pth --input ../dataset-test --output visualization_inference
    python infer.py --checkpoint checkpoints/doccornernet_v2/best.pth --input image.jpg --output output.jpg
"""

import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import torchvision.transforms.functional as TF

from model import create_model
from dataset import IMAGENET_MEAN, IMAGENET_STD


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on images")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input image or directory of images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="visualization_inference",
        help="Output directory or file",
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


def preprocess_image(image: Image.Image, img_size: int) -> torch.Tensor:
    """Preprocess image for model input."""
    # Resize
    image = image.resize((img_size, img_size), Image.BILINEAR)
    # To tensor and normalize
    tensor = TF.to_tensor(image)
    tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
    return tensor


def coords_to_points(coords: np.ndarray, width: int, height: int) -> list:
    """Convert normalized coords [8] to list of (x, y) pixel tuples."""
    points = []
    for i in range(0, 8, 2):
        x = int(coords[i] * width)
        y = int(coords[i + 1] * height)
        points.append((x, y))
    return points


def draw_polygon(draw: ImageDraw, points: list, color: str, width: int = 3):
    """Draw a closed polygon."""
    for i in range(4):
        p1 = points[i]
        p2 = points[(i + 1) % 4]
        draw.line([p1, p2], fill=color, width=width)


def draw_corners(draw: ImageDraw, points: list, color: str, radius: int = 6):
    """Draw corner points with labels."""
    labels = ["TL", "TR", "BR", "BL"]
    for i, (x, y) in enumerate(points):
        # Draw circle
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=color,
            outline="white",
            width=2,
        )
        # Draw label
        label = labels[i]
        draw.text((x + radius + 4, y - radius), label, fill=color)


def main():
    args = parse_args()
    device = get_device(args.device)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

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

    # Get input images
    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path]
    else:
        # Get all images in directory
        image_paths = list(input_path.glob("*.jpg")) + \
                      list(input_path.glob("*.jpeg")) + \
                      list(input_path.glob("*.png")) + \
                      list(input_path.glob("*.JPG")) + \
                      list(input_path.glob("*.JPEG")) + \
                      list(input_path.glob("*.PNG"))
        image_paths = sorted(image_paths)

    if not image_paths:
        print(f"No images found in {input_path}")
        return

    print(f"Found {len(image_paths)} images")

    # Setup output
    output_path = Path(args.output)
    if len(image_paths) == 1 and output_path.suffix in [".jpg", ".jpeg", ".png"]:
        # Single output file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_files = [output_path]
    else:
        # Output directory
        output_path.mkdir(parents=True, exist_ok=True)
        output_files = [output_path / f"{p.stem}_pred.jpg" for p in image_paths]

    print(f"Output: {output_path}")
    print()

    # Process images
    with torch.no_grad():
        for img_path, out_path in tqdm(zip(image_paths, output_files), total=len(image_paths), desc="Processing"):
            # Load image
            original_image = Image.open(img_path).convert("RGB")
            orig_width, orig_height = original_image.size

            # Preprocess
            tensor = preprocess_image(original_image, img_size)
            tensor = tensor.unsqueeze(0).to(device)

            # Inference
            pred_coords, pred_score = model(tensor)
            pred_coords = pred_coords[0].cpu().numpy()
            pred_score = torch.sigmoid(pred_score[0]).cpu().item()

            # Draw on original size image
            draw = ImageDraw.Draw(original_image)

            # Convert coords to original image size
            pred_points = coords_to_points(pred_coords, orig_width, orig_height)

            # Draw predicted polygon (red)
            draw_polygon(draw, pred_points, color="#FF0000", width=3)
            draw_corners(draw, pred_points, color="#FF0000", radius=8)

            # Add confidence percentage inside bbox (top-right corner)
            # pred_points[1] is TR (top-right)
            tr_x, tr_y = pred_points[1]
            conf_text = f"{pred_score*100:.1f}%"

            # Try to load a medium font, fallback to default
            try:
                font_size = max(14, min(orig_width, orig_height) // 40)
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
            except:
                font = ImageFont.load_default()

            # Get text bounding box for proper background sizing
            bbox = draw.textbbox((0, 0), conf_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Position text inside top-right corner with padding
            padding = 4
            text_x = tr_x - text_width - padding * 2
            text_y = tr_y + padding

            # Draw background rectangle
            bg_rect = [text_x - padding, text_y - padding,
                       text_x + text_width + padding, text_y + text_height + padding]
            draw.rectangle(bg_rect, fill=(0, 0, 0, 200))
            draw.text((text_x, text_y), conf_text, fill="#00FF00", font=font)

            # Save
            original_image.save(out_path, quality=95)

    print()
    print(f"Saved {len(image_paths)} visualizations")


if __name__ == "__main__":
    main()
