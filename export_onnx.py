"""
Export DocCornerNet-Enhanced to ONNX format.

Usage:
    python export_onnx.py --checkpoint checkpoints/best.pth --output model.onnx
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import onnx

from model import create_model


class DocCornerNetExport(nn.Module):
    """Wrapper model that outputs combined tensor [coords, sigmoid(score)]."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        coords, score = self.model(x)
        score_expanded = torch.sigmoid(score).unsqueeze(-1)
        return torch.cat([coords, score_expanded], dim=-1)


def export_to_onnx(model: nn.Module, img_size: int, onnx_path: Path, opset: int = 13):
    """Export PyTorch model to ONNX format."""
    model.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None
    )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    print(f"ONNX opset version: {opset}")
    print(f"Input: {onnx_model.graph.input[0].name}, shape: [1, 3, {img_size}, {img_size}]")
    print(f"Output: {onnx_model.graph.output[0].name}, shape: [1, 9]")

    return onnx_model


def main():
    parser = argparse.ArgumentParser(description="Export DocCornerNet-Enhanced to ONNX")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth",
                        help="Path to PyTorch checkpoint")
    parser.add_argument("--output", type=str, default="model.onnx", help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    config = checkpoint.get('config', {})
    img_size = config.get('img_size', 320)
    width_mult = config.get('width_mult', 1.0)
    coord_activation = config.get('coord_activation', 'clamp')
    dropout = config.get('dropout', 0.2)

    print(f"Model config: img_size={img_size}, width_mult={width_mult}, coord_activation={coord_activation}")

    model = create_model(
        img_size=img_size,
        width_mult=width_mult,
        pretrained=False,
        dropout=dropout,
        coord_activation=coord_activation
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    export_model = DocCornerNetExport(model)
    export_model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    print(f"\nExporting to ONNX: {output_path}")
    export_to_onnx(export_model, img_size, output_path, args.opset)

    onnx_size = output_path.stat().st_size / (1024 * 1024)
    print(f"\nONNX model size: {onnx_size:.2f} MB")
    print(f"\n Export complete: {output_path}")


if __name__ == "__main__":
    main()
