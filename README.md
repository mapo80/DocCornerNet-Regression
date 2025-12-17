# DocCornerNet - Regression

A lightweight PyTorch model for document corner detection using **direct coordinate regression** with MobileNetV3-Small backbone.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-DocCornerDataset-yellow)](https://huggingface.co/datasets/mapo80/DocCornerDataset)

## Results

### Validation Set Performance

| Model | Input Size | Mean IoU | Median IoU | Corner Error | Recall@50 | Recall@75 | Recall@90 |
|-------|------------|----------|------------|--------------|-----------|-----------|-----------|
| **best_320** | 320×320 | **96.11%** | 96.93% | **1.45 px** | 99.89% | 99.31% | **96.95%** |
| best_224 | 224×224 | 95.82% | 96.76% | 1.61 px | 99.87% | 99.31% | 96.34% |

*All models trained with Clamp activation + Wing Loss + Geometry Loss*

### Model Specifications

| Metric | Value |
|--------|-------|
| **Parameters** | ~1M |
| **Model Size (FP32)** | 3.84 MB |
| **Model Size (INT8)** | 1.12 MB |
| **Compression** | 3.4× |

## Dataset

This model is trained on the **DocCornerDataset** available on HuggingFace:

**[mapo80/DocCornerDataset](https://huggingface.co/datasets/mapo80/DocCornerDataset)**

### Download

```bash
pip install huggingface_hub
huggingface-cli download mapo80/DocCornerDataset --repo-type dataset --local-dir ./data
```

## Architecture

DocCornerNet uses **direct coordinate regression**:

```
Input Image [B, 3, H, H] (224 or 320)
       ↓
MobileNetV3-Small Backbone (927K params)
       ↓
Global Average Pooling → [B, 576]
       ↓
MLP Head: Linear(576→128) → Hardswish → Dropout → Linear(128→9)
       ↓
Output: coords [B, 8] + score [B, 1]
```

### Key Features

- **MobileNetV3-Small backbone**: Pretrained on ImageNet
- **Wing Loss**: Better gradients for small errors (landmark localization)
- **Geometry Loss**: Prevents quad shrinking/expanding
- **Clamp activation**: Recommended for border accuracy (vs sigmoid)
- **~1M parameters**: Optimized for mobile deployment

## Installation

```bash
git clone https://github.com/mapo80/DocCornerNet-Regression.git
cd DocCornerNet-Regression
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py \
    --data_root /path/to/dataset \
    --img_size 320 \
    --coord_activation clamp \
    --use_wing_loss \
    --lambda_geometry 0.1 \
    --augment_preset strong \
    --batch_size 128 \
    --num_epochs 200 \
    --lr 0.001 \
    --warmup_epochs 10 \
    --scheduler cosine \
    --early_stopping_patience 30 \
    --use_iou_for_best \
    --output_dir ./checkpoints
```

**Training Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_root` | required | Path to dataset |
| `--img_size` | 224 | Input image size (224 or 320) |
| `--coord_activation` | sigmoid | Coordinate activation: sigmoid, clamp |
| `--use_wing_loss` | False | Use Wing Loss for coordinates |
| `--lambda_geometry` | 0.1 | Weight for geometry loss |
| `--augment_preset` | default | Augmentation: default, geo_light, strong |
| `--batch_size` | 32 | Batch size |
| `--num_epochs` | 100 | Training epochs |
| `--lr` | 1e-3 | Learning rate |
| `--warmup_epochs` | 0 | LR warmup epochs |
| `--scheduler` | plateau | LR scheduler: plateau, cosine |
| `--use_iou_for_best` | False | Use IoU for best model selection |

### Evaluation

```bash
python evaluate.py \
    --checkpoint ./checkpoints/best.pth \
    --data_root /path/to/dataset \
    --split val \
    --batch_size 32
```

### Detailed Evaluation

```bash
python evaluate_detailed.py \
    --checkpoint ./checkpoints/best.pth \
    --data_root /path/to/dataset \
    --split test \
    --output_json results.json
```

### Inference

```bash
# Single image
python infer.py \
    --checkpoint ./checkpoints/best.pth \
    --input image.jpg \
    --output result.jpg

# Directory
python infer.py \
    --checkpoint ./checkpoints/best.pth \
    --input ./images/ \
    --output ./results/
```

### Export to ONNX

```bash
python export_onnx.py \
    --checkpoint ./checkpoints/best.pth \
    --output model.onnx \
    --img_size 320
```

## Loss Functions

### Total Loss

```
L_total = λ_coords × L_coords + λ_geometry × L_geometry + λ_score × L_score
```

Default weights: λ_coords=1.0, λ_geometry=0.1, λ_score=2.0

### Wing Loss (Recommended)

Stronger gradients for small errors:

| Error (px) | SmoothL1 | Wing Loss | Improvement |
|------------|----------|-----------|-------------|
| 0.1 | 0.1 | 0.48 | 4.8× |
| 1.0 | 0.5 | 3.33 | 6.7× |
| 5.0 | 1.0 | 5.56 | 5.6× |
| 10.0 | 1.0 | 6.93 | 6.9× |

### Geometry Loss

Prevents quad deformation using:
- **Area Loss**: Shoelace formula for quad area matching
- **Edge Length Loss**: Match predicted vs GT edge lengths

## Output Format

### Corner Order

```
(x0, y0) = Top-Left      (x1, y1) = Top-Right
(x3, y3) = Bottom-Left   (x2, y2) = Bottom-Right

coords = [x0, y0, x1, y1, x2, y2, x3, y3]  # normalized [0, 1]
```

### Model Output

- `coords`: [B, 8] - Normalized corner coordinates
- `score`: [B] - Document presence logit (apply sigmoid)

## TFLite Conversion

```bash
# PyTorch → ONNX → TFLite
python export_onnx.py --checkpoint best.pth --output model.onnx

# Use onnx2tf for TFLite conversion
pip install onnx2tf
onnx2tf -i model.onnx -o saved_model
```

### Model Sizes

| Format | Size | Compression |
|--------|------|-------------|
| FP32 TFLite | 3.84 MB | 1× |
| INT8 (Dynamic Range) | 1.12 MB | 3.4× |

## Files

```
├── model.py              # MobileNetV3-Small + regression head
├── dataset.py            # PyTorch Dataset with augmentations
├── metrics.py            # IoU, corner error, recall metrics
├── train.py              # Training script with Wing/Geometry loss
├── evaluate.py           # Basic evaluation
├── evaluate_detailed.py  # Comprehensive evaluation
├── infer.py              # Single image inference
├── inference_video.py    # Video inference
├── export_onnx.py        # ONNX export
├── train_hft_ema*.py     # EMA training variants
├── quantize_evaluate.py  # TFLite quantization evaluation
├── diagnostics.py        # Model diagnostics
├── visualize_outliers.py # Outlier analysis
├── refine_quad.py        # Quad refinement utilities
└── README.md
```

## References

- [MobileNetV3](https://arxiv.org/abs/1905.02244) - Howard et al., ICCV 2019
- [Wing Loss](https://arxiv.org/abs/1711.06753) - Feng et al., CVPR 2018
- [Focal Loss](https://arxiv.org/abs/1708.02002) - Lin et al., ICCV 2017

## License

MIT License - see [LICENSE](LICENSE) file.

## Citation

```bibtex
@misc{doccornernet_regression2024,
  title={DocCornerNet-Regression: Direct Coordinate Regression for Document Corner Detection},
  year={2024},
  url={https://github.com/mapo80/DocCornerNet-Regression}
}
```
