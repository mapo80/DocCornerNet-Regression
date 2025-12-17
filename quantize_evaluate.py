"""
Quantize TFLite model to INT8 and evaluate accuracy.
"""

import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tqdm import tqdm

# ImageNet normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DATA_ROOT = Path("/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-labeled")
IMG_SIZE = 224
NUM_CALIBRATION = 200


def load_image(img_path: Path) -> np.ndarray:
    """Load and preprocess image for model input."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD
    return img_array


def load_label(label_path: Path) -> np.ndarray:
    """Load ground truth coordinates from YOLO OBB format."""
    with open(label_path) as f:
        parts = f.read().strip().split()
        coords = np.array([float(x) for x in parts[1:9]], dtype=np.float32)
    return coords


def get_calibration_dataset():
    """Generator for calibration data."""
    val_file = DATA_ROOT / "val.txt"
    with open(val_file) as f:
        image_names = [line.strip() for line in f if line.strip()]

    # Use subset for calibration
    np.random.seed(42)
    np.random.shuffle(image_names)
    calibration_names = image_names[:NUM_CALIBRATION]

    for img_name in calibration_names:
        base_name = img_name.rsplit('.', 1)[0] if '.' in img_name else img_name
        img_path = DATA_ROOT / "images" / f"{base_name}.jpg"
        if not img_path.exists():
            img_path = DATA_ROOT / "images" / f"{base_name}.png"
        if not img_path.exists():
            continue

        img_array = load_image(img_path)
        yield [img_array[np.newaxis, ...].astype(np.float32)]


def quantize_model(saved_model_dir: Path, output_path: Path):
    """Quantize SavedModel to INT8 TFLite."""
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))

    # Full integer quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = get_calibration_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.float32  # Keep output as float for coords

    print("Quantizing model...")
    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    print(f"Quantized model saved: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
    return output_path


def quantize_dynamic_range(saved_model_dir: Path, output_path: Path):
    """Dynamic range quantization (simpler, weights-only)."""
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    print("Applying dynamic range quantization...")
    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    print(f"Dynamic range model saved: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
    return output_path


def compute_iou(pred_coords: np.ndarray, gt_coords: np.ndarray) -> float:
    """Compute IoU between predicted and ground truth quadrilaterals."""
    from shapely.geometry import Polygon

    pred_points = [(pred_coords[i*2], pred_coords[i*2+1]) for i in range(4)]
    gt_points = [(gt_coords[i*2], gt_coords[i*2+1]) for i in range(4)]

    try:
        pred_poly = Polygon(pred_points)
        gt_poly = Polygon(gt_points)
        if not pred_poly.is_valid or not gt_poly.is_valid:
            return 0.0
        intersection = pred_poly.intersection(gt_poly).area
        union = pred_poly.union(gt_poly).area
        return intersection / union if union > 0 else 0.0
    except:
        return 0.0


def evaluate_tflite(tflite_path: Path, split: str = "test", max_samples: int = None):
    """Evaluate TFLite model accuracy."""
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_dtype = input_details['dtype']
    input_shape = input_details['shape']

    # Check if quantized input
    is_quantized_input = input_dtype == np.uint8
    if is_quantized_input:
        input_scale = input_details['quantization'][0]
        input_zero_point = input_details['quantization'][1]

    split_file = DATA_ROOT / f"{split}.txt"
    with open(split_file) as f:
        image_names = [line.strip() for line in f if line.strip()]

    if max_samples:
        image_names = image_names[:max_samples]

    ious = []
    corner_errors = []

    for img_name in tqdm(image_names, desc=f"Evaluating {tflite_path.name}"):
        base_name = img_name.rsplit('.', 1)[0] if '.' in img_name else img_name

        img_path = DATA_ROOT / "images" / f"{base_name}.jpg"
        if not img_path.exists():
            img_path = DATA_ROOT / "images" / f"{base_name}.png"
        label_path = DATA_ROOT / "labels" / f"{base_name}.txt"

        if not img_path.exists() or not label_path.exists():
            continue

        # Load and preprocess
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        img_array = load_image(img_path)

        # Prepare input
        if is_quantized_input:
            # Quantize input: float -> uint8
            img_quantized = img_array / input_scale + input_zero_point
            img_quantized = np.clip(img_quantized, 0, 255).astype(np.uint8)
            input_data = img_quantized[np.newaxis, ...]
        else:
            input_data = img_array[np.newaxis, ...].astype(np.float32)

        # Run inference
        interpreter.set_tensor(input_details['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details['index'])[0]

        pred_coords = output[:8]
        gt_coords = load_label(label_path)

        # Compute IoU
        iou = compute_iou(pred_coords, gt_coords)
        ious.append(iou)

        # Compute corner errors in pixels
        for i in range(4):
            px = pred_coords[i*2] * orig_w
            py = pred_coords[i*2+1] * orig_h
            gx = gt_coords[i*2] * orig_w
            gy = gt_coords[i*2+1] * orig_h
            dist = np.sqrt((px - gx)**2 + (py - gy)**2)
            corner_errors.append(dist)

    return {
        'mean_iou': np.mean(ious) * 100,
        'recall_50': np.mean([iou >= 0.5 for iou in ious]) * 100,
        'recall_75': np.mean([iou >= 0.75 for iou in ious]) * 100,
        'recall_90': np.mean([iou >= 0.90 for iou in ious]) * 100,
        'corner_error_mean': np.mean(corner_errors),
        'corner_error_p50': np.percentile(corner_errors, 50),
        'corner_error_p95': np.percentile(corner_errors, 95),
        'num_samples': len(ious)
    }


def main():
    saved_model_dir = Path("checkpoints/tflite_224")

    # Paths
    fp32_path = saved_model_dir / "model_float32.tflite"
    fp16_path = saved_model_dir / "model_float16.tflite"
    dynamic_path = saved_model_dir / "model_dynamic_range.tflite"
    int8_path = saved_model_dir / "model_int8.tflite"

    # Quantize
    print("\n" + "="*60)
    print("QUANTIZATION")
    print("="*60)

    if not dynamic_path.exists():
        quantize_dynamic_range(saved_model_dir, dynamic_path)

    if not int8_path.exists():
        quantize_model(saved_model_dir, int8_path)

    # Print sizes
    print("\n" + "="*60)
    print("MODEL SIZES")
    print("="*60)
    for path in [fp32_path, fp16_path, dynamic_path, int8_path]:
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"{path.name}: {size_kb:.1f} KB ({size_kb/1024:.2f} MB)")

    # Evaluate all models
    print("\n" + "="*60)
    print("ACCURACY EVALUATION (test set)")
    print("="*60)

    results = {}
    for name, path in [("FP32", fp32_path), ("FP16", fp16_path),
                       ("Dynamic", dynamic_path), ("INT8", int8_path)]:
        if path.exists():
            print(f"\n--- {name} ---")
            results[name] = evaluate_tflite(path, split="test")
            r = results[name]
            print(f"  Mean IoU:     {r['mean_iou']:.2f}%")
            print(f"  Recall@50:    {r['recall_50']:.2f}%")
            print(f"  Recall@75:    {r['recall_75']:.2f}%")
            print(f"  Recall@90:    {r['recall_90']:.2f}%")
            print(f"  Corner Error: {r['corner_error_mean']:.2f} px (P50: {r['corner_error_p50']:.2f}, P95: {r['corner_error_p95']:.2f})")

    # Summary table
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Model':<12} {'Size (KB)':<12} {'IoU':<10} {'R@90':<10} {'Corner Err':<12}")
    print("-"*60)
    for name, path in [("FP32", fp32_path), ("FP16", fp16_path),
                       ("Dynamic", dynamic_path), ("INT8", int8_path)]:
        if path.exists() and name in results:
            size_kb = path.stat().st_size / 1024
            r = results[name]
            print(f"{name:<12} {size_kb:<12.1f} {r['mean_iou']:<10.2f} {r['recall_90']:<10.2f} {r['corner_error_mean']:<12.2f}")


if __name__ == "__main__":
    main()
