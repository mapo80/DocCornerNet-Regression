"""
Real-time video inference with TFLite model.

Usage:
    python inference_video.py --video path/to/video.mp4
    python inference_video.py --video path/to/video.mp4 --output output.mp4
    python inference_video.py --camera 0  # webcam
"""

import argparse
from pathlib import Path
import time

import cv2
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Corner colors: TL=red, TR=green, BR=blue, BL=yellow
CORNER_COLORS = [
    (0, 0, 255),    # TL - Red
    (0, 255, 0),    # TR - Green
    (255, 0, 0),    # BR - Blue
    (0, 255, 255),  # BL - Yellow
]
CORNER_NAMES = ['TL', 'TR', 'BR', 'BL']


class TFLiteInference:
    def __init__(self, model_path: str, img_size: int = 224):
        self.img_size = img_size

        if tf is None:
            raise ImportError("TensorFlow is required")

        self.interpreter = tf.lite.Interpreter(
            model_path=model_path,
            num_threads=4
        )
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        print(f"Model loaded: {model_path}")
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Output shape: {self.output_details[0]['shape']}")

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for inference."""
        # Resize
        img = cv2.resize(frame, (self.img_size, self.img_size))
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        # Add batch dimension (NHWC format for TFLite)
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, frame: np.ndarray) -> tuple:
        """Run inference on a frame.

        Returns:
            corners: [(x, y), ...] 4 corner points in pixel coordinates
            score: confidence score [0, 1]
            inference_time: time in ms
        """
        h, w = frame.shape[:2]

        # Preprocess
        input_data = self.preprocess(frame)

        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        start = time.perf_counter()
        self.interpreter.invoke()
        inference_time = (time.perf_counter() - start) * 1000

        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        # Parse output: [x0, y0, x1, y1, x2, y2, x3, y3, score]
        coords = output[:8]
        score = output[8]

        # Convert normalized coords to pixel coordinates
        corners = []
        for i in range(4):
            x = coords[i * 2] * w
            y = coords[i * 2 + 1] * h
            corners.append((int(x), int(y)))

        return corners, float(score), inference_time


def draw_detection(frame: np.ndarray, corners: list, score: float,
                   inference_time: float, threshold: float = 0.5,
                   model_name: str = "") -> np.ndarray:
    """Draw detection results on frame."""
    h, w = frame.shape[:2]

    # Draw corners and polygon if confident
    if score >= threshold:
        # Draw filled polygon with transparency
        overlay = frame.copy()
        pts = np.array(corners, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)

        # Draw polygon outline
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

        # Draw corners with labels
        for i, (corner, color, name) in enumerate(zip(corners, CORNER_COLORS, CORNER_NAMES)):
            cv2.circle(frame, corner, 10, color, -1)
            cv2.circle(frame, corner, 12, (255, 255, 255), 2)
            # Label
            label_pos = (corner[0] + 15, corner[1] - 8)
            cv2.putText(frame, name, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, color, 2)

    # Info panel - larger and more visible
    panel_h = 100
    panel_w = 300
    cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (255, 255, 255), 2)

    # Model name
    if model_name:
        cv2.putText(frame, model_name, (20, 32),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Score (large, percentage)
    score_pct = score * 100
    score_color = (0, 255, 0) if score >= threshold else (0, 0, 255)
    cv2.putText(frame, f"Confidence: {score_pct:.1f}%", (20, 58),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2)

    # FPS (large)
    fps = 1000 / inference_time if inference_time > 0 else 0
    cv2.putText(frame, f"FPS: {fps:.0f}", (20, 85),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Inference time (smaller)
    cv2.putText(frame, f"({inference_time:.1f} ms)", (120, 85),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    return frame


def main():
    parser = argparse.ArgumentParser(description="Real-time video inference")
    parser.add_argument("--video", type=str, help="Path to input video")
    parser.add_argument("--camera", type=int, help="Camera device index")
    parser.add_argument("--model", type=str,
                       default="checkpoints/doccornernet_v2/doccornernet_int8.tflite",
                       help="Path to TFLite model")
    parser.add_argument("--output", type=str, help="Output video path (optional)")
    parser.add_argument("--img_size", type=int, default=224, help="Model input size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    parser.add_argument("--no_display", action="store_true", help="Don't show window")
    args = parser.parse_args()

    if args.video is None and args.camera is None:
        print("Error: specify --video or --camera")
        return

    # Load model
    model = TFLiteInference(args.model, args.img_size)

    # Open video source
    if args.video:
        cap = cv2.VideoCapture(args.video)
        source_name = Path(args.video).name
    else:
        cap = cv2.VideoCapture(args.camera)
        source_name = f"Camera {args.camera}"

    if not cap.isOpened():
        print(f"Error: cannot open {source_name}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nSource: {source_name}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.1f}")
    if total_frames > 0:
        print(f"Frames: {total_frames}")

    # Setup output video
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Output: {args.output}")

    # Get model name for display
    model_name = Path(args.model).stem

    print("\nPress 'q' to quit, 'p' to pause\n")

    frame_count = 0
    total_inference_time = 0
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference
            corners, score, inference_time = model.predict(frame)
            total_inference_time += inference_time
            frame_count += 1

            # Draw results
            result_frame = draw_detection(frame, corners, score, inference_time,
                                         args.threshold, model_name)

            # Write to output
            if writer:
                writer.write(result_frame)

        # Display
        if not args.no_display:
            cv2.imshow('DocCornerNet Inference', result_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print("Paused" if paused else "Resumed")

    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # Stats
    if frame_count > 0:
        avg_time = total_inference_time / frame_count
        print(f"\n{'='*50}")
        print(f"Processed {frame_count} frames")
        print(f"Average inference: {avg_time:.2f} ms")
        print(f"Average FPS: {1000/avg_time:.1f}")
        if args.output:
            print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
