#!/usr/bin/env python3
"""
YOLOv8 Tank Detection — Raspberry Pi Inference (ONNX Runtime)
=============================================================
Runs a custom-trained YOLOv8 ONNX model on Raspberry Pi.
No PyTorch or GPU required — lightweight CPU inference.

Usage:
    python3 detect.py                              # Default camera
    python3 detect.py --model best.onnx --source 0 # USB Camera
    python3 detect.py --source video.mp4           # Video file
    python3 detect.py --size 320                   # Fast mode (higher FPS)
    python3 detect.py --save output.mp4            # Save output video
"""

import cv2
import numpy as np
import argparse
import time

try:
    import onnxruntime as ort
except ImportError:
    print("[ERROR] onnxruntime is not installed!")
    print("  Install: pip3 install onnxruntime")
    exit(1)


# ─── Preprocessing ───

def letterbox(img, new_shape=(640, 640)):
    """Resize image with aspect ratio preservation and padding."""
    h, w = img.shape[:2]
    ratio = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(w * ratio), int(h * ratio))

    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2

    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, ratio, (dw, dh)


def preprocess(frame, input_size=640):
    """Prepare frame for ONNX model input."""
    img, ratio, pad = letterbox(frame, (input_size, input_size))

    # BGR → RGB, HWC → CHW, normalize to [0, 1]
    img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
    img /= 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    return img, ratio, pad


# ─── Postprocessing ───

def postprocess(output, ratio, pad, conf_threshold=0.50, iou_threshold=0.45):
    """Convert ONNX output to bounding boxes (YOLOv8 format)."""
    # YOLOv8 output: [1, 5+num_classes, 8400] → transpose → [8400, 5+num_classes]
    predictions = output[0].transpose()

    boxes = []
    scores = []
    class_ids = []

    for pred in predictions:
        cx, cy, w, h = pred[:4]
        class_scores = pred[4:]
        max_score = np.max(class_scores)

        if max_score < conf_threshold:
            continue

        class_id = np.argmax(class_scores)

        # Center to corner format (xywh → xyxy)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        boxes.append([x1, y1, x2, y2])
        scores.append(float(max_score))
        class_ids.append(int(class_id))

    if len(boxes) == 0:
        return [], [], []

    # Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(
        np.array(boxes).tolist(),
        np.array(scores).tolist(),
        conf_threshold,
        iou_threshold,
    )

    if len(indices) == 0:
        return [], [], []

    indices = indices.flatten()

    # Scale coordinates back to original image
    dw, dh = pad
    final_boxes = []
    for i in indices:
        x1, y1, x2, y2 = boxes[i]
        x1 = (x1 - dw) / ratio
        y1 = (y1 - dh) / ratio
        x2 = (x2 - dw) / ratio
        y2 = (y2 - dh) / ratio
        final_boxes.append([int(x1), int(y1), int(x2), int(y2)])

    return final_boxes, [scores[i] for i in indices], [class_ids[i] for i in indices]


# ─── Visualization ───

def draw_detections(frame, boxes, scores, class_ids, class_names=None):
    """Draw bounding boxes and labels on the frame."""
    colors = [
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 165, 0),  # Orange
        (255, 0, 255),  # Purple
    ]

    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        color = colors[cls_id % len(colors)]

        if class_names and cls_id < len(class_names):
            label = f"{class_names[cls_id]}: {score:.0%}"
        else:
            label = f"Class {cls_id}: {score:.0%}"

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return frame


# ─── Main ───

def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8 Tank Detection — Raspberry Pi ONNX Inference"
    )
    parser.add_argument("--model", type=str, default="best.onnx",
                        help="Path to ONNX model file")
    parser.add_argument("--source", type=str, default="0",
                        help="Camera index (0, 1) or video file path")
    parser.add_argument("--conf", type=float, default=0.50,
                        help="Confidence threshold (0.0 - 1.0)")
    parser.add_argument("--size", type=int, default=640,
                        help="Input size (320=faster, 640=more accurate)")
    parser.add_argument("--show", action="store_true", default=True,
                        help="Display detection window")
    parser.add_argument("--no-show", action="store_true", default=False,
                        help="Headless mode (no display)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save output to video file (e.g., output.mp4)")
    args = parser.parse_args()

    if args.no_show:
        args.show = False

    # Class names (must match data.yaml order)
    class_names = ["tank"]

    # ── ONNX Runtime Session ──
    print(f"[INFO] Loading model: {args.model}")

    # Optimize for Raspberry Pi CPU
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4       # RPi 4/5 → 4 cores
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        args.model, sess_options,
        providers=['CPUExecutionProvider']
    )

    input_name = session.get_inputs()[0].name
    print(f"[INFO] Model input: {input_name} ({args.size}x{args.size})")

    # ── Camera / Video Source ──
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {args.source}")
        print("  Try: --source 0  (USB camera)")
        print("  Try: --source /dev/video0  (Pi camera)")
        exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Video writer
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save, fourcc, 15.0, (640, 480))
        print(f"[INFO] Saving output to: {args.save}")

    print(f"[INFO] Detection started | Conf: {args.conf:.0%} | Size: {args.size}")
    print(f"[INFO] Press 'q' to quit.\n")

    fps_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Source ended.")
            break

        t_start = time.time()

        # Preprocess
        input_tensor, ratio, pad = preprocess(frame, args.size)

        # ONNX Inference
        outputs = session.run(None, {input_name: input_tensor})

        # Postprocess
        boxes, scores, class_ids = postprocess(
            outputs[0], ratio, pad,
            conf_threshold=args.conf,
            iou_threshold=0.45,
        )

        # Draw detections
        frame = draw_detections(frame, boxes, scores, class_ids, class_names)

        # FPS calculation
        fps = 1.0 / (time.time() - t_start + 1e-9)
        fps_list.append(fps)
        avg_fps = sum(fps_list[-30:]) / len(fps_list[-30:])

        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if len(boxes) > 0:
            cv2.putText(frame, f"Detected: {len(boxes)} tank(s)", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            print(f"  [DETECT] {len(boxes)} tank(s) | FPS: {fps:.1f}")

        # Display
        if args.show:
            cv2.imshow("YOLOv8 Tank Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Save
        if writer:
            writer.write(frame)

    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    avg = sum(fps_list) / max(len(fps_list), 1)
    print(f"\n[INFO] Average FPS: {avg:.1f}")


if __name__ == "__main__":
    main()
