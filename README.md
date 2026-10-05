<div align="center">

# Tank Detection — YOLOv8 Edge AI System

**Real-time military vehicle detection powered by a custom-trained YOLOv8 model,
engineered for deployment on resource-constrained edge devices.**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-6C3CFC?style=for-the-badge&logo=ultralytics&logoColor=white)](https://docs.ultralytics.com)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-E67E22?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-4%2F5-C51A4A?style=for-the-badge&logo=raspberrypi&logoColor=white)](https://raspberrypi.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-2ECC71?style=for-the-badge)](LICENSE)

---

[Overview](#-overview) · [Architecture](#-architecture) · [Quick Start](#-quick-start) · [Training](#-training) · [Performance](#-performance-benchmarks) · [API Reference](#-cli-reference) · [License](#-license)

</div>

<br>

## 📋 Overview

This project implements an end-to-end pipeline for detecting military vehicles (tanks) in real-time video streams. The system is designed around a two-stage workflow:

1. **Training** — A YOLOv8n (nano) model is custom-trained on a curated dataset of ~8,800 labeled images using GPU acceleration.
2. **Inference** — The trained model is exported to ONNX format and deployed on a Raspberry Pi for lightweight, GPU-free, real-time detection.

| Property | Details |
|---|---|
| **Model** | YOLOv8n (nano) — custom-trained, single-class |
| **Dataset** | 8,798 labeled images (80/20 train/val split) |
| **Classes** | `tank` |
| **Training Hardware** | NVIDIA RTX 4070 Laptop GPU |
| **Inference Target** | Raspberry Pi 4/5 (CPU only, via ONNX Runtime) |
| **Input Resolution** | 640×640 (training) · 320×320 (fast inference mode) |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   TRAINING PIPELINE (GPU)                    │
│                                                             │
│   Labeled Dataset (8.8K images)                             │
│         │                                                   │
│         ▼                                                   │
│   train.py ─── YOLOv8n + RTX 4070 ──► best.pt              │
│                                           │                 │
│                                    ONNX Export              │
│                                           │                 │
│                                       best.onnx             │
└───────────────────────────┬─────────────────────────────────┘
                            │  Transfer (SCP / USB)
┌───────────────────────────▼─────────────────────────────────┐
│                INFERENCE PIPELINE (Edge CPU)                 │
│                                                             │
│   Camera Input ──► detect.py (ONNX Runtime)                 │
│                         │                                   │
│                         ▼                                   │
│            Real-time Bounding Boxes + Confidence            │
│                                                             │
│   ~6–8 FPS @ 320px (RPi 4)  ·  ~12–15 FPS @ 320px (RPi 5) │
└─────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
tank-detection-yolov8/
│
├── train.py                # GPU training script (dataset split + YOLOv8 fine-tuning)
├── detect.py               # Edge inference script (ONNX Runtime, no PyTorch needed)
├── data.yaml               # Dataset class & path configuration
├── requirements.txt        # Inference-only Python dependencies
├── .gitignore              # Git ignore rules (weights, datasets, IDE files)
├── LICENSE                 # MIT License
│
└── dataset_split/          # Generated train/val split (excluded from repo)
    ├── images/
    │   ├── train/          # ~7,038 training images
    │   └── val/            # ~1,760 validation images
    └── labels/
        ├── train/          # YOLO-format bounding box annotations
        └── val/
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- A USB camera or Pi Camera module (for live detection)
- A pre-trained ONNX model file (`best.onnx`)

### 1 — Clone the Repository

```bash
git clone https://github.com/<your-username>/tank-detection-yolov8.git
cd tank-detection-yolov8
```

### 2 — Install Dependencies

**Raspberry Pi (inference only):**

```bash
sudo apt update && sudo apt install -y python3-opencv
pip3 install -r requirements.txt
```

**PC / Workstation (for training):**

```bash
pip install ultralytics opencv-python torch torchvision
```

### 3 — Run Detection

```bash
# Live camera feed (USB camera, default)
python3 detect.py --model best.onnx --source 0

# Video file input
python3 detect.py --model best.onnx --source video.mp4

# Fast mode — lower resolution, higher throughput
python3 detect.py --model best.onnx --source 0 --size 320

# Record output to file
python3 detect.py --model best.onnx --source 0 --save output.mp4
```

Press **`q`** to quit the detection window.

---

## 🎓 Training

> **Note:** Training requires a CUDA-capable GPU (tested on NVIDIA RTX 4070 Laptop, 8 GB VRAM).

To retrain the model on your own dataset, place your images and YOLO-format labels under the source directory and run:

```bash
python train.py
```

The training script automatically handles dataset splitting (80/20), `data.yaml` generation, model training, validation, and ONNX/NCNN export.

### Training Configuration

| Parameter | Value | Rationale |
|---|---|---|
| **Epochs** | 100 | Sufficient convergence with early stopping |
| **Image Size** | 640×640 | High-resolution feature extraction |
| **Batch Size** | 16 | Optimized for 8 GB VRAM |
| **Optimizer** | AdamW | Better stability on smaller datasets vs. SGD |
| **Learning Rate** | 0.01 → 0.0001 (cosine) | Gradual decay with 5-epoch warmup |
| **Augmentation** | Mosaic, MixUp, HSV, Erasing, Flip | Robustness to lighting, occlusion, and scale |
| **Early Stopping** | 25-epoch patience | Prevents overfitting |

### Augmentation Strategy

The augmentation pipeline is specifically tuned for military vehicle detection in challenging environments:

- **HSV Value ±80%** — Simulates low-light and nighttime conditions
- **Random Erasing (50%)** — Handles partial occlusion by foliage or structures
- **Mosaic + MixUp** — Increases background diversity and reduces overfitting
- **Rotation ±15°** — Accounts for non-level camera angles
- **Scale 40–160%** — Simulates varying engagement distances

---

## ⚡ Performance Benchmarks

### Inference Latency (ONNX Runtime, CPU)

| Device | Resolution | FPS | Latency |
|---|---|---|---|
| Raspberry Pi 4 (4 GB) | 320×320 | ~6–8 | ~130 ms |
| Raspberry Pi 4 (4 GB) | 640×640 | ~2–3 | ~400 ms |
| Raspberry Pi 5 | 320×320 | ~12–15 | ~75 ms |
| Raspberry Pi 5 | 640×640 | ~5–7 | ~170 ms |

### Model Specifications

| Property | Value |
|---|---|
| Architecture | YOLOv8n (nano) |
| Parameters | 3.2 M |
| Model Size (ONNX) | ~6 MB |
| Confidence Threshold | 0.50 |
| NMS IoU Threshold | 0.45 |

---

## 🔧 CLI Reference

```
python3 detect.py [OPTIONS]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model` | `str` | `best.onnx` | Path to the ONNX model file |
| `--source` | `str` | `0` | Camera index (`0`, `1`, …) or path to a video file |
| `--conf` | `float` | `0.50` | Minimum confidence threshold (0.0–1.0) |
| `--size` | `int` | `640` | Input resolution (`320` for speed, `640` for accuracy) |
| `--show` | `flag` | `True` | Display the live detection window |
| `--no-show` | `flag` | `False` | Run in headless mode (no GUI) |
| `--save` | `str` | `None` | Save annotated output to a video file |

---

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| Object Detection | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) |
| Edge Inference | [ONNX Runtime](https://onnxruntime.ai) |
| Image Processing | [OpenCV](https://opencv.org) |
| Numerical Computing | [NumPy](https://numpy.org) |

---

## 🤝 Contributing

Contributions are welcome. Please open an [issue](../../issues) to discuss proposed changes before submitting a pull request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

<div align="center">
<sub>Built for edge AI deployment with YOLOv8 + ONNX Runtime</sub>
</div>
