# Meso Fibre Characterisation

A multi-subsystem Python pipeline for automated fibre detection, reconstruction, measurement, and characterisation from microscopy images. The system uses deep learning (Mask R-CNN) combined with classical computer vision techniques to segment individual fibres, bridge fragmented detections, and measure each fibre's length and width in physical units (mm).

---

## Repo Rules

1. There should be zero conflict because of the way the repo is structured, so always update your branch before you commit.
2. `[20260405 - This commit is...]` is the standard for commit messages.

---

## System Architecture

Three subsystems communicate via asynchronous message queues, orchestrated by a central launcher:

```
SS3 (Image Capture)
    │  image_data_message (image path + calibration metadata)
    ▼
SS4 (Image Processing)
    │  processing_result (per-fibre measurements)
    ▼
SS5 (Modelling)  ── optional
```

`main.py` spawns each subsystem as a daemon process and wires their message queues together. After each cycle SS4 (or SS5) sends a `ready_message` back to SS3 to trigger the next image.

---

## Subsystems

### SS3 — Image Capture (`src/ss3/ss3.py`)
Emulates an image capture device. Maintains a stack of test images with associated calibration metadata (physical pixel size in mm). On receiving a `ready_message` it pops the next image and forwards it to SS4 via `image_data_message`. Sends a `no_images` message when the stack is exhausted.

### SS4 — Image Processing (`src/ss4/ss4.py`)
The main processing hub. Runs four stages in sequence for each incoming image:

| Stage | Module | Description |
|---|---|---|
| Segmentation | `seg/infer.py` | Mask R-CNN (ResNet-50-FPN) produces per-fibre masks, bounding boxes, scores, centroids, and orientations |
| Reconstruction | `recon/fibre_reconstruction.py` | Bridges fragmented detections by dilating lines between close components; marks reconstructed regions for exclusion |
| Measurement | `meas/fibre_measure.py` | Skeletonises each mask for length; casts perpendicular rays for width; converts pixels → mm using calibration |
| Persistence | `common/db.py` | Writes per-fibre `(length_mm, width_mm)` to SQLite; optionally broadcasts via WebSocket |

Results are also aggregated into a size-distribution plot (length histogram, width histogram, scatter, summary stats).

### SS5 — Modelling (`src/ss5/ss5.py`)
Receives the processed characterisation data from SS4. Currently a stub — intended for downstream statistical modelling. Enable/disable via `SS5_ENABLED` in `src/common/config.py`.

---

## Project Structure

```
meso_fibre_char/
├── src/
│   ├── main.py                  # Orchestrator — launches all subsystems
│   ├── common/
│   │   ├── common.py            # Node class, message types, SharedImage
│   │   ├── config.py            # Feature flags and routing config
│   │   ├── db.py                # SQLite helpers
│   │   └── paths.py             # Centralised path constants
│   ├── ss3/ss3.py               # Image capture subsystem
│   ├── ss4/
│   │   ├── ss4.py               # Processing subsystem entry point
│   │   ├── seg/
│   │   │   ├── model.py         # Mask R-CNN model definition
│   │   │   └── infer.py         # Inference + Fibre dataclass
│   │   ├── recon/
│   │   │   └── fibre_reconstruction.py
│   │   ├── meas/
│   │   │   └── fibre_measure.py
│   │   └── size_distribution.py
│   └── ss5/ss5.py               # Modelling subsystem (stub)
├── data/
│   ├── images/                  # Input microscopy images (PNG)
│   └── db/results.db            # SQLite results database (auto-created)
├── train_yolov8.py              # Optional YOLOv8 training script
└── train_solov2.py              # Optional SOLOv2 training script
```

---

## Installation

**Python 3.10+ recommended.**

Install core dependencies:

```bash
pip install torch torchvision
pip install opencv-python numpy scipy scikit-image matplotlib Pillow
pip install websockets pycocotools
```

For training scripts (optional):

```bash
pip install ultralytics          # YOLOv8
pip install mmengine mmcv mmdet  # SOLOv2
```

---

## Running

```bash
python -m src.main
```

The system will process all images in `data/images/` in sequence, writing results to `data/db/results.db`.

A trained Mask R-CNN checkpoint must be present at:

```
src/ss4/seg/runs/fibre_maskrcnn/best.pth
```

---

## Configuration

Edit `src/common/config.py` to change runtime behaviour:

| Flag | Default | Effect |
|---|---|---|
| `SS5_ENABLED` | `True` | Enable the modelling subsystem |
| `READY_DEPENDENCIES` | — | Which subsystems must be ready before triggering the next cycle |
| `DOWNSTREAM_PROCESSORS` | — | Which subsystems receive processing results from SS4 |

---

## Outputs

- **Database** — `data/db/results.db` with tables `ss4_results` (image_id, mesh_id, length_mm, width_mm) and `ss5_results`.
- **Size distribution plot** — saved per batch showing length/width histograms and summary statistics.
- **Debug visualisations** — per-image overlay PNGs, per-fibre grids, and JSON prediction files written to `src/ss4/seg/predictions/` and `src/ss4/meas/inf_dbg/`.

---

## Test Images

Ground-truth test images with known fibre dimensions are included:

| File | Length | Width |
|---|---|---|
| `data/images/test_fibre_gt_1.png` | 1.2 mm | 0.06 mm |
| `data/images/test_fibre_gt_2.png` | 1.2 mm | 0.06 mm |