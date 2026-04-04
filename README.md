# 🚗 Vehicle Speed Estimation Using a Single Camera and Computer Vision Techniques

> **Course:** Digital Image Processing (DIPR430685E)  
> **Institution:** HCMC University of Technology and Engineering (HCMUTE)  
> **Group:** 2 — Faculty of Information Technology

---

## 📋 Table of Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Calibration](#calibration)
- [Dataset](#dataset)
- [Methods & Theory](#methods--theory)
- [Expected Results](#expected-results)
- [Limitations](#limitations)
- [Team](#team)

---

## Overview

This project estimates the real-world speed (km/h) of vehicles in traffic video using **a single fixed camera** — no radar, no LiDAR, no additional physical sensors required.

The system processes video frames through a complete DIP/CV pipeline: detection → tracking → geometric transformation → speed calculation, with a full Tkinter GUI for interactive control.

**Key applications:**
- Intelligent Transportation Systems (ITS)
- Traffic speed violation monitoring
- Urban traffic flow analysis

---

## Pipeline

```
[Video Input — file or webcam]
     ↓
[FrameReader — threaded video decoding with queue]
     ↓
[Vehicle Detection — YOLOv8]
  - Pretrained on COCO (yolov8n.pt)
  - Classes: car, motorcycle, bus, truck
  - Confidence threshold ≥ 0.25
  - Runs every 3 frames (DETECT_EVERY)
     ↓
[HSV Histogram Embedding]
  - Crop detection region → HSV color space
  - 3-channel histogram (H:32 + S:32 + V:32 = 96-D)
  - L2 normalized → appearance feature for DeepSORT
     ↓
[Vehicle Tracking — DeepSORT]
  - Kalman Filter (motion prediction)
  - Hungarian Algorithm (optimal assignment)
  - HSV appearance descriptor (re-identification)
  - Track confirmed after 5 consecutive matches (N_INIT)
     ↓
[Perspective Transformation — Homography]
  - Manual: 4 reference points + known real-world size
  - Auto: camera parameters (height, tilt, FOV, slope)
  - Homography matrix H via cv2.findHomography()
  - Pixel coords → real-world coords (meters)
     ↓
[Speed Calculation — Linear Regression]
  - World positions collected in sliding window (12 frames)
  - Least-squares regression: pos(t) = v·t + c
  - speed = √(vx² + vy²) × 3.6  [km/h]
  - EMA smoothing (α = 0.5)
     ↓
[Output: Tkinter GUI with Annotated Video + Controls]
  - BBox + Track ID + Vehicle class + Speed (km/h)
  - ROI polygon overlay
  - HUD: FPS, tracked count, homography status
  - Sidebar: seekbar, pause, reset, screenshot, load video
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.x |
| IDE | Visual Studio Code |
| Object Detection | Ultralytics YOLOv8 (yolov8n.pt) |
| Object Tracking | DeepSORT (deep-sort-realtime) |
| Image Processing | OpenCV |
| Deep Learning Backend | PyTorch |
| Numerical Computing | NumPy |
| GUI | Tkinter + PIL/Pillow |
| 3D Preview | Matplotlib (optional, for calibration visualization) |

---

## Project Structure

```
DIPR-vehicle-speed-estimation/
│
├── scripts/
│   ├── main.py                      # Entry point — AppController, Pipeline (daemon thread)
│   ├── config.py                    # Tất cả hằng số cấu hình (không magic number ở file khác)
│   ├── yolov8n.pt                   # Trọng số YOLOv8-Nano (COCO pretrained)
│   │
│   ├── core/
│   │   ├── calibration.py           # HomographyCalibrator (Manual + Auto camera params)
│   │   ├── speed_estimator.py       # SpeedEstimator (5 lớp ổn định: px EMA → path → spike → EMA → deadband)
│   │   └── tracker.py               # make_tracker() + compute_embeddings() (DeepSORT + HSV 96D)
│   │
│   ├── ui/
│   │   ├── unified_window.py        # Cửa sổ chính hợp nhất (top bar + calibration + tracking)
│   │   ├── app_window.py            # Tracking GUI — video canvas + sidebar controls
│   │   └── calibration_window.py    # Calibration GUI standalone
│   │
│   └── utils/
│       ├── drawing.py               # draw_track, draw_roi (gradient), draw_hud, draw_roi_quad
│       ├── frame_reader.py          # Threaded video reader với queue + replay()
│       ├── screen.py                # get_screen_size() + fit_to_screen()
│       └── ui_helpers.py            # Shared Tkinter components (label, entry, button, param_row)
│
├── videos/                          # Video đầu vào + file *_cal.json (auto-generated)
│
├── .gitignore
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/fransico08/DIPR.git
cd DIPR
```

**2. Switch to the project branch**
```bash
git checkout new-vehicle-source-code-with-camera-degree
```

**3. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate           # Windows
source venv/bin/activate        # Linux/Mac
```

**4. Install dependencies**
```bash
pip install -r requirements.txt
```

**requirements.txt**
```
ultralytics
opencv-python
torch
torchvision
numpy
deep-sort-realtime
Pillow
matplotlib
```

---

## Usage

**Run the application:**
```bash
cd scripts
python main.py
```

1. A file dialog opens — select a video file (mp4, avi, mov, mkv, wmv)
2. The calibration window appears — choose Manual or Auto mode (see [Calibration](#calibration))
3. The main window starts processing with live annotated video

**Keyboard shortcuts (main window):**

| Key | Action |
|---|---|
| `SPACE` | Pause / Resume |
| `R` | Reset tracker |
| `S` | Save screenshot |
| `L` | Load new video |
| `ESC` | Quit |

---

## Calibration

The system supports two calibration modes to compute the Homography matrix:

### Manual Mode
1. Click 4 road points on the first frame:
   - 1 = Top-Left, 2 = Top-Right, 3 = Bottom-Left, 4 = Bottom-Right
2. Enter the real-world Width (m) and Length (m) of the selected rectangle
3. Press `ENTER` to confirm

**Best references:** lane markings, crosswalks, parking bays — any flat rectangle with known dimensions.

### Auto Mode (Camera Parameters)
Instead of clicking points, enter camera geometry:

| Parameter | Default | Description |
|---|---|---|
| Height above road (m) | 6.0 | Vertical distance from camera to road surface |
| Tilt / depression angle (°) | 30.0 | Angle below horizontal (0° = level, 90° = straight down) |
| Horizontal FOV (°) | 60.0 | Camera field of view — check camera specs or estimate |
| Road slope (°) | 0.0 | 0 = flat, positive = uphill away from camera |

- **Live preview** shows estimated W × L (meters) as you adjust parameters
- Press `P` to open **3D Geometry Preview** (matplotlib) — shows camera position, frustum rays, ROI footprint, tilt angle arc
- Press `ENTER` to confirm

### Other options
- **Use Previous Calibration** — loads `calibration.npz` from the video directory
- **Use Default** — applies hardcoded default points from `config.py`

---

## Dataset

| Source | Description |
|---|---|
| Self-recorded | Fixed mount, daylight conditions. Reference markers measured manually on road. |
| [Pexels stock videos](https://www.pexels.com) | Urban traffic footage from various angles for testing |
| [UA-DETRAC](https://detrac-db.rit.albany.edu) | Real-world traffic dataset, fixed cameras, multiple vehicle types |
| [KITTI](https://www.cvlibs.net/datasets/kitti) | Diverse lighting & vehicle conditions for additional testing |

---

## Methods & Theory

### Object Detection — YOLOv8
Single-pass CNN detector outputting bounding boxes, confidence scores, and class labels per frame. Uses YOLOv8-Nano for real-time performance (~27–31 FPS without GPU). Only vehicle classes are retained (car, motorcycle, bus, truck).

### Appearance Feature — HSV Histogram
Each detection is cropped, converted to HSV color space, and a 96-dimensional histogram vector is computed (32 bins per channel). HSV separates color from brightness, providing stable appearance features under varying lighting. This replaces heavier deep Re-ID models.

### Object Tracking — DeepSORT
Extends SORT with appearance features for robust re-identification:
- **Kalman Filter** — predicts next position from current velocity state
- **Hungarian Algorithm** — finds optimal detection-to-track assignment using combined motion (Mahalanobis) and appearance (cosine) distances
- **Track lifecycle** — new tracks require N_INIT=5 consecutive matches; lost tracks are removed after MAX_AGE=10 frames

### Perspective Transformation — Homography
Camera captures a 2D projection of the 3D world. A **Homography matrix H** (3×3) maps pixel coordinates to real-world coordinates (meters):

$$H: \text{pixel coords} \rightarrow \text{real-world coords (meters)}$$

- **Manual:** computed via `cv2.findHomography()` using 4 user-selected road reference points
- **Auto:** derived analytically from pinhole camera model — focal length from FOV, rotation matrix from tilt, ray-plane intersection for each ROI corner

### Speed Calculation — Linear Regression
Given tracked positions in world coordinates over a sliding window:

1. Collect (timestamp, world_x, world_y) for each track
2. Fit linear regression via least squares: `pos(t) = v·t + c`
3. Velocity components: `vx, vy` (m/s) = regression slopes
4. Speed: `√(vx² + vy²) × 3.6` → km/h
5. EMA smoothing: `s = α·speed_new + (1-α)·speed_old` with α = 0.5

Linear regression over all window points is more robust to tracker jitter than simple endpoint distance.

---

## Expected Results

| Metric | Target |
|---|---|
| MAE (Mean Absolute Error) | < 5 km/h (good lighting, stable camera) |
| MAPE | < 10% on most test cases |
| FPS (offline) | ≥ 20 FPS on standard hardware |

Speed ground truth obtained by manually timing vehicles across known road distance markers in self-recorded footage.

---

## Limitations

- **Occlusion** — overlapping vehicles may cause ID switching or lost tracks
- **Camera angle** — shallow angles reduce homography accuracy
- **Lateral movement** — speed estimation is most accurate for vehicles moving longitudinally (toward/away from camera)
- **Low FPS** — cameras below 15 FPS degrade speed accuracy
- **Night / adverse weather** — detection confidence drops without IR camera or dehazing preprocessing
- **Reference point dependency** — incorrect field measurements or camera parameters propagate systematic error throughout all speed outputs
- **Auto calibration** — requires accurate camera specs; stock/unknown videos may need manual tuning of height and tilt

---

## Team

| Member | Responsibility |
|---|---|
| Bùi Duy Phong | Report, Video Acquisition, Detection module |
| Lê Minh Ngọc | Report, Tracking module (DeepSORT) |
| Huỳnh Minh Tài | Report, Perspective Transform, Speed Calculation, Slides |

---

## References

- Redmon et al. (2016). *You Only Look Once: Unified, Real-Time Object Detection.* CVPR.
- Wojke et al. (2017). *Simple Online and Realtime Tracking with a Deep Association Metric.* ICIP.
- Gonzalez & Woods. *Digital Image Processing*, 4th ed. Pearson.
- Hartley & Zisserman. *Multiple View Geometry in Computer Vision*, 2nd ed. Cambridge.
- OpenCV Docs: https://docs.opencv.org
- Ultralytics YOLOv8: https://docs.ultralytics.com

---

## 🙏 Acknowledgments

- Course: Digital Image Processing
- Institution: Ho Chi Minh City University of Technology and Education (HCMUTE)
- Instructor: Hoàng Văn Dũng
- Course code: DIPR430685E

## 📞 Contact

For questions about this project, please contact:
- 19110131@student.hcmute.edu.vn
- 22110056@student.hcmute.edu.vn
- 22110068@student.hcmute.edu.vn

> *Final project by Group 2 for Digital Image Processing — HCMUTE, 2025–2026*
