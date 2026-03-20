# DIPR
Update all in4 from subject as project
# 🚗 Vehicle Speed Estimation Using a Single Camera and Computer Vision Techniques

> **Course:** Digital Image Processing (DIPR)  
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
- [Dataset](#dataset)
- [Methods & Theory](#methods--theory)
- [Expected Results](#expected-results)
- [Limitations](#limitations)
- [Team](#team)

---

## Overview

This project estimates the real-world speed (km/h) of vehicles in traffic video using **a single fixed camera** — no radar, no LiDAR, no additional physical sensors required.

The system processes video frames through a complete DIP/CV pipeline: preprocessing → detection → tracking → geometric transformation → speed calculation.

**Key applications:**
- Intelligent Transportation Systems (ITS)
- Traffic speed violation monitoring
- Urban traffic flow analysis

---

## Pipeline

```
[Video Input]
     ↓
[Preprocessing]
  - Resize, Gaussian Blur
  - CLAHE (contrast enhancement)
  - Background Subtraction (MOG2)
  - Morphological Processing (Dilation, Closing)
     ↓
[Vehicle Detection — YOLOv8]
  - Pretrained on COCO
  - Classes: car, motorcycle, bus, truck
  - Confidence threshold ≥ 0.5
     ↓
[Vehicle Tracking — DeepSORT]
  - Kalman Filter (motion prediction)
  - Hungarian Algorithm (assignment)
  - Appearance descriptor (re-identification)
     ↓
[Perspective Transformation]
  - Camera calibration via 4 reference points
  - Homography matrix H (cv2.getPerspectiveTransform)
  - Bird's Eye View → pixel-to-meter ratio
     ↓
[Speed Calculation]
  - Euclidean distance in real-world coordinates
  - v = (d / Δt) × 3.6  [km/h]
  - Rolling average smoothing
     ↓
[Output: Annotated Video with BBox + ID + Speed]
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.1.4 |
| IDE | Visual Studio Code |
| Object Detection | Ultralytics YOLOv8 |
| Object Tracking | DeepSORT |
| Image Processing | OpenCV |
| Deep Learning Backend | PyTorch |
| Numerical Computing | NumPy |
| Camera | Rapoo C200 (720p, 30 FPS) |

---

## Project Structure

```
vehicle-speed-estimation/
│
├── main.py                  # Entry point
├── config.py                # Thresholds, paths, camera params
│
├── modules/
│   ├── preprocessing.py     # Image enhancement, background subtraction
│   ├── detection.py         # YOLOv8 inference wrapper
│   ├── tracking.py          # DeepSORT integration
│   ├── perspective.py       # Homography & Bird's Eye View
│   └── speed.py             # Distance & speed calculation
│
├── data/
│   ├── videos/              # Input video files
│   └── reference_points/    # Calibration reference configs
│
├── models/
│   └── yolov8n.pt           # YOLOv8 pretrained weights
│
├── output/
│   └── results/             # Annotated output videos
│
├── requirements.txt
└── README.md
```

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/fransico08/DIPR
cd DIPR
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
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
```

---

## Usage

**Run on a video file:**
```bash
python main.py --source data/videos/traffic.mp4
```

**Run on webcam (real-time):**
```bash
python main.py --source 0
```

**Key parameters in `config.py`:**
```python
CONFIDENCE_THRESHOLD = 0.5      # YOLO detection threshold
ALLOWED_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
FPS = 30                        # Camera frame rate
PIXEL_TO_METER = 0.05           # Calibrated scale (adjust per scene)
ROLLING_WINDOW = 10             # Frames for speed smoothing
```

**Camera calibration** — Edit reference points in `config.py`:
```python
# 4 points on road with known real-world distance
SRC_POINTS = [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]  # pixel coords
DST_POINTS = [(0,0), (W,0), (W,H), (0,H)]            # real-world coords (meters)
```

---

## Dataset

| Source | Description |
|---|---|
| Self-recorded | Captured with Rapoo C200, fixed mount, daylight conditions. Reference markers measured manually on road. |
| [UA-DETRAC](https://detrac-db.rit.albany.edu) | Real-world traffic dataset, fixed cameras, multiple vehicle types |
| [KITTI](https://www.cvlibs.net/datasets/kitti) | Diverse lighting & vehicle conditions for additional testing |

---

## Methods & Theory

### Image Processing (DIP Core)
- **Gaussian Blur** — noise reduction before detection
- **CLAHE** — adaptive histogram equalization for uneven lighting
- **MOG2 Background Subtraction** — foreground mask extraction
- **Morphological Processing** — dilation & closing to clean binary mask

### Perspective Transformation
Camera captures a 2D projection of the 3D world. To measure real distances, we apply a **Homography transform** that converts the angled camera view into a Bird's Eye View (BEV):

$$H: \text{pixel coords} \rightarrow \text{real-world coords (meters)}$$

Computed via `cv2.getPerspectiveTransform()` using 4 road reference points.

### Speed Calculation
Given tracked position of vehicle across frames:

$$v = \frac{d(P_{t_1}, P_{t_2})}{\Delta t} \times 3.6 \quad \text{[km/h]}$$

where $d$ is Euclidean distance in real-world space (meters), $\Delta t = n/\text{FPS}$ (seconds).

### Object Detection — YOLOv8
Single-pass detector outputting bounding boxes, confidence scores, and class labels per frame. Filters applied for traffic-relevant classes only.

### Object Tracking — DeepSORT
Extends SORT with deep appearance features for robust re-identification. Uses **Kalman Filter** to predict next position and **Hungarian Algorithm** for optimal detection-track assignment.

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
- **Reference point dependency** — incorrect field measurements propagate systematic error throughout all speed outputs

---

## Team

| Member | Responsibility |
|---|---|
| Bùi Duy Phong | Report , Video Acquisition, Detection module |
| Lê Minh Ngọc | Report , Tracking module (DeepSORT) |
| Huỳnh Minh Tài | Report , Perspective Transform, Speed Calculation, Slides |

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
- Institution: Ho Chi Minh University of Technology and Engineering
- Instructor: Hoàng Văn Dũng
- References: Digital Image Processing (DIPR430685E)

## 📞 Contact

For questions about this project, plz contact to email:
- 19110131@student.hcmute.edu.vn
- 22110056@student.hcmute.edu.vn
- 22110068@student.hcmute.edu.vn    

> *Final project by group 2 for Digital Image Processing — HCMUTE, 2025–2026*
