# =============================================================
#  VEHICLE SPEED ESTIMATION
#  Run in Spyder: F5
#  Tools > Preferences > Working Directory > DIPR/scripts
#  Requirements: pip install ultralytics deep-sort-realtime opencv-python numpy Pillow
# =============================================================

import cv2
import numpy as np
import math
import time
import os
import queue
from collections import deque, defaultdict

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

import tkinter as tk
from tkinter import filedialog
from ui.app_window import AppWindow
from ui.calibration_window import run_calibration_window
from utils.frame_reader import FrameReader
from utils.drawing import get_color, draw_track, draw_roi, draw_hud
from utils.screen import get_screen_size, fit_to_screen
from utils.ui_helpers import SIDEBAR_W

from config import (MODEL_PATH, ROAD_WIDTH_M, ROAD_LENGTH_M, DEFAULT_IMG_PTS,
                    VEHICLE_CLASSES, CONF_THRESHOLD, IOU_THRESHOLD, YOLO_IMGSZ,
                    DETECT_EVERY, MAX_AGE, N_INIT, MAX_IOU_DIST,
                    SPEED_WINDOW, MIN_HISTORY, SPEED_SMOOTH, SPEED_SCALE,
                    DEFAULT_CAM_HEIGHT_M, DEFAULT_CAM_TILT_DEG,
                    DEFAULT_CAM_FOV_H_DEG, DEFAULT_ROAD_SLOPE_DEG,
                    AUTO_ROI_TOP, AUTO_ROI_BOT)

# =============================================================
#  HSV HISTOGRAM EMBEDDER
# =============================================================

def compute_embeddings(frame, detections):
    H, W = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    out = []
    for det in detections:
        x, y, w, h = [int(v) for v in det[0]]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        if x2 <= x1 or y2 <= y1:
            out.append(np.zeros(96, dtype=np.float32))
            continue
        crop = hsv[y1:y2, x1:x2]
        feat = np.concatenate([
            cv2.calcHist([crop], [0], None, [32], [0,180]).flatten(),
            cv2.calcHist([crop], [1], None, [32], [0,256]).flatten(),
            cv2.calcHist([crop], [2], None, [32], [0,256]).flatten(),
        ])
        n = np.linalg.norm(feat)
        if n > 0:
            feat /= n
        out.append(feat)
    return out

# =============================================================
#  CALIBRATION
# =============================================================

class HomographyCalibrator:
    def __init__(self):
        self.image_points = None
        self._H           = None
        self.real_w       = ROAD_WIDTH_M
        self.real_l       = ROAD_LENGTH_M

    def is_calibrated(self): return self._H is not None

    def pixel_to_world(self, point):
        if self._H is None: return np.array([0.0, 0.0])
        return cv2.perspectiveTransform(
            np.array([[point]], dtype=np.float32), self._H)[0][0]

    def set_default(self):
        self._compute(np.float32(DEFAULT_IMG_PTS),
                      np.float32([[0,0],[ROAD_WIDTH_M,0],
                                  [0,ROAD_LENGTH_M],[ROAD_WIDTH_M,ROAD_LENGTH_M]]))
        print("[Calibration] Using default homography.")

    def save(self, path):
        np.savez(path, image_points=np.float32(self.image_points),
                 H=self._H, real_w=self.real_w, real_l=self.real_l)
        print(f"[Calibration] Saved -> {path}")

    def load(self, path):
        try:
            d = np.load(path)
            self.image_points = d["image_points"].tolist()
            self._H           = d["H"]
            self.real_w       = float(d.get("real_w", ROAD_WIDTH_M))
            self.real_l       = float(d.get("real_l", ROAD_LENGTH_M))
            print(f"[Calibration] Loaded -> {path}")
            return True
        except Exception: return False

    def apply_points(self, clicked, w_m, l_m):
        self.real_w, self.real_l = w_m, l_m
        self._compute(np.float32(clicked),
                      np.float32([[0,0],[w_m,0],[0,l_m],[w_m,l_m]]))
        print(f"[Calibration] Done! {w_m}m x {l_m}m | Points: {clicked}")

    def apply_camera_params(self, img_w, img_h,
                            cam_height_m=DEFAULT_CAM_HEIGHT_M,
                            tilt_deg=DEFAULT_CAM_TILT_DEG,
                            fov_h_deg=DEFAULT_CAM_FOV_H_DEG,
                            slope_deg=DEFAULT_ROAD_SLOPE_DEG):
        """
        Auto-compute homography from pinhole camera geometry.
        Width and length are derived analytically — no manual point click needed.

        World frame: X=right, Y=forward, Z=up
        Camera: X_c=right, Y_c=down(image), Z_c=forward(into scene)
        """
        theta = math.radians(tilt_deg)
        alpha = math.radians(slope_deg)

        # Focal length from horizontal FOV
        f  = (img_w / 2.0) / math.tan(math.radians(fov_h_deg) / 2.0)
        cx, cy = img_w / 2.0, img_h / 2.0

        # Rotation: camera → world
        # col0=X_c→world, col1=Y_c→world, col2=Z_c→world
        R = np.array([
            [1,  0,                   0               ],
            [0, -math.sin(theta),  math.cos(theta) ],
            [0, -math.cos(theta), -math.sin(theta) ],
        ], dtype=np.float64)

        C = np.array([0.0, 0.0, cam_height_m])

        # Road plane: n·P = 0 (passes through origin, slope alpha uphill in +Y)
        n_plane = np.array([0.0, -math.sin(alpha), math.cos(alpha)])
        nC = float(n_plane @ C)   # = cos(alpha) * cam_height_m

        def _project(u, v):
            d_cam   = np.array([(u - cx) / f, (v - cy) / f, 1.0])
            d_world = R @ d_cam
            denom   = float(n_plane @ d_world)
            if abs(denom) < 1e-9:
                return None
            t = -nC / denom
            if t < 0:
                return None
            P = C + t * d_world
            return (P[0], P[1])   # (X_right, Y_forward)

        # ROI corners: TL, TR, BL, BR
        img_pts = np.float32([
            [int(img_w * 0.10), int(img_h * AUTO_ROI_TOP)],
            [int(img_w * 0.90), int(img_h * AUTO_ROI_TOP)],
            [int(img_w * 0.10), int(img_h * AUTO_ROI_BOT)],
            [int(img_w * 0.90), int(img_h * AUTO_ROI_BOT)],
        ])

        world_xys = []
        for u, v in img_pts:
            wp = _project(float(u), float(v))
            if wp is None:
                print("[AutoCal] Cannot project pixel to road — check tilt/height. Using default.")
                self.set_default()
                return
            world_xys.append(wp)

        # Normalise world coords: shift minimum to (0,0).
        # NOTE: top-image pixels project to LARGER Y (farther from camera), so
        # we must NOT assume any ordering — use min/max over all 4 points.
        world_pts  = np.float32(world_xys)          # (4,2): TL, TR, BL, BR
        min_xy     = world_pts.min(axis=0)
        world_norm = world_pts - min_xy

        self.real_w = float(world_pts[:, 0].max() - world_pts[:, 0].min())
        self.real_l = float(world_pts[:, 1].max() - world_pts[:, 1].min())

        if self.real_w <= 0 or self.real_l <= 0:
            print(f"[AutoCal] Degenerate projection (W={self.real_w:.2f} L={self.real_l:.2f}). Using default.")
            self.set_default()
            return

        print(f"[AutoCal] Raw world XYs → TL:{world_xys[0]} TR:{world_xys[1]} "
              f"BL:{world_xys[2]} BR:{world_xys[3]}")
        self._compute(img_pts, world_norm)
        print(f"[AutoCal] W={self.real_w:.2f}m  L={self.real_l:.2f}m | "
              f"h={cam_height_m}m  tilt={tilt_deg}°  fov={fov_h_deg}°  slope={slope_deg}°")

    def _compute(self, img_pts, real_pts):
        self.image_points = [list(p) for p in img_pts]
        self._H, _ = cv2.findHomography(img_pts, real_pts, cv2.RANSAC, 5.0)
        if self._H is None:
            print("[Calibration] WARNING: failed. Using default.")
            self.set_default()

# =============================================================
#  SPEED ESTIMATOR
# =============================================================

class SpeedEstimator:
    def __init__(self, cal):
        self.cal   = cal
        self._hist = defaultdict(lambda: deque(maxlen=SPEED_WINDOW+1))
        self._smo  = {}

    def update(self, tid, px, ts):
        if not self.cal.is_calibrated():
            return 0.0
        wx, wy = self.cal.pixel_to_world(px)
        self._hist[tid].append((ts, wx, wy))
        h = self._hist[tid]
        if len(h) < MIN_HISTORY:
            return 0.0

        # Linear regression over all points in window: pos(t) = v*t + c
        # Slope v (m/s) is more robust to tracker jitter than using only endpoints.
        ts_s = np.array([e[0] for e in h], dtype=np.float64) / 1000.0
        xs   = np.array([e[1] for e in h], dtype=np.float64)
        ys   = np.array([e[2] for e in h], dtype=np.float64)
        ts_s -= ts_s[0]                   # relative time, starting at 0

        dt = ts_s[-1]
        if dt <= 0:
            return self._smo.get(tid, 0.0)

        # Fit vx, vy via least squares  [A] [vx] = [xs]
        #                               [ ] [vy]   [ys]
        A  = np.column_stack([ts_s, np.ones(len(ts_s))])
        coeff, _, _, _ = np.linalg.lstsq(A, np.column_stack([xs, ys]), rcond=None)
        vx, vy = float(coeff[0, 0]), float(coeff[0, 1])   # m/s

        spd = math.hypot(vx, vy) * 3.6                    # km/h
        s   = SPEED_SMOOTH * spd + (1 - SPEED_SMOOTH) * self._smo.get(tid, spd)
        self._smo[tid] = s
        return s

    def reset(self):
        self._hist.clear();
        self._smo.clear()

# =============================================================
#  TRACKER FACTORY
# =============================================================

def make_tracker():
    return DeepSort(max_age = MAX_AGE, n_init = N_INIT,
                    max_iou_distance = MAX_IOU_DIST,
                    max_cosine_distance = 0.4,
                    embedder = None, half = False, bgr = True)

# =============================================================
#  MAIN PIPELINE
# =============================================================

def run_pipeline(video_path, calibrator, speed_scale=None):
    global _cmap
    _cmap = {}

    # Mutable container so inner callbacks can mutate it
    _scale = [SPEED_SCALE if speed_scale is None else speed_scale]

    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    try:
        import torch
        use_cuda = torch.cuda.is_available()
    except Exception: use_cuda = False
    print(f"[INFO] CUDA: {use_cuda}")

    reader = FrameReader(video_path)
    scr_w, scr_h = get_screen_size()
    disp_w, disp_h = fit_to_screen(reader.width, reader.height,
                                    scr_w - SIDEBAR_W, scr_h, margin = 80)
    reader.start()

    tracker = make_tracker()
    speed_est = SpeedEstimator(calibrator)

    paused = False
    stopped = False
    load_new = False
    frame_id = 0
    dets = []
    fps_buf = deque(maxlen=60)
    t_prev = time.perf_counter()
    scr_dir = os.path.dirname(video_path)
    cur_f = [None]
    cur_fno = [0]

    def do_reset():
        nonlocal tracker
        tracker = make_tracker()
        speed_est.reset()
        _cmap.clear()
        print("[INFO] Tracker reset.")

    def do_pause():
        nonlocal paused
        paused = not paused

    def do_quit():
        nonlocal stopped
        stopped = True

    def do_screenshot():
        if cur_f[0] is not None:
            p = os.path.join(scr_dir, f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png")
            cv2.imwrite(p, cur_f[0])
            print(f"[INFO] Screenshot -> {p}")

    def do_load():
        nonlocal load_new, stopped
        load_new = True
        stopped = True

    def do_seek(fn):
        reader.seek(fn)
        do_reset()

    def do_scale(v):
        try:
            _scale[0] = max(0.01, float(v))
            print(f"[INFO] Speed scale factor set to {_scale[0]:.3f}")
        except ValueError:
            pass

    app = AppWindow(
        video_w=disp_w, video_h=disp_h,
        on_reset=do_reset, on_pause=do_pause, on_quit=do_quit,
        on_screenshot=do_screenshot, on_load_video=do_load,
        on_seek=do_seek, on_scale=do_scale,
        scale_factor=_scale[0],
        total_frames=reader.total_frames, fps_src=reader.fps_src,
    )

    def _key(event):
        k = event.keysym.lower()
        if   k == "space":
            do_pause()
        elif k == "r":
            do_reset()
        elif k == "s":
            do_screenshot()
        elif k == "l":
            do_load()
        elif k == "escape":
            app._hard_quit()

    app.root.bind("<KeyPress>", _key)
    app.root.focus_force()

    while not stopped:
        if paused:
            try:
                app.update_stats(0, 0, calibrator.is_calibrated(),
                                 True, False, cur_fno[0])
            except Exception:
                stopped = True
            continue

        if reader.is_done():
            print("[INFO] Video finished.")
            paused = True
            continue

        try:
            frame, ts_ms, fno = reader.read()
        except queue.Empty:
            paused = True
            continue

        cur_fno[0] = fno
        frame_id += 1
        frame = cv2.resize(frame, (disp_w, disp_h))

        if frame_id % DETECT_EVERY == 0:
            res  = model(frame, verbose=False, conf = CONF_THRESHOLD,
                         iou = IOU_THRESHOLD, imgsz = YOLO_IMGSZ, half = use_cuda)[0]
            dets = []
            for box in res.boxes:
                cid = int(box.cls[0])
                if cid not in VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                dets.append(([x1, y1, x2 - x1, y2 - y1], float(box.conf[0]), cid))

        embeds = compute_embeddings(frame, dets)
        tracks = tracker.update_tracks(dets, embeds=embeds)

        active = 0
        for track in tracks:
            if not track.is_confirmed():
                continue
            active += 1
            l, t, r, b = map(int, track.to_ltrb())
            tid = track.track_id
            cid = getattr(track, "det_class", 2)
            color = get_color(tid)
            speed = speed_est.update(int(tid), ((l + r) // 2, b), ts_ms) * _scale[0]
            draw_track(frame, l, t, r, b, tid,
                       VEHICLE_CLASSES.get(cid,"vehicle"), speed, color)

        draw_roi(frame, calibrator.image_points)
        t_now = time.perf_counter()
        fps_buf.append(1.0 / max(t_now - t_prev, 1e-9))
        t_prev = t_now
        fps = float(np.mean(fps_buf))
        draw_hud(frame, fps, active, calibrator.is_calibrated(), paused)
        cur_f[0] = frame.copy()

        try:
            app.update_frame(frame)
            app.update_stats(fps, active, calibrator.is_calibrated(),
                             paused, False, fno)
        except Exception:
            stopped = True

    reader.stop()

    if load_new:
        try: app.destroy()
        except Exception: pass
        return "load_new"

    # Video finished — stay open until user closes
    if app._alive:
        try:
            app.update_stats(0, 0, calibrator.is_calibrated(),
                             False, True, cur_fno[0])
            app.root.mainloop()
        except Exception: pass

    print("[INFO] Done.")
    return "quit"

# =============================================================
#  STARTUP
# =============================================================

def main():
    # 1. Pick video
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="Select video file",
        filetypes=[("Video files","*.mp4 *.avi *.mov *.mkv *.wmv"),
                   ("All files","*.*")]
    )
    root.destroy()
    if not video_path:
        print("[INFO] No video selected. Exiting.")
        return

    # 2. Calibration window (no messagebox — goes straight to window)
    calibrator = HomographyCalibrator()
    run_calibration_window(video_path, calibrator)

    # 3. Pipeline loop
    while True:
        result = run_pipeline(video_path, calibrator)
        if result == "load_new":
            root2 = tk.Tk()
            root2.withdraw()
            video_path = filedialog.askopenfilename(
                title="Select video file",
                filetypes=[("Video files","*.mp4 *.avi *.mov *.mkv *.wmv"),
                           ("All files","*.*")]
            )
            root2.destroy()
            if not video_path:
                break
            calibrator = HomographyCalibrator()
            run_calibration_window(video_path, calibrator)
        else:
            break

# =============================================================
if __name__ == "__main__":
    main()