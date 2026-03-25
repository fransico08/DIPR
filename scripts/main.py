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

from config import MODEL_PATH, ROAD_WIDTH_M, ROAD_LENGTH_M, DEFAULT_IMG_PTS, VEHICLE_CLASSES, CONF_THRESHOLD, IOU_THRESHOLD, YOLO_IMGSZ, DETECT_EVERY, MAX_AGE, N_INIT, MAX_IOU_DIST, SPEED_WINDOW, MIN_HISTORY, SPEED_SMOOTH

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
        
        t0, x0, y0 = h[0];
        t1, x1, y1 = h[-1]
        dt = (t1-t0) / 1000.0
        if dt <= 0:
            return self._smo.get(tid, 0.0)
        
        spd = math.hypot(x1 - x0, y1 - y0) / dt * 3.6
        s = SPEED_SMOOTH * spd + (1 - SPEED_SMOOTH) * self._smo.get(tid, spd)
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

def run_pipeline(video_path, calibrator):
    global _cmap
    _cmap = {}

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

    app = AppWindow(
        video_w=disp_w, video_h=disp_h,
        on_reset=do_reset, on_pause=do_pause, on_quit=do_quit,
        on_screenshot=do_screenshot, on_load_video=do_load,
        on_seek=do_seek,
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
            speed = speed_est.update(int(tid), ((l + r) // 2, b), ts_ms)
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