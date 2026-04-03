# =============================================================
#  main.py — Detection / tracking pipeline
#  Run in Spyder: F5
#  Tools > Preferences > Working Directory > project root
#  Requirements: pip install ultralytics deep-sort-realtime opencv-python numpy Pillow
#
#  Architecture: ONE tk.Tk() lives in UnifiedWindow.
#  The pipeline runs in a background thread so Tk is never blocked.
#  Thread communication: UnifiedWindow.post_frame() / post_stats() queues.
# =============================================================

import cv2
import numpy as np
import time
import os
import queue
import threading
from collections import deque

from ultralytics import YOLO

from config import (
    MODEL_PATH, CONF_THRESHOLD, IOU_THRESHOLD, YOLO_IMGSZ,
    DETECT_EVERY, VEHICLE_CLASSES,
    FPS_BUF_SIZE, FPS_EPSILON,
)
from core.calibration     import HomographyCalibrator
from core.speed_estimator import SpeedEstimator
from core.tracker         import make_tracker, compute_embeddings
from ui.unified_window    import UnifiedWindow
from utils.frame_reader   import FrameReader
from utils.drawing        import (get_color, clear_color_cache,
                                  draw_track, draw_roi, draw_hud)
from utils.screen         import get_screen_size, fit_to_screen


def _point_in_quad(pt, quad_pts):
    """
    Check if a point (x, y) is inside the calibration quadrilateral.
    quad_pts order: [TL, TR, BL, BR]
    Uses cv2.pointPolygonTest for robustness.
    """
    if not quad_pts or len(quad_pts) < 4:
        return True  # no calibration = allow all
    # Reorder to convex polygon: TL, TR, BR, BL
    tl, tr, bl, br = [np.array(p, dtype=np.float32) for p in quad_pts]
    poly = np.array([tl, tr, br, bl], dtype=np.float32)
    result = cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False)
    return result >= 0


# =============================================================
#  PIPELINE  (runs in a daemon thread)
# =============================================================

class Pipeline:
    """
    Wraps the YOLO + DeepSORT + speed-estimation loop.
    Communicates with UnifiedWindow via post_frame() / post_stats().
    Controlled via flags set by the UI callbacks.
    """

    def __init__(self, video_path: str, calibrator: HomographyCalibrator,
                 window: UnifiedWindow):
        self._video_path = video_path
        self._cal        = calibrator
        self._win        = window

        self._paused     = threading.Event()
        self._stopped    = threading.Event()
        self._reset_req  = threading.Event()

        # FIX: pipeline starts PAUSED until user is on tracking screen
        self._paused.set()

        self._scr_dir    = os.path.dirname(video_path)
        self._cur_frame  = None   # for screenshot

        self._thread     = threading.Thread(target=self._run, daemon=True)

    # ── Public control API (called from Tk thread) ─────────────

    def start(self):  self._thread.start()
    def stop(self):   self._stopped.set()

    def resume(self):
        """Called when user switches to tracking screen."""
        self._paused.clear()

    def pause(self):
        if self._paused.is_set():
            self._paused.clear()
        else:
            self._paused.set()

    def set_paused(self, paused: bool):
        if paused:
            self._paused.set()
        else:
            self._paused.clear()

    def reset(self):
        """FIX: Properly reset and unpause so replay works after video ends."""
        # Unpause FIRST so the loop can reach the reset block
        self._paused.clear()
        self._reset_req.set()

    def seek(self, frame_no: int):
        self._seek_to = frame_no
        self._reset_req.set()

    def screenshot(self):
        if self._cur_frame is not None:
            ts  = time.strftime("%Y%m%d_%H%M%S")
            dst = os.path.join(self._scr_dir, f"screenshot_{ts}.png")
            cv2.imwrite(dst, self._cur_frame)
            print(f"[INFO] Screenshot -> {dst}")

    @property
    def reader(self): return self._reader

    # ── Pipeline loop ─────────────────────────────────────────

    def _run(self):
        print(f"[INFO] Loading model: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        try:
            import torch
            use_cuda = torch.cuda.is_available()
        except Exception:
            use_cuda = False
        print(f"[INFO] CUDA: {use_cuda}")

        self._reader = FrameReader(self._video_path)
        scr_w, scr_h = get_screen_size()
        from config import SIDEBAR_W
        disp_w, disp_h = fit_to_screen(
            self._reader.width, self._reader.height,
            scr_w - SIDEBAR_W, scr_h)

        self._reader.start()
        self._seek_to = None

        # Notify window of video size
        self._win.init_screens(
            self._reader.width, self._reader.height,
            self._reader.total_frames, self._reader.fps_src)

        tracker   = make_tracker()
        speed_est = SpeedEstimator(self._cal)
        clear_color_cache()

        frame_id = 0
        dets: list = []
        fps_buf = deque(maxlen=FPS_BUF_SIZE)
        t_prev  = time.perf_counter()

        while not self._stopped.is_set():

            # ── Pause ──────────────────────────────────────────
            if self._paused.is_set():
                self._win.post_stats(0, 0, self._cal.is_calibrated(),
                                     True, False, 0)
                time.sleep(0.05)
                continue

            # ── Reset / seek ───────────────────────────────────
            if self._reset_req.is_set():
                self._reset_req.clear()
                tracker   = make_tracker()
                speed_est = SpeedEstimator(self._cal)
                clear_color_cache()
                if self._seek_to is not None:
                    self._reader.seek(self._seek_to)
                    self._seek_to = None
                else:
                    # FIX: Full replay — use replay() which reopens the cap
                    self._reader.replay()
                frame_id = 0
                dets = []

            # ── Read ───────────────────────────────────────────
            # is_done() is checked AFTER reset_req so that replay()
            # can clear the stopped flag before we loop back here.
            if self._reader.is_done():
                self._win.post_stats(0, 0, self._cal.is_calibrated(),
                                     False, True, self._reader.total_frames)
                time.sleep(0.05)
                continue

            try:
                frame, ts_ms, fno = self._reader.read()
            except queue.Empty:
                continue

            frame_id += 1
            frame = cv2.resize(frame, (disp_w, disp_h))

            # ── Detection ──────────────────────────────────────
            if frame_id % DETECT_EVERY == 0:
                res  = model(frame, verbose=False,
                             conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
                             imgsz=YOLO_IMGSZ, half=use_cuda)[0]
                dets = [
                    ([x1, y1, x2-x1, y2-y1], float(box.conf[0]), int(box.cls[0]))
                    for box in res.boxes
                    if int(box.cls[0]) in VEHICLE_CLASSES
                    for x1, y1, x2, y2 in [map(int, box.xyxy[0])]
                ]

            # ── Tracking ───────────────────────────────────────
            embeds = compute_embeddings(frame, dets)
            tracks = tracker.update_tracks(dets, embeds=embeds)

            cal_pts = self._cal.image_points if self._cal.is_calibrated() else None

            active = 0
            for track in tracks:
                if not track.is_confirmed():
                    continue

                l, t, r, b = map(int, track.to_ltrb())
                tid  = track.track_id
                cid  = getattr(track, "det_class", 2)
                color = get_color(tid)

                # Bottom-center point = vehicle foot position
                cx, cy = (l + r) // 2, b

                # FIX: only track/show speed for vehicles inside calibration zone
                if cal_pts and not _point_in_quad((cx, cy), cal_pts):
                    # Draw dimmed box for out-of-zone vehicles
                    draw_track(frame, l, t, r, b, tid,
                               VEHICLE_CLASSES.get(cid, "vehicle"), None, color,
                               in_zone=False)
                    continue

                active += 1
                speed = speed_est.update(
                    int(tid), (cx, cy), ts_ms)
                draw_track(frame, l, t, r, b, tid,
                           VEHICLE_CLASSES.get(cid, "vehicle"), speed, color,
                           in_zone=True)

            draw_roi(frame, self._cal.image_points)

            t_now = time.perf_counter()
            fps_buf.append(1.0 / max(t_now - t_prev, FPS_EPSILON))
            t_prev = t_now
            fps = float(np.mean(fps_buf))

            draw_hud(frame, fps, active, self._cal.is_calibrated(), False)

            self._cur_frame = frame.copy()
            self._win.post_frame(frame)
            self._win.post_stats(fps, active, self._cal.is_calibrated(),
                                 False, False, fno)

        self._reader.stop()
        print("[INFO] Pipeline stopped.")


# =============================================================
#  APP CONTROLLER
# =============================================================

class AppController:
    """
    Owns the UnifiedWindow + Pipeline and wires them together.
    Handles "load new video" by restarting the pipeline.
    """

    def __init__(self):
        self._video_path = None
        self._calibrator = None
        self._pipeline   = None
        self._window     = None

    def run(self) -> None:
        # FIX: Start with window first, then let user pick video
        self._window = UnifiedWindow(
            on_quit      = self._on_quit,
            on_load_new  = self._on_load_new,
        )
        # Enter Tk event loop (blocks until window is closed)
        self._window.run()

    def _start_pipeline(self):
        if self._pipeline:
            self._pipeline.stop()

        self._pipeline = Pipeline(
            video_path = self._video_path,
            calibrator = self._calibrator,
            window     = self._window,
        )
        self._pipeline.start()

        # Wire UI callbacks
        self._window.set_pipeline_callbacks(
            on_pause      = self._pipeline.pause,
            on_resume     = self._pipeline.resume,
            on_reset      = self._pipeline.reset,
            on_screenshot = self._pipeline.screenshot,
            on_seek       = self._pipeline.seek,
        )

    def _on_load_new(self, new_path: str):
        """Called from Tk thread when user picks a new video."""
        self._video_path = new_path
        self._calibrator = HomographyCalibrator()
        self._window._video_path  = new_path
        self._window._calibrator  = self._calibrator
        import cv2
        cap = cv2.VideoCapture(new_path)
        video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Build screens BEFORE accessing UI elements
        self._window.init_screens(video_w, video_h, total, fps)
        self._window._cal_status.set("New video loaded. Adjust params & Confirm.")

        # Stop old pipeline if running
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None

        # Reload first frame for calibration preview
        self._window._load_cal_first_frame()
        self._window._render_cal(None)
        self._window.show_calibration()

        # Start new pipeline (starts paused)
        self._pipeline = Pipeline(
            video_path = self._video_path,
            calibrator = self._calibrator,
            window     = self._window,
        )
        self._window.set_pipeline_callbacks(
            on_pause      = self._pipeline.pause,
            on_resume     = self._pipeline.resume,
            on_reset      = self._pipeline.reset,
            on_screenshot = self._pipeline.screenshot,
            on_seek       = self._pipeline.seek,
        )
        self._pipeline.start()

    def _on_quit(self):
        if self._pipeline:
            self._pipeline.stop()


# =============================================================
#  ENTRY POINT
# =============================================================

def main():
    AppController().run()


if __name__ == "__main__":
    main()
