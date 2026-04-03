# =============================================================
#  main.py — Detection / tracking pipeline
#  Run in Spyder: F5
#  Tools > Preferences > Working Directory > project root
#  Requirements: pip install ultralytics deep-sort-realtime opencv-python numpy Pillow
# =============================================================

import cv2
import numpy as np
import time
import os
import queue
from collections import deque

from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog

from config import (
    MODEL_PATH, CONF_THRESHOLD, IOU_THRESHOLD, YOLO_IMGSZ,
    DETECT_EVERY, VEHICLE_CLASSES,
    SIDEBAR_W, FPS_BUF_SIZE, FPS_EPSILON,
)
from core.calibration    import HomographyCalibrator
from core.speed_estimator import SpeedEstimator
from core.tracker        import make_tracker, compute_embeddings
from ui.app_window       import AppWindow
from ui.calibration_window import run_calibration_window
from utils.frame_reader  import FrameReader
from utils.drawing       import get_color, clear_color_cache, draw_track, draw_roi, draw_hud
from utils.screen        import get_screen_size, fit_to_screen


# =============================================================
#  MAIN PIPELINE
# =============================================================

def run_pipeline(video_path: str,
                 calibrator: HomographyCalibrator) -> str:
    """
    Run the detection / tracking loop for one video.

    Returns
    -------
    "load_new"  — user requested a new video
    "quit"      — user closed the application
    """
    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    try:
        import torch
        use_cuda = torch.cuda.is_available()
    except Exception:
        use_cuda = False
    print(f"[INFO] CUDA: {use_cuda}")

    # ── Video reader ──────────────────────────────────────────
    reader = FrameReader(video_path)
    scr_w, scr_h = get_screen_size()
    disp_w, disp_h = fit_to_screen(
        reader.width, reader.height,
        scr_w - SIDEBAR_W, scr_h,
    )
    reader.start()

    # ── Core components ───────────────────────────────────────
    tracker   = make_tracker()
    speed_est = SpeedEstimator(calibrator)
    clear_color_cache()

    # ── State ─────────────────────────────────────────────────
    paused   = False
    stopped  = False
    load_new = False
    frame_id = 0
    dets:    list = []
    fps_buf  = deque(maxlen=FPS_BUF_SIZE)
    t_prev   = time.perf_counter()
    scr_dir  = os.path.dirname(video_path)
    cur_f    = [None]
    cur_fno  = [0]

    # ── Callbacks (pipeline → UI) ─────────────────────────────
    def do_reset() -> None:
        nonlocal tracker
        tracker = make_tracker()
        speed_est.reset()
        clear_color_cache()
        print("[INFO] Tracker reset.")

    def do_pause() -> None:
        nonlocal paused
        paused = not paused

    def do_quit() -> None:
        nonlocal stopped
        stopped = True

    def do_screenshot() -> None:
        if cur_f[0] is not None:
            ts  = time.strftime("%Y%m%d_%H%M%S")
            dst = os.path.join(scr_dir, f"screenshot_{ts}.png")
            cv2.imwrite(dst, cur_f[0])
            print(f"[INFO] Screenshot -> {dst}")

    def do_load() -> None:
        nonlocal load_new, stopped
        load_new = True
        stopped  = True

    def do_seek(frame_no: int) -> None:
        reader.seek(frame_no)
        do_reset()

    # ── Build window ──────────────────────────────────────────
    app = AppWindow(
        video_w      = disp_w,
        video_h      = disp_h,
        on_reset     = do_reset,
        on_pause     = do_pause,
        on_quit      = do_quit,
        on_screenshot = do_screenshot,
        on_load_video = do_load,
        on_seek      = do_seek,
        total_frames = reader.total_frames,
        fps_src      = reader.fps_src,
    )

    def _on_key(event: tk.Event) -> None:
        k = event.keysym.lower()
        if   k == "space":  do_pause()
        elif k == "r":      do_reset()
        elif k == "s":      do_screenshot()
        elif k == "l":      do_load()
        elif k == "escape": app._hard_quit()

    app.root.bind("<KeyPress>", _on_key)
    app.root.focus_force()

    # ── Main loop ─────────────────────────────────────────────
    while not stopped:

        # Pause: pump Tk events and wait for resume
        if paused:
            try:
                app.update_stats(0, 0, calibrator.is_calibrated(),
                                 True, False, cur_fno[0])
            except Exception:
                stopped = True
            continue

        # End of video: stay open, let user replay or load new
        if reader.is_done():
            print("[INFO] Video finished.")
            paused = True
            continue

        try:
            frame, ts_ms, fno = reader.read()
        except queue.Empty:
            paused = True
            continue

        cur_fno[0]  = fno
        frame_id   += 1
        frame       = cv2.resize(frame, (disp_w, disp_h))

        # Detection (every DETECT_EVERY frames)
        if frame_id % DETECT_EVERY == 0:
            res  = model(frame, verbose=False,
                         conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
                         imgsz=YOLO_IMGSZ, half=use_cuda)[0]
            dets = [
                ([x1, y1, x2 - x1, y2 - y1], float(box.conf[0]), int(box.cls[0]))
                for box in res.boxes
                if int(box.cls[0]) in VEHICLE_CLASSES
                for x1, y1, x2, y2 in [map(int, box.xyxy[0])]
            ]

        # Tracking
        embeds = compute_embeddings(frame, dets)
        tracks = tracker.update_tracks(dets, embeds=embeds)

        # Per-track rendering
        active = 0
        for track in tracks:
            if not track.is_confirmed():
                continue
            active += 1
            l, t, r, b = map(int, track.to_ltrb())
            tid        = track.track_id
            cid        = getattr(track, "det_class", 2)
            color      = get_color(tid)
            speed      = speed_est.update(int(tid), ((l + r) // 2, (t + b) // 2), ts_ms)
            draw_track(frame, l, t, r, b, tid,
                       VEHICLE_CLASSES.get(cid, "vehicle"), speed, color)

        draw_roi(frame, calibrator.image_points)

        t_now = time.perf_counter()
        fps_buf.append(1.0 / max(t_now - t_prev, FPS_EPSILON))
        t_prev = t_now
        fps    = float(np.mean(fps_buf))

        draw_hud(frame, fps, active, calibrator.is_calibrated(), paused)
        cur_f[0] = frame.copy()

        try:
            app.update_frame(frame)
            app.update_stats(fps, active, calibrator.is_calibrated(),
                             paused, False, fno)
        except Exception:
            stopped = True

    # ── Cleanup ───────────────────────────────────────────────
    reader.stop()

    if load_new:
        try:
            app.destroy()
        except Exception:
            pass
        return "load_new"

    # Video finished — keep window open for replay / load
    if app._alive:
        try:
            app.update_stats(0, 0, calibrator.is_calibrated(),
                             False, True, cur_fno[0])
            app.root.mainloop()
        except Exception:
            pass

    print("[INFO] Done.")
    return "quit"


# =============================================================
#  STARTUP
# =============================================================

def _pick_video() -> str | None:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select video file",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
            ("All files",   "*.*"),
        ],
    )
    root.destroy()
    return path or None


def main() -> None:
    video_path = _pick_video()
    if not video_path:
        print("[INFO] No video selected. Exiting.")
        return

    calibrator = HomographyCalibrator()
    run_calibration_window(video_path, calibrator)

    while True:
        result = run_pipeline(video_path, calibrator)
        if result == "load_new":
            video_path = _pick_video()
            if not video_path:
                break
            calibrator = HomographyCalibrator()
            run_calibration_window(video_path, calibrator)
        else:
            break


# =============================================================
if __name__ == "__main__":
    main()
