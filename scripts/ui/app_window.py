# ui/app_window.py — Main detection window (Tkinter, pure UI)

import cv2
import tkinter as tk
from tkinter import ttk
from typing import Callable

from config import (
    SIDEBAR_W, SEEK_STEP_SHORT, SEEK_STEP_LONG,
    COLOR_SIDEBAR_BG,
    COLOR_BTN_PAUSE, COLOR_BTN_RESET, COLOR_BTN_SCREENSHOT,
    COLOR_BTN_REPLAY, COLOR_BTN_LOAD, COLOR_BTN_QUIT,
)
from ui.ui_helpers import (
    create_header_label, create_section,
    create_var_label, create_static_label,
    create_button, create_inline_button,
    create_seekbar,
)


class AppWindow:
    """
    Single Tkinter window: left = video canvas, right = control sidebar.

    All callbacks (on_reset, on_pause, …) are supplied by the pipeline layer
    so this class contains zero business logic.
    """

    def __init__(self,
                 video_w: int, video_h: int,
                 on_reset:       Callable,
                 on_pause:       Callable,
                 on_quit:        Callable,
                 on_screenshot:  Callable,
                 on_load_video:  Callable,
                 on_seek:        Callable,
                 total_frames:   int,
                 fps_src:        float):

        self.video_w  = video_w
        self.video_h  = video_h
        self._on_quit = on_quit
        self._on_seek = on_seek
        self._total   = total_frames
        self._alive   = True

        # ── Root window ───────────────────────────────────────
        self.root = tk.Tk()
        self.root.title("Vehicle Speed Estimation")
        self.root.protocol("WM_DELETE_WINDOW", self._hard_quit)
        self.root.resizable(False, False)
        self.root.geometry(f"{video_w + SIDEBAR_W}x{video_h}")

        # ── Video canvas (left column) ────────────────────────
        self.canvas = tk.Canvas(self.root,
                                width=video_w, height=video_h,
                                bg="black", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self._photo = None

        # ── Sidebar (right column) ────────────────────────────
        sb = tk.Frame(self.root, bg=COLOR_SIDEBAR_BG, width=SIDEBAR_W)
        sb.grid(row=0, column=1, sticky="nsew")
        sb.grid_propagate(False)
        self.root.columnconfigure(1, minsize=SIDEBAR_W)

        create_header_label(sb, "CONTROLS")
        self._build_stats(sb)
        self._build_seekbar(sb, total_frames, fps_src)
        self._build_buttons(sb, on_pause, on_reset, on_screenshot, on_load_video)

    # ── Section builders ──────────────────────────────────────

    def _build_stats(self, parent: tk.Widget) -> None:
        sf = create_section(parent, "Live Stats")
        self._fv = tk.StringVar(value="FPS: --")
        self._tv = tk.StringVar(value="Tracked: --")
        self._cv = tk.StringVar(value="Homography: --")
        self._sv = tk.StringVar(value="Status: starting")
        for v in (self._fv, self._tv, self._cv, self._sv):
            create_var_label(sf, v)

    def _build_seekbar(self, parent: tk.Widget,
                       total_frames: int, fps_src: float) -> None:
        vf = create_section(parent, "Video Position")

        self._pos_v  = tk.StringVar(value="Frame: 0 / 0")
        create_var_label(vf, self._pos_v)

        self._seek_v = tk.IntVar(value=0)
        create_seekbar(vf, self._seek_v, total_frames,
                       lambda fn: self._on_seek(fn))

        ff = tk.Frame(vf, bg="#1a1a2e")
        ff.pack(fill="x")
        for label, seconds in [
            (f"<<{SEEK_STEP_LONG}s",  -SEEK_STEP_LONG),
            (f"<{SEEK_STEP_SHORT}s",  -SEEK_STEP_SHORT),
            (f"+{SEEK_STEP_SHORT}s>",  SEEK_STEP_SHORT),
            (f"+{SEEK_STEP_LONG}s>>",  SEEK_STEP_LONG),
        ]:
            delta_frames = int(seconds * fps_src)
            create_inline_button(
                ff, label,
                command=lambda d=delta_frames: self._seek_rel(d))

    def _build_buttons(self, parent: tk.Widget,
                       on_pause, on_reset, on_screenshot, on_load_video
                       ) -> None:
        bf = create_section(parent, "Actions")
        for text, cmd, bg in [
            ("Pause / Resume  [SPACE]", on_pause,         COLOR_BTN_PAUSE),
            ("Reset Tracker   [R]",     on_reset,          COLOR_BTN_RESET),
            ("Screenshot      [S]",     on_screenshot,     COLOR_BTN_SCREENSHOT),
            ("Replay Video",            self._replay,      COLOR_BTN_REPLAY),
            ("Load New Video  [L]",     on_load_video,     COLOR_BTN_LOAD),
            ("Quit            [ESC]",   self._hard_quit,   COLOR_BTN_QUIT),
        ]:
            create_button(bf, text, cmd, bg)

    # ── Seek helpers ──────────────────────────────────────────

    def _seek_rel(self, delta_frames: int) -> None:
        target = max(0, min(self._total - 1, self._seek_v.get() + delta_frames))
        self._seek_v.set(target)
        self._on_seek(target)

    def _replay(self) -> None:
        self._seek_v.set(0)
        self._on_seek(0)

    # ── Quit ──────────────────────────────────────────────────

    def _hard_quit(self) -> None:
        """Immediately close the window and signal the pipeline to stop."""
        self._alive = False
        self._on_quit()
        try:
            self.root.destroy()
        except Exception:
            pass

    # ── Frame and stats update (called every pipeline iteration) ──

    def update_frame(self, bgr_frame) -> None:
        import PIL.Image, PIL.ImageTk
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        self._photo = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(rgb))
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

    def update_stats(self, fps: float, tracked: int,
                     calibrated: bool, paused: bool,
                     stopped: bool, frame_no: int) -> None:
        if not self._alive:
            return
        self._fv.set(f"FPS: {fps:.1f}")
        self._tv.set(f"Tracked: {tracked}")
        self._cv.set(f"Homography: {'Calibrated' if calibrated else 'Default'}")
        if stopped:
            status = "Finished"
        elif paused:
            status = "PAUSED"
        else:
            status = "Running"
        self._sv.set(f"Status: {status}")
        self._seek_v.set(frame_no)
        self._pos_v.set(f"Frame: {frame_no} / {self._total}")
        self.root.update()

    def destroy(self) -> None:
        self._alive = False
        try:
            self.root.destroy()
        except Exception:
            pass
