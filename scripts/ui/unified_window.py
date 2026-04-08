# ui/unified_window.py — Single Tk window with two screens: Calibration & Tracking
#
# Architecture
# ─────────────
# ONE tk.Tk() for the entire app lifetime.
# Two full-size tk.Frame pages stacked on top of each other; tkraise() switches them.
#
#   ┌─────────────────────────────────────────────────────┐
#   │  Screen A: Calibration                              │
#   │  ┌──────────────────────┐ ┌──────────────────────┐  │
#   │  │   canvas (preview)   │ │   sidebar (params)   │  │
#   │  └──────────────────────┘ └──────────────────────┘  │
#   └─────────────────────────────────────────────────────┘
#   ┌─────────────────────────────────────────────────────┐
#   │  Screen B: Tracking                                 │
#   │  ┌──────────────────────┐ ┌──────────────────────┐  │
#   │  │   canvas (video)     │ │   sidebar (controls) │  │
#   │  └──────────────────────┘ └──────────────────────┘  │
#   └─────────────────────────────────────────────────────┘
#
# Pipeline interaction
# ─────────────────────
# The tracking pipeline runs in a background thread so the Tk event-loop
# is never blocked.  The pipeline posts frames / stats to thread-safe
# queues; the Tk loop polls them with root.after().

import cv2
import json
import os
import queue
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Callable

import numpy as np

from config import (
    # camera calibration defaults
    CAM_HEIGHT_M_DEFAULT, CAM_TILT_DEG_DEFAULT, CAM_FOV_H_DEG_DEFAULT,
    CAM_ROAD_WIDTH_DEFAULT, CAM_ROAD_DEPTH_DEFAULT, CAM_SLOPE_DEG_DEFAULT,
    VIDEO_CONFIG_SUFFIX,
    # layout
    TOPBAR_H, SIDEBAR_W, CAL_SCREEN_MARGIN,
    # colours
    COLOR_SIDEBAR_BG, COLOR_HEADER_BG, COLOR_HEADER_FG,
    COLOR_FRAME_FG, COLOR_LABEL_FG, COLOR_ENTRY_BG,
    COLOR_BTN_DEFAULT, COLOR_BTN_DANGER,
    COLOR_STATUS_WARN,
    # fonts / padding
    FONT_TITLE, FONT_NORMAL, FONT_BOLD, FONT_SMALL,
    UI_HEADER_PADY, UI_SECTION_PADX, UI_SECTION_PADY,
    UI_SECTION_PACK_PADX, UI_SECTION_PACK_PADY,
    UI_BTN_PADX, UI_BTN_PADY, UI_BTN_PACK_PADY,
    UI_SEEKBAR_PADY, UI_STATUS_FONT_SIZE, UI_STATUS_WRAP_PX,
    UI_STATUS_BOTTOM_PADY,
    # seek
    SEEK_STEP_SHORT, SEEK_STEP_LONG,
    # ROI drawing
    CAL_LINE_COLOR, CAL_POINT_COLOR, CAL_LINE_THICKNESS,
    ROI_QUAD_ORDER, ROI_LABEL_NAMES,
    ROI_DOT_RADIUS_CAL, ROI_LABEL_FONT_SCALE_CAL,
)
from utils.drawing import draw_roi_quad
from utils.screen import get_screen_size, fit_to_screen

# ── Poll interval for frame/stats queues ─────────────────────
_POLL_MS = 33   # ~30 Hz — reduced to lessen UI contention


# =============================================================
#  WIDGET HELPERS  (private to this module)
# =============================================================

def _header(parent, text):
    lbl = tk.Label(parent, text=text,
                   bg=COLOR_HEADER_BG, fg=COLOR_HEADER_FG,
                   font=FONT_TITLE, pady=UI_HEADER_PADY)
    lbl.pack(fill="x")
    return lbl


def _section(parent, title):
    f = tk.LabelFrame(parent, text=title,
                      bg=COLOR_SIDEBAR_BG, fg=COLOR_FRAME_FG,
                      font=FONT_NORMAL,
                      padx=UI_SECTION_PADX, pady=UI_SECTION_PADY)
    f.pack(fill="x", padx=UI_SECTION_PACK_PADX, pady=UI_SECTION_PACK_PADY)
    return f


def _varlabel(parent, var):
    lbl = tk.Label(parent, textvariable=var, anchor="w",
                   bg=COLOR_SIDEBAR_BG, fg=COLOR_LABEL_FG, font=FONT_SMALL)
    lbl.pack(fill="x")
    return lbl


def _button(parent, text, cmd, bg=COLOR_BTN_DEFAULT):
    btn = tk.Button(parent, text=text, command=cmd,
                    bg=bg, fg="white", font=FONT_BOLD, relief="flat",
                    padx=UI_BTN_PADX, pady=UI_BTN_PADY,
                    cursor="hand2", anchor="w")
    btn.pack(fill="x", pady=UI_BTN_PACK_PADY)
    return btn


def _inline_btn(parent, text, cmd):
    btn = tk.Button(parent, text=text, command=cmd,
                    bg=COLOR_BTN_DEFAULT, fg="white", font=FONT_SMALL,
                    relief="flat", padx=2, cursor="hand2")
    btn.pack(side="left", expand=True, fill="x",
             padx=UI_BTN_PACK_PADY, pady=UI_BTN_PACK_PADY)
    return btn


def _seekbar(parent, var, total, on_seek, owner):
    """
    FIX: Seekbar anti-jitter implementation.
    - _user_dragging flag blocks pipeline updates to the var.
    - Pipeline updates the var only when user is NOT dragging.
    - Release triggers seek, then re-enables updates after short delay.
    """
    bar = ttk.Scale(parent,
                    from_=0,
                    to=max(total - 1, 1),
                    orient="horizontal",
                    variable=var)

    def _on_press(event):
        owner._user_dragging = True

    def _on_move(v):
        # Only do preview if dragging (don't fire on pipeline-driven updates)
        if owner._user_dragging:
            value = int(float(v))
            frame = owner._get_preview_frame(value)
            if frame is not None:
                owner._show_frame(frame)

    def _on_release(event):
        value = int(var.get())
        on_seek(value)
        # Keep dragging=True for a short time so pipeline update doesn't
        # immediately snap the seekbar back
        owner.root.after(300, lambda: setattr(owner, "_user_dragging", False))

    bar.bind("<ButtonPress-1>", _on_press)
    bar.bind("<ButtonRelease-1>", _on_release)
    bar.configure(command=_on_move)
    bar.pack(fill="x", pady=UI_SEEKBAR_PADY)
    return bar


# =============================================================
#  PER-VIDEO CONFIG  (JSON next to the video file)
# =============================================================

def _cal_path(video_path):
    return os.path.splitext(video_path)[0] + VIDEO_CONFIG_SUFFIX


def _save_config(video_path, params):
    with open(_cal_path(video_path), "w") as f:
        json.dump(params, f, indent=2)
    print(f"[Cal] Config saved -> {_cal_path(video_path)}")


def _load_config(video_path):
    p = _cal_path(video_path)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


# =============================================================
#  UNIFIED WINDOW
# =============================================================

class UnifiedWindow:
    """
    Single-Tk-root application window.

    Public API (called from main.py / pipeline)
    ────────────────────────────────────────────
    show_calibration()          — switch to calibration screen
    show_tracking()             — switch to tracking screen
    post_frame(bgr)             — push a video frame (from pipeline thread)
    post_stats(fps,tracked,...) — push live stats
    set_callbacks(...)          — wire pipeline callbacks
    run()                       — enter mainloop (blocks until quit)
    """

    # ── Construction ──────────────────────────────────────────

    def __init__(self,
                 on_quit:     Callable,
                 on_load_new: Callable):

        self._video_path  = None
        self._calibrator  = None
        self._on_quit     = on_quit
        self._on_load_new = on_load_new

        # Pipeline callbacks (set later via set_pipeline_callbacks)
        self._cb_pause      = lambda: None
        self._cb_resume     = lambda: None
        self._cb_reset      = lambda: None
        self._cb_screenshot = lambda: None
        self._cb_seek       = lambda fn: None
        self._total_frames  = 1
        self._fps_src       = 30.0

        # FIX: single _user_dragging flag controls seekbar jitter
        self._user_dragging = False
        self._preview_cap   = None
        self._preview_last_frame = None

        # Screens built once video is loaded
        self._screens_built = False

        # Thread-safe queues
        self._frame_q: queue.Queue = queue.Queue(maxsize=2)
        self._stats_q: queue.Queue = queue.Queue(maxsize=4)

        # Compute display sizes (placeholder until video loaded)
        scr_w, scr_h  = get_screen_size()
        self._disp_w, self._disp_h = fit_to_screen(
            1920, 1080,
            scr_w - SIDEBAR_W, scr_h,
        )
        self._win_w = self._disp_w + SIDEBAR_W
        self._win_h = self._disp_h + TOPBAR_H

        # ── Root ──────────────────────────────────────────────
        self.root = tk.Tk()
        self.root.title("Vehicle Speed Estimation")
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        self.root.resizable(True, True)
        self._fullscreen = False
        self.root.bind("<F11>", lambda event: self._toggle_fullscreen())
        self.root.bind("<Escape>", lambda event: self._exit_fullscreen())

        # Container holds both screens at the same grid cell
        self._build_topbar()
        self._container = tk.Frame(self.root)
        self._container.pack(fill="both", expand=True)
        self._container.grid_rowconfigure(0, weight=1)
        self._container.grid_columnconfigure(0, weight=1)

        self._cal_page   = tk.Frame(self._container)
        self._track_page = tk.Frame(self._container)
        for pg in (self._cal_page, self._track_page):
            pg.grid(row=0, column=0, sticky="nsew")

        self._photo_ref  = None   # PIL photo must be kept alive
        self._alive      = True

        # Debounce id for auto-preview
        self._preview_after_id = None

        # Show welcome / no-video screen
        self._build_welcome_screen()
        self._show_welcome()

    # ── Welcome screen (shown before any video is loaded) ─────

    def _build_welcome_screen(self):
        self._welcome_page = tk.Frame(self._container, bg="#0d1117")
        self._welcome_page.grid(row=0, column=0, sticky="nsew")

        tk.Label(
            self._welcome_page,
            text="Vehicle Speed Estimation",
            bg="#0d1117", fg="#80d2ff",
            font=("Consolas", 18, "bold"),
        ).pack(pady=(80, 10))

        tk.Label(
            self._welcome_page,
            text="Click  Open Video  in the toolbar to begin.",
            bg="#0d1117", fg="#aaa",
            font=("Consolas", 11),
        ).pack(pady=6)

    def _show_welcome(self):
        self._welcome_page.tkraise()
        self.root.geometry(f"700x400")
        self.root.title("Vehicle Speed Estimation — No Video")

    # ── Two-phase init (call after pipeline knows video size) ─

    def init_screens(self, video_w: int, video_h: int,
                     total_frames: int, fps_src: float) -> None:
        """
        Called once the pipeline has opened the video and knows its size.
        Builds both screens and resizes the window to fit.
        """
        scr_w, scr_h = get_screen_size()
        self._disp_w, self._disp_h = fit_to_screen(
            video_w, video_h, scr_w - SIDEBAR_W, scr_h)
        self._win_w  = self._disp_w + SIDEBAR_W
        self._win_h  = self._disp_h
        self._total_frames = total_frames
        self._fps_src      = fps_src

        win_w = max(self._win_w, self._disp_w + SIDEBAR_W)
        win_h = self._win_h
        self.root.update_idletasks()
        x = max(0, (self.root.winfo_screenwidth()  - win_w) // 2)
        y = max(0, (self.root.winfo_screenheight() - win_h) // 2)
        self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")

        # Load first frame for calibration preview
        self._load_cal_first_frame()

        if not self._screens_built:
            self._build_calibration_screen()
            self._build_tracking_screen()
            self._screens_built = True
        else:
            # Rebuild screens for new video
            for widget in self._cal_page.winfo_children():
                widget.destroy()
            for widget in self._track_page.winfo_children():
                widget.destroy()
            self._build_calibration_screen()
            self._build_tracking_screen()

        self.show_calibration()

        if self._preview_cap:
            self._preview_cap.release()
        self._preview_cap = cv2.VideoCapture(self._video_path)

    def _get_preview_frame(self, frame_no: int):
        if not self._preview_cap:
            return None
        self._preview_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = self._preview_cap.read()
        if not ret:
            return None
        frame = cv2.resize(frame, (self._disp_w, self._disp_h))
        return frame

    def _build_base_screen(self, parent):
        parent.configure(bg="black")

        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, minsize=SIDEBAR_W)
        parent.grid_rowconfigure(0, weight=1)

        canvas = tk.Canvas(parent,
                           width=self._disp_w,
                           height=self._disp_h,
                           bg="black",
                           highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")

        sidebar = self._build_scrollable_sidebar(parent)
        return canvas, sidebar

    def _build_scrollable_sidebar(self, parent):
        outer = tk.Frame(parent, bg=COLOR_SIDEBAR_BG, width=SIDEBAR_W)
        outer.grid(row=0, column=1, sticky="nsew")
        outer.grid_propagate(False)

        SCROLLBAR_W = 15
        canvas = tk.Canvas(
            outer,
            bg=COLOR_SIDEBAR_BG,
            highlightthickness=0,
            width=SIDEBAR_W - SCROLLBAR_W
        )
        canvas.pack(side="left", fill="both")

        scrollbar = tk.Scrollbar(outer, orient="vertical",
                                 command=canvas.yview)
        scrollbar.pack(side="right", fill="y")

        canvas.configure(yscrollcommand=scrollbar.set)

        inner = tk.Frame(canvas, bg=COLOR_SIDEBAR_BG)
        window = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(window, width=canvas.winfo_width())

        inner.bind("<Configure>", _on_configure)

        canvas.bind("<MouseWheel>",
                    lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        return inner

    def _build_topbar(self):
        bar = tk.Frame(self.root, bg="#1f2a36", height=TOPBAR_H)
        bar.pack(side="top", fill="x")

        def btn(text, cmd, bg=COLOR_BTN_DEFAULT):
            return tk.Button(
                bar,
                text=text,
                command=cmd,
                bg=bg,
                fg="white",
                relief="flat",
                padx=12,
                pady=6,
                cursor="hand2"
            )

        btn("Open Video", self._open_video, COLOR_BTN_DEFAULT).pack(side="left", padx=6, pady=4)
        btn("Calibration", self._go_calibration, COLOR_BTN_DEFAULT).pack(side="left", padx=6)
        btn("Tracking", self._go_tracking, COLOR_BTN_DEFAULT).pack(side="left", padx=6)
        btn("Fullscreen", self._toggle_fullscreen, COLOR_BTN_DEFAULT).pack(side="left", padx=6)
        btn("Quit", self._quit, COLOR_BTN_DANGER).pack(side="left", padx=6)

        tk.Label(
            bar,
            text="Bùi Duy Phong - 19110131   |   Huỳnh Minh Tài - 22110068   |   Lê Minh Ngọc - 22110056",
            bg="#1f2a36", fg="white",
            font=FONT_SMALL,
        ).pack(side="right", padx=12)

    def _toggle_fullscreen(self):
        self._fullscreen = not self._fullscreen
        self.root.attributes("-fullscreen", self._fullscreen)

    def _exit_fullscreen(self):
        if self._fullscreen:
            self._fullscreen = False
            self.root.attributes("-fullscreen", False)

    # ─────────────────────────────────────────────────────────
    #  SCREEN: CALIBRATION
    # ─────────────────────────────────────────────────────────

    def _load_cal_first_frame(self):
        cap = cv2.VideoCapture(self._video_path)
        ret, raw = cap.read()
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        self._cal_orig_w = orig_w or (raw.shape[1] if ret else 1920)
        self._cal_orig_h = orig_h or (raw.shape[0] if ret else 1080)
        if ret:
            self._cal_first_frame = cv2.resize(raw, (self._disp_w, self._disp_h))
        else:
            self._cal_first_frame = np.zeros((self._disp_h, self._disp_w, 3), np.uint8)

    def _build_calibration_screen(self):
        pg = self._cal_page
        self._cal_canvas, sb_outer = self._build_base_screen(self._cal_page)
        sb = self._build_scrollable_sidebar(pg)

        _header(sb, "CALIBRATION")

        self._cal_status = tk.StringVar(value="")
        tk.Label(sb, textvariable=self._cal_status, anchor="nw",
                 bg=COLOR_SIDEBAR_BG, fg=COLOR_STATUS_WARN,
                 font=("Consolas", UI_STATUS_FONT_SIZE),
                 wraplength=SIDEBAR_W - 20, justify="left", height=3
                 ).pack(fill="x", padx=8, pady=(4, UI_STATUS_BOTTOM_PADY))

        # Parameter fields
        saved = _load_config(self._video_path)

        def _saved(key, default):
            return saved[key] if saved and key in saved else default

        def _field_slider(parent, label, key, default, hint="", lo=0, hi=100):
            tk.Label(parent, text=label, anchor="w",
                     bg=COLOR_SIDEBAR_BG, fg="#ccc",
                     font=("Consolas", 8)).pack(fill="x")
            var = tk.StringVar(value=str(_saved(key, default)))
            row = tk.Frame(parent, bg=COLOR_SIDEBAR_BG)
            row.pack(fill="x", pady=1)
            ent = tk.Entry(row, textvariable=var, font=("Consolas", 9),
                           bg="#2c3e50", fg="white",
                           insertbackground="white", width=8)
            ent.pack(side="left", padx=(0, 4))
            sc = tk.Scale(row, from_=lo, to=hi, orient="horizontal",
                          resolution=0.1, showvalue=0,
                          bg=COLOR_SIDEBAR_BG, fg="white",
                          highlightthickness=0,
                          command=lambda v: var.set(str(round(float(v), 2))))
            try:
                sc.set(float(var.get()))
            except Exception:
                sc.set(lo)
            sc.pack(side="left", fill="x", expand=True)

            def _sync(*_):
                try:
                    val = float(var.get())
                    if lo <= val <= hi:
                        sc.set(val)
                except Exception:
                    pass
                self._do_preview()

            var.trace_add("write", _sync)
            if hint:
                tk.Label(parent, text=hint, anchor="w",
                         bg=COLOR_SIDEBAR_BG, fg="#555",
                         font=("Consolas", 7)).pack(fill="x")
            return var

        cf = _section(sb, "Camera Parameters")
        self._v_height = _field_slider(cf, "Camera height (m)",
                                "cam_height_m", CAM_HEIGHT_M_DEFAULT,
                                "Height of lens above road surface", 0, 20)
        self._v_tilt = _field_slider(cf, "Camera tilt (°, below horizontal)",
                                "cam_tilt_deg", CAM_TILT_DEG_DEFAULT,
                                "0=horizontal  90=straight down  CCTV: 20-40°", 0, 90)
        self._v_fov = _field_slider(cf, "Horizontal FOV (°)",
                                "fov_h_deg", CAM_FOV_H_DEG_DEFAULT,
                                "Wide: 90-120°  Standard: 50-70°  Tele: 20-40°", 1, 180)
        self._v_slope = _field_slider(cf, "Road slope (°, uphill away)",
                                "road_slope_deg", CAM_SLOPE_DEG_DEFAULT,
                                "Flat=0°  Uphill=positive  Downhill=negative", -15, 15)

        sf = _section(sb, "Scale")
        self._v_width = _field_slider(sf, "Scale width — road width (m)",
                                "road_width_m", CAM_ROAD_WIDTH_DEFAULT,
                                "1-lane ~3.5 m  2-lane ~7 m", 0.5, 30)
        self._v_depth = _field_slider(sf, "Scale length — depth (m)",
                                "road_depth_m", CAM_ROAD_DEPTH_DEFAULT,
                                "Distance from camera to far edge of zone", 5, 200)

        # Action buttons
        af = _section(sb, "Actions")
        _button(af, "Save Config", self._cal_save, COLOR_BTN_DEFAULT)
        _button(af, "Reset Config", self._cal_reset_to_saved, COLOR_BTN_DANGER)

        # Auto-preview on first show
        if saved:
            self.root.after(150, self._do_preview)
        else:
            self._render_cal(None)

    def _schedule_preview(self):
        """Debounced preview: only fires 400 ms after last change."""
        if self._preview_after_id:
            self.root.after_cancel(self._preview_after_id)
        self._preview_after_id = self.root.after(400, self._do_preview)

    def _get_params(self):
        try:
            p = {
                "cam_height_m": float(self._v_height.get()),
                "cam_tilt_deg": float(self._v_tilt.get()),
                "fov_h_deg": float(self._v_fov.get()),
                "road_slope_deg": float(self._v_slope.get()),
                "road_width_m": float(self._v_width.get()),
                "road_depth_m": float(self._v_depth.get()),
            }
            if p["cam_height_m"] <= 0:
                raise ValueError("Camera height must be > 0")
            if not (0 < p["fov_h_deg"] < 180):
                raise ValueError("FOV must be between 0 and 180°")
            if p["road_width_m"] <= 0 or p["road_depth_m"] <= 0:
                raise ValueError("Scale values must be > 0")
            return p
        except ValueError as e:
            self._cal_status.set(f"⚠ {e}")
            return None

    def _do_preview(self):
        if not self._video_path:
            return
        p = self._get_params()
        if p is None:
            return
        self._calibrator.apply_camera_params(
            self._disp_w, self._disp_h,
            p["cam_height_m"], p["cam_tilt_deg"], p["fov_h_deg"],
            p["road_width_m"], p["road_depth_m"], p["road_slope_deg"],
        )
        self._render_cal(self._calibrator.image_points)
        self._cal_status.set("Preview updated.\nSave = write JSON | Confirm = start tracking")

    def _render_cal(self, pts):
        import PIL.Image, PIL.ImageTk
        f = self._cal_first_frame.copy()
        if pts and len(pts) == 4:
            _draw_roi_enhanced(f, pts)
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        self._cal_photo = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(rgb))
        self._cal_canvas.delete("all")
        canvas_w = self._cal_canvas.winfo_width() or self._disp_w
        canvas_h = self._cal_canvas.winfo_height() or self._disp_h
        self._cal_canvas.create_image(canvas_w // 2,
                                      canvas_h // 2,
                                      anchor="center",
                                      image=self._cal_photo)

    def _cal_save(self):
        p = self._get_params()
        if p is None:
            return
        _save_config(self._video_path, p)
        self._cal_status.set(f"Saved → {os.path.basename(_cal_path(self._video_path))}")

    def _cal_reset_to_saved(self):
        saved = _load_config(self._video_path)
        if not saved:
            self._cal_status.set("⚠ No saved config found")
            return

        self._v_height.set(str(saved.get("cam_height_m", CAM_HEIGHT_M_DEFAULT)))
        self._v_tilt.set(str(saved.get("cam_tilt_deg", CAM_TILT_DEG_DEFAULT)))
        self._v_fov.set(str(saved.get("fov_h_deg", CAM_FOV_H_DEG_DEFAULT)))
        self._v_slope.set(str(saved.get("road_slope_deg", CAM_SLOPE_DEG_DEFAULT)))
        self._v_width.set(str(saved.get("road_width_m", CAM_ROAD_WIDTH_DEFAULT)))
        self._v_depth.set(str(saved.get("road_depth_m", CAM_ROAD_DEPTH_DEFAULT)))

        self._do_preview()
        self._cal_status.set("Restored saved config")

    def _cal_confirm(self):
        p = self._get_params()
        if p is None:
            return
        # Apply homography at original video resolution
        orig_w = getattr(self, "_cal_orig_w", self._disp_w)
        orig_h = getattr(self, "_cal_orig_h", self._disp_h)
        self._calibrator.apply_camera_params(
            orig_w, orig_h,
            p["cam_height_m"], p["cam_tilt_deg"], p["fov_h_deg"],
            p["road_width_m"], p["road_depth_m"], p["road_slope_deg"],
        )
        # Keep display-space image_points for ROI overlay in tracking screen
        self._calibrator.apply_camera_params(
            self._disp_w, self._disp_h,
            p["cam_height_m"], p["cam_tilt_deg"], p["fov_h_deg"],
            p["road_width_m"], p["road_depth_m"], p["road_slope_deg"],
        )
        disp_pts = self._calibrator.image_points
        self._calibrator.apply_camera_params(
            orig_w, orig_h,
            p["cam_height_m"], p["cam_tilt_deg"], p["fov_h_deg"],
            p["road_width_m"], p["road_depth_m"], p["road_slope_deg"],
        )
        self._calibrator.image_points = disp_pts
        self.show_tracking()

    # ─────────────────────────────────────────────────────────
    #  SCREEN: TRACKING
    # ─────────────────────────────────────────────────────────

    def _build_tracking_screen(self):
        pg = self._track_page
        self._vid_canvas, sb = self._build_base_screen(self._track_page)
        sb = self._build_scrollable_sidebar(pg)

        _header(sb, "CONTROLS")

        # Stats
        sf = _section(sb, "Live Stats")
        self._t_fps = tk.StringVar(value="FPS: --")
        self._t_tracked = tk.StringVar(value="Tracked: --")
        self._t_cal = tk.StringVar(value="Homography: --")
        self._t_status  = tk.StringVar(value="Status: starting")
        for v in (self._t_fps, self._t_tracked, self._t_cal, self._t_status):
            _varlabel(sf, v)

        # Video position
        vf = _section(sb, "Video Position")
        self._t_pos  = tk.StringVar(value="Frame: 0 / 0")
        _varlabel(vf, self._t_pos)
        self._seek_v = tk.IntVar(value=0)
        _seekbar(vf, self._seek_v, self._total_frames,
                 lambda fn: self._cb_seek(fn), self)
        ff = tk.Frame(vf, bg=COLOR_SIDEBAR_BG)
        ff.pack(fill="x")
        for label, sec in [
            (f"<<{SEEK_STEP_LONG}s", -SEEK_STEP_LONG),
            (f"<{SEEK_STEP_SHORT}s", -SEEK_STEP_SHORT),
            (f"+{SEEK_STEP_SHORT}s>", SEEK_STEP_SHORT),
            (f"+{SEEK_STEP_LONG}s>>", SEEK_STEP_LONG),
        ]:
            df = int(sec * self._fps_src)
            _inline_btn(ff, label, lambda d=df: self._seek_rel(d))

        # Actions
        af = _section(sb, "Actions")
        _button(af, "Pause / Resume  [SPACE]", lambda: self._cb_pause(), COLOR_BTN_DEFAULT)
        _button(af, "Replay Video    [R]", self._replay, COLOR_BTN_DANGER)
        _button(af, "Screenshot      [S]", self._do_screenshot, COLOR_BTN_DEFAULT)

        # Screenshot notification label
        self._notice_v = tk.StringVar(value="")
        tk.Label(af, textvariable=self._notice_v, anchor="w",
                 bg=COLOR_SIDEBAR_BG, fg="#7dffb3",
                 font=("Consolas", 8), wraplength=SIDEBAR_W - 20,
                 justify="left").pack(fill="x", pady=(2, 0))

        # Expand spacer — keeps content at top in tall windows
        tk.Frame(sb, bg=COLOR_SIDEBAR_BG).pack(fill="both", expand=True)

        # Keyboard shortcuts for tracking screen
        pg.bind_all("<space>", lambda e: self._cb_pause())
        pg.bind_all("<r>", lambda e: self._replay())
        pg.bind_all("<s>", lambda e: self._do_screenshot())

    # ── Video frame / stats pushed from pipeline thread ───────

    def post_frame(self, bgr_frame: np.ndarray) -> None:
        """Non-blocking; drops old frame if queue is full."""
        try:
            self._frame_q.put_nowait(bgr_frame)
        except queue.Full:
            try:
                self._frame_q.get_nowait()
            except queue.Empty:
                pass
            self._frame_q.put_nowait(bgr_frame)

    def post_stats(self, fps: float, tracked: int,
                   calibrated: bool, paused: bool,
                   stopped: bool, frame_no: int) -> None:
        try:
            self._stats_q.put_nowait((fps, tracked, calibrated,
                                      paused, stopped, frame_no))
        except queue.Full:
            try:
                self._stats_q.get_nowait()
            except queue.Empty:
                pass
            self._stats_q.put_nowait((fps, tracked, calibrated,
                                      paused, stopped, frame_no))

    def _poll(self):
        """Called by Tk every _POLL_MS ms to drain queues."""
        # FIX: only render frame when on tracking screen
        if self._track_page.winfo_ismapped():
            try:
                frame = self._frame_q.get_nowait()
                if not self._user_dragging:
                    self._show_frame(frame)
            except queue.Empty:
                pass
        else:
            # Drain queue to avoid memory buildup when on calibration screen
            try:
                while True:
                    self._frame_q.get_nowait()
            except queue.Empty:
                pass

        try:
            fps, tracked, calibrated, paused, stopped, fno = self._stats_q.get_nowait()
            if hasattr(self, "_t_fps"):
                self._t_fps.set(f"FPS: {fps:.1f}")
                self._t_tracked.set(f"Tracked: {tracked}")
                self._t_cal.set(f"Homography: {'Calibrated' if calibrated else 'Default'}")
                if stopped:
                    self._t_status.set("Status: Finished")
                elif paused:
                    self._t_status.set("Status: PAUSED")
                else:
                    self._t_status.set("Status: Running")

                # FIX: Only update seekbar var when not dragging
                if not self._user_dragging:
                    self._seek_v.set(fno)
                self._t_pos.set(f"Frame: {fno} / {self._total_frames}")
        except queue.Empty:
            pass

        if self._alive:
            self.root.after(_POLL_MS, self._poll)

    def _show_frame(self, bgr):
        import PIL.Image, PIL.ImageTk
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._photo_ref = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(rgb))
        self._vid_canvas.delete("all")
        canvas_w = self._vid_canvas.winfo_width() or self._disp_w
        canvas_h = self._vid_canvas.winfo_height() or self._disp_h
        self._vid_canvas.create_image(canvas_w // 2,
                                      canvas_h // 2,
                                      anchor="center",
                                      image=self._photo_ref)

    # ── Screen switching ──────────────────────────────────────

    def _go_calibration(self):
        if not self._video_path:
            return
        # Pause pipeline when going to calibration
        self._cb_pause()
        self.show_calibration()

    def _go_tracking(self):
        if not self._video_path:
            return
        self.show_tracking()

    def show_calibration(self):
        self._cal_page.tkraise()
        self.root.title("Vehicle Speed Estimation — Calibration")

    def show_tracking(self):
        self._track_page.tkraise()
        self.root.title("Vehicle Speed Estimation — Tracking")
        # FIX: resume pipeline when switching to tracking
        self._cb_resume()

    # ── Helpers ───────────────────────────────────────────────

    def _seek_rel(self, delta):
        target = max(0, min(self._total_frames - 1,
                            self._seek_v.get() + delta))
        self._user_dragging = True
        self._seek_v.set(target)
        self._cb_seek(target)
        self.root.after(300, lambda: setattr(self, "_user_dragging", False))

    def _replay(self):
        """FIX: properly reset pipeline so video replays from start."""
        self._user_dragging = True
        self._seek_v.set(0)
        self._cb_reset()  # this clears pause and resets to frame 0
        self.root.after(400, lambda: setattr(self, "_user_dragging", False))

    def _do_screenshot(self):
        path = self._cb_screenshot()
        if path:
            fname = os.path.basename(path)
            self._show_notice(f"Saved: {fname}", ms=3000)

    def _show_notice(self, msg: str, ms: int = 3000):
        if not self._alive:
            return
        if hasattr(self, "_notice_v"):
            self._notice_v.set(msg)
            self.root.after(ms, lambda: self._notice_v.set("") if self._alive else None)

    def _open_video(self):
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                       ("All files", "*.*")],
        )
        if path:
            self._video_path = path
            self._on_load_new(path)

    def _quit(self):
        if self._preview_cap:
            self._preview_cap.release()
        self._alive = False
        self._on_quit()
        try:
            self.root.destroy()
        except Exception:
            pass

    # ── Pipeline callback wiring ──────────────────────────────

    def set_pipeline_callbacks(self,
                               on_pause:      Callable,
                               on_resume:     Callable,
                               on_reset:      Callable,
                               on_screenshot: Callable,
                               on_seek:       Callable) -> None:
        self._cb_pause      = on_pause
        self._cb_resume     = on_resume
        self._cb_reset      = on_reset
        self._cb_screenshot = on_screenshot
        self._cb_seek       = on_seek

    # ── Entry point ───────────────────────────────────────────

    def run(self):
        self.root.after(_POLL_MS, self._poll)
        self.root.mainloop()

    @property
    def is_alive(self):
        return self._alive


# =============================================================
#  ENHANCED ROI DRAWING  (gradient + depth cues)
# =============================================================

def _draw_roi_enhanced(frame: np.ndarray, pts: list) -> None:
    """
    Draw the calibration ROI with visual depth/perspective cues:
    - Gradient fill: near=warm (amber), far=cool (cyan), semi-transparent
    - Corner dots: larger near, smaller far
    - Edge lines: thicker near, thinner far
    - Labels with background chips
    - Perspective grid lines inside the zone
    """
    if not pts or len(pts) < 4:
        return

    tl, tr, bl, br = [tuple(map(int, p)) for p in pts]

    # ── Gradient fill overlay ──────────────────────────────────
    overlay = frame.copy()

    # Build a filled polygon mask
    poly = np.array([tl, tr, br, bl], dtype=np.int32)

    # Near zone (bottom) = warm amber, far zone (top) = cool cyan
    # We draw horizontal gradient scanlines inside the polygon
    y_min = min(tl[1], tr[1])
    y_max = max(bl[1], br[1])
    h_range = max(y_max - y_min, 1)

    # Colour stops: far=cyan (0,220,255), near=amber (255,180,0)
    far_color  = np.array([0,   220, 255], dtype=np.float32)   # BGR cyan
    near_color = np.array([255, 160,   0], dtype=np.float32)   # BGR amber

    for y in range(y_min, y_max + 1):
        t_val = (y - y_min) / h_range              # 0=far, 1=near
        color = (far_color * (1 - t_val) + near_color * t_val).astype(int)

        # Interpolate left and right edge at this y
        # Left edge: tl→bl,  Right edge: tr→br
        def interp_x(p1, p2, y_val):
            if p2[1] == p1[1]:
                return p1[0]
            frac = (y_val - p1[1]) / (p2[1] - p1[1])
            return int(p1[0] + frac * (p2[0] - p1[0]))

        x_left  = interp_x(tl, bl, y)
        x_right = interp_x(tr, br, y)
        if x_left > x_right:
            x_left, x_right = x_right, x_left
        cv2.line(overlay, (x_left, y), (x_right, y),
                 (int(color[0]), int(color[1]), int(color[2])), 1)

    cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)

    # ── Perspective grid lines inside zone ────────────────────
    # 2 vertical + 2 horizontal lines fading with depth
    grid_alpha = frame.copy()
    for frac in (0.33, 0.67):
        # Vertical: interpolate between (tl→tr) and (bl→br)
        top_x    = int(tl[0] + frac * (tr[0] - tl[0]))
        top_y    = int(tl[1] + frac * (tr[1] - tl[1]))
        bot_x    = int(bl[0] + frac * (br[0] - bl[0]))
        bot_y    = int(bl[1] + frac * (br[1] - bl[1]))
        cv2.line(grid_alpha, (top_x, top_y), (bot_x, bot_y), (180, 255, 200), 1, cv2.LINE_AA)

        # Horizontal: interpolate between (tl→bl) and (tr→br)
        left_x  = int(tl[0] + frac * (bl[0] - tl[0]))
        left_y  = int(tl[1] + frac * (bl[1] - tl[1]))
        right_x = int(tr[0] + frac * (br[0] - tr[0]))
        right_y = int(tr[1] + frac * (br[1] - tr[1]))
        cv2.line(grid_alpha, (left_x, left_y), (right_x, right_y), (180, 255, 200), 1, cv2.LINE_AA)

    cv2.addWeighted(grid_alpha, 0.45, frame, 0.55, 0, frame)

    # ── Edge lines: near=thick amber, far=thin cyan ───────────
    # Far edge (top): tl→tr — thin cyan
    cv2.line(frame, tl, tr, (0, 220, 255), 1, cv2.LINE_AA)
    # Near edge (bottom): bl→br — thick amber
    cv2.line(frame, bl, br, (255, 160, 0), 3, cv2.LINE_AA)
    # Left side: tl→bl — gradient (just use mid color)
    cv2.line(frame, tl, bl, (100, 200, 180), 2, cv2.LINE_AA)
    # Right side: tr→br
    cv2.line(frame, tr, br, (100, 200, 180), 2, cv2.LINE_AA)

    # ── Corner dots: far=small cool, near=large warm ──────────
    corner_cfg = [
        (tl, (0,   220, 255), 5,  "TL", "FAR"),
        (tr, (0,   220, 255), 5,  "TR", "FAR"),
        (bl, (255, 160,   0), 9,  "BL", "NEAR"),
        (br, (255, 160,   0), 9,  "BR", "NEAR"),
    ]
    for pt, col, radius, label, hint in corner_cfg:
        # White ring
        cv2.circle(frame, pt, radius + 2, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(frame, pt, radius,     col,             -1, cv2.LINE_AA)

        # Label chip background
        lx = pt[0] + 10
        ly = pt[1] - 10
        label_text = f"{label} ({hint})"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(frame, (lx - 2, ly - th - 3), (lx + tw + 2, ly + 3),
                      (20, 20, 20), -1)
        cv2.putText(frame, label_text, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)