# ui/calibration_window.py — Camera-parameter calibration with live preview
#
# Changes vs previous version:
#  - Auto-preview: trapezoid updates as you type (debounced 400 ms)
#  - Save Config button (separate from Confirm) — lets user preview freely
#    without overwriting saved config until ready
#  - road_slope_deg field added
#  - road_width_m label clarified as "Scale width"
#  - road_depth_m label clarified as "Scale length (depth)"
#  - ROI drawing uses draw_roi_quad from drawing.py → identical to live HUD

import cv2
import json
import os
import tkinter as tk
from typing import TYPE_CHECKING

from config import (
    CAM_HEIGHT_M_DEFAULT, CAM_TILT_DEG_DEFAULT,
    CAM_FOV_H_DEG_DEFAULT, CAM_ROAD_WIDTH_DEFAULT, CAM_ROAD_DEPTH_DEFAULT,
    CAM_SLOPE_DEG_DEFAULT,
    COLOR_SIDEBAR_BG, COLOR_STATUS_WARN,
    COLOR_BTN_DEFAULT, COLOR_BTN_QUIT, COLOR_BTN_RESET,
    COLOR_BTN_CONFIRM_ON,
    VIDEO_CONFIG_SUFFIX,
    SIDEBAR_W, CAL_SCREEN_MARGIN,
    UI_STATUS_FONT_SIZE, UI_STATUS_WRAP_PX, UI_STATUS_BOTTOM_PADY,
)
from ui.ui_helpers import (
    create_header_label, create_section, create_button,
)
from utils.drawing import draw_roi_quad
from utils.screen import get_screen_size, fit_to_screen

if TYPE_CHECKING:
    from core.calibration import HomographyCalibrator


# ── Per-video config helpers ──────────────────────────────────

def video_cal_path(video_path: str) -> str:
    return os.path.splitext(video_path)[0] + VIDEO_CONFIG_SUFFIX

def save_video_config(video_path: str, params: dict) -> None:
    path = video_cal_path(video_path)
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"[Calibration] Config saved -> {path}")

def load_video_config(video_path: str) -> dict | None:
    path = video_cal_path(video_path)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        print(f"[Calibration] Config loaded -> {path}")
        return data
    except Exception:
        return None


# ── Main window ───────────────────────────────────────────────

def run_calibration_window(video_path: str,
                           calibrator: "HomographyCalibrator") -> bool:
    import PIL.Image, PIL.ImageTk

    # ── First frame ───────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    ret, raw_frame = cap.read()
    img_w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not ret:
        print("[ERROR] Cannot read first frame.")
        calibrator.set_default()
        return False

    scr_w, scr_h = get_screen_size()
    cal_sidebar  = SIDEBAR_W + 80
    canvas_w, canvas_h = fit_to_screen(
        raw_frame.shape[1], raw_frame.shape[0],
        scr_w, scr_h - 100,
        margin=CAL_SCREEN_MARGIN + 20,
    )
    display_frame = cv2.resize(raw_frame, (canvas_w, canvas_h))

    saved  = load_video_config(video_path)
    result = [None]

    # ── Root window ───────────────────────────────────────────
    root = tk.Tk()
    root.title("Calibration")
    root.resizable(False, False)
    root.geometry(f"{canvas_w + cal_sidebar}x{canvas_h}")
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, minsize=cal_sidebar)
    root.grid_rowconfigure(0, weight=1)

    # ── Canvas ────────────────────────────────────────────────
    canvas = tk.Canvas(root, width=canvas_w, height=canvas_h,
                       bg="black", highlightthickness=0)
    canvas.grid(row=0, column=0, sticky="nsew")
    _photo = [None]

    def _render(pts_display):
        f = display_frame.copy()
        if pts_display and len(pts_display) == 4:
            draw_roi_quad(f, pts_display)
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        _photo[0] = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(rgb))
        canvas.create_image(0, 0, anchor="nw", image=_photo[0])

    # ── Scrollable sidebar ────────────────────────────────────
    sb_outer = tk.Frame(root, bg=COLOR_SIDEBAR_BG, width=cal_sidebar)
    sb_outer.grid(row=0, column=1, sticky="nsew")
    sb_outer.grid_propagate(False)

    sb_cv = tk.Canvas(sb_outer, bg=COLOR_SIDEBAR_BG, highlightthickness=0)
    sb_cv.pack(side="left", fill="both", expand=True)

    vsb = tk.Scrollbar(sb_outer, orient="vertical", command=sb_cv.yview)
    vsb.pack(side="right", fill="y")
    sb_cv.configure(yscrollcommand=vsb.set)

    sb = tk.Frame(sb_cv, bg=COLOR_SIDEBAR_BG)
    sb_win = sb_cv.create_window((0, 0), window=sb, anchor="nw")

    def _on_sb_configure(e):
        sb_cv.configure(scrollregion=sb_cv.bbox("all"))
        sb_cv.itemconfig(sb_win, width=sb_cv.winfo_width())

    sb.bind("<Configure>", _on_sb_configure)
    sb_cv.bind("<MouseWheel>",
               lambda e: sb_cv.yview_scroll(int(-1*(e.delta/120)), "units"))

    # ── Build sidebar content ─────────────────────────────────
    create_header_label(sb, "CALIBRATION")

    status_var = tk.StringVar(value="")
    tk.Label(sb, textvariable=status_var, anchor="w",
             bg=COLOR_SIDEBAR_BG, fg=COLOR_STATUS_WARN,
             font=("Consolas", UI_STATUS_FONT_SIZE),
             wraplength=UI_STATUS_WRAP_PX, justify="left"
             ).pack(fill="x", padx=8, pady=(4, UI_STATUS_BOTTOM_PADY))

    # ── Parameter rows ────────────────────────────────────────
    def _saved(key, default):
        return saved[key] if saved and key in saved else default
    
    def _field(parent, label, key, default, hint="", min_val=0, max_val=100):
        tk.Label(parent, text=label, anchor="w",
                 bg=COLOR_SIDEBAR_BG, fg="#ccc",
                 font=("Consolas", 8)).pack(fill="x")
    
        var = tk.StringVar(value=str(_saved(key, default)))
    
        # Row container (Entry LEFT, Slider RIGHT)
        row = tk.Frame(parent, bg=COLOR_SIDEBAR_BG)
        row.pack(fill="x", pady=1)
    
        # Entry (left)
        ent = tk.Entry(row, textvariable=var,
                       font=("Consolas", 9),
                       bg="#2c3e50", fg="white",
                       insertbackground="white",
                       width=10)
        ent.pack(side="left", padx=(0, 6))
    
        # Slider (right)
        scale = tk.Scale(row,
                         from_=min_val, to=max_val,
                         orient="horizontal",
                         resolution=0.1,
                         showvalue=0,
                         bg=COLOR_SIDEBAR_BG,
                         fg="white",
                         highlightthickness=0,
                         command=lambda v: var.set(str(v)))
        try:
            scale.set(float(var.get()))
        except:
            scale.set(min_val)
        scale.pack(side="left", fill="x", expand=True)
    
        # Sync Entry -> Slider
        def _on_var_change(*_):
            try:
                val = float(var.get())
                if min_val <= val <= max_val:
                    scale.set(val)
            except:
                pass
            _do_preview()
    
        var.trace_add("write", _on_var_change)
    
        if hint:
            tk.Label(parent, text=hint, anchor="w",
                     bg=COLOR_SIDEBAR_BG, fg="#949494",
                     font=("Consolas", 8)).pack(fill="x")
    
        return var
    
    cf = create_section(sb, "Camera Parameters")
    v_height = _field(cf, "Camera height (m)",
                      "cam_height_m", CAM_HEIGHT_M_DEFAULT,
                      "Height of lens above road surface",
                      0, 20)
    
    v_tilt = _field(cf, "Camera tilt (°, below horizontal)",
                      "cam_tilt_deg", CAM_TILT_DEG_DEFAULT,
                      "0=horizontal  90=straight down  typical CCTV: 20-40°",
                      0, 90)
    
    v_fov = _field(cf, "Horizontal FOV (°)",
                      "fov_h_deg", CAM_FOV_H_DEG_DEFAULT,
                      "Wide-angle: 90-120°  Standard: 50-70°  Tele: 20-40°",
                      0, 120)
    
    v_slope = _field(cf, "Road slope (°, uphill away from cam)",
                      "road_slope_deg", CAM_SLOPE_DEG_DEFAULT,
                      "Flat road = 0°  Uphill = positive  Downhill = neg",
                      -10, 10)
    
    sf = create_section(sb, "Scale")
    v_width = _field(sf, "Scale width — road width (m)",
                      "road_width_m", CAM_ROAD_WIDTH_DEFAULT,
                      "1-lane ~3.5 m  set to visible road width",
                      0, 20)
    
    v_depth = _field(sf, "Scale length — calibration depth (m)",
                      "road_depth_m", CAM_ROAD_DEPTH_DEFAULT,
                      "Distance from camera base to far edge of zone",
                      0, 150)


    # ── Actions ───────────────────────────────────────────────
    af = create_section(sb, "Actions")

    def _get_params() -> dict | None:
        try:
            p = {
                "cam_height_m":  float(v_height.get()),
                "cam_tilt_deg":  float(v_tilt.get()),
                "fov_h_deg":     float(v_fov.get()),
                "road_slope_deg": float(v_slope.get()),
                "road_width_m":  float(v_width.get()),
                "road_depth_m":  float(v_depth.get()),
            }
            # Basic sanity checks
            if p["cam_height_m"] <= 0:
                raise ValueError("Camera height must be > 0")
            if not (0 < p["fov_h_deg"] < 180):
                raise ValueError("FOV must be between 0 and 180°")
            if p["road_width_m"] <= 0 or p["road_depth_m"] <= 0:
                raise ValueError("Scale values must be > 0")
            return p
        except ValueError as e:
            status_var.set(f"⚠ {e}")
            return None

    def _do_preview():
        """Compute and render the calibration trapezoid on the canvas."""
        p = _get_params()
        if p is None:
            return
        calibrator.apply_camera_params(
            canvas_w, canvas_h,
            p["cam_height_m"], p["cam_tilt_deg"], p["fov_h_deg"],
            p["road_width_m"], p["road_depth_m"], p["road_slope_deg"],
        )
        _render(calibrator.image_points)
        status_var.set("Press Save to keep, Confirm to apply & run.")

    def _do_save():
        """Save current params to JSON without closing the window."""
        p = _get_params()
        if p is None:
            return
        save_video_config(video_path, p)
        status_var.set(f"Config saved to:\n"
                       f"{os.path.basename(video_cal_path(video_path))}")

    def _do_confirm():
        p = _get_params()
        if p is None:
            return

        # 1. Luôn tính homography và image_points trên độ phân giải GỐC của video
        calibrator.apply_camera_params(
            img_w_orig, img_h_orig,                    # <--- Quan trọng
            p["cam_height_m"], p["cam_tilt_deg"], p["fov_h_deg"],
            p["road_width_m"], p["road_depth_m"], p["road_slope_deg"]
        )

        # 2. Lưu image_points từ độ phân giải gốc
        calibrator.image_points = [list(pt) for pt in calibrator.image_points]

        # 3. (Tùy chọn) Preview cuối cùng trên canvas để người dùng thấy
        calibrator.apply_camera_params(
            canvas_w, canvas_h,
            p["cam_height_m"], p["cam_tilt_deg"], p["fov_h_deg"],
            p["road_width_m"], p["road_depth_m"], p["road_slope_deg"]
        )

        result[0] = "apply"
        root.quit()     
        
    def _use_default():
        result[0] = "default"
        root.quit()

    create_button(af, "Save Config  [S]",    _do_save,    COLOR_BTN_RESET)
    create_button(af, "Confirm  [ENTER]",    _do_confirm, COLOR_BTN_CONFIRM_ON)
    create_button(af, "Use Default  [ESC]",  _use_default, COLOR_BTN_QUIT)

    # ── Key bindings ──────────────────────────────────────────
    def _on_key(e):
        k = e.keysym.lower()
        if   k == "return": _do_confirm()
        elif k == "s":      _do_save()
        elif k == "escape": _use_default()

    root.bind("<KeyPress>", _on_key)
    root.protocol("WM_DELETE_WINDOW",
                  lambda: (result.__setitem__(0, "default"), root.quit()))

    # ── Initial render ────────────────────────────────────────
    _render(None)
    if saved:
        root.after(100, _do_preview)

    root.mainloop()
    root.destroy()

    # ── Apply final result ────────────────────────────────────
    if result[0] == "apply":
        return True

    # ESC / close → fall back
    if saved:
        p = saved
        calibrator.apply_camera_params(
            img_w_orig, img_h_orig,
            p.get("cam_height_m", CAM_HEIGHT_M_DEFAULT),
            p.get("cam_tilt_deg", CAM_TILT_DEG_DEFAULT),
            p.get("fov_h_deg", CAM_FOV_H_DEG_DEFAULT),
            p.get("road_width_m", CAM_ROAD_WIDTH_DEFAULT),
            p.get("road_depth_m", CAM_ROAD_DEPTH_DEFAULT),
            p.get("road_slope_deg", CAM_SLOPE_DEG_DEFAULT),
        )
        return True

    calibrator.set_default()
    return True
