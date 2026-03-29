import cv2
import os

import tkinter as tk
from utils.screen import get_screen_size, fit_to_screen
from config import (ROAD_WIDTH_M, ROAD_LENGTH_M,
                    DEFAULT_CAM_HEIGHT_M, DEFAULT_CAM_TILT_DEG,
                    DEFAULT_CAM_FOV_H_DEG, DEFAULT_ROAD_SLOPE_DEG)
from utils.ui_helpers import SIDEBAR_W, COLOR_SIDEBAR_BG, create_label_title, create_label_frame, create_label, create_entry, create_button

NEEDED_POINT = 4

POINT_RADIUS = 8
POINT_COLOR = (0, 255, 100)
LINE_COLOR = (0, 210, 255)
STATUS_TEXT_COLOR = (100, 255, 180)

TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE_MAIN = 0.8
TEXT_SCALE_STATUS = 0.55

TEXT_THICKNESS_MAIN = 2
TEXT_THICKNESS_STATUS = 1

TEXT_OFFSET_X = 10
TEXT_OFFSET_Y = -10

STATUS_POS_X = 10
STATUS_MARGIN_BOTTOM = 12

LINE_THICKNESS = 2

COLOR_STATUS_WARN = "#f4a261"
COLOR_CONFIRM_ENABLED = "#1d9e75"
COLOR_CONFIRM_DISABLED = "#444"


def run_calibration_window(video_path, calibrator):
    """
    Opens a single Tk window:
      Left  : OpenCV frame embedded via PIL (click to place points)
      Right : sidebar with buttons and dimension inputs

    Returns True if calibration was applied (new or loaded).
    Returns False if user chose to skip (use existing/default).
    """
    import PIL.Image, PIL.ImageTk

    # Read and scale first frame
    cap = cv2.VideoCapture(video_path)
    ret, raw_frame = cap.read()
    cap.release()
    if not ret:
        print("[ERROR] Cannot read first frame.")
        return False

    scr_w, scr_h = get_screen_size()
    cw, ch = fit_to_screen(raw_frame.shape[1], raw_frame.shape[0],
                            scr_w - SIDEBAR_W, scr_h, margin=80)
    display_frame = cv2.resize(raw_frame, (cw, ch))

    # Check for existing calibration file
    cal_path = os.path.join(os.path.dirname(video_path), "calibration.npz")
    has_saved = os.path.exists(cal_path)

    clicked = []
    result   = [None]   # "new", "auto", "load", "default", "cancel"
    img_size = [None]   # [frame_w, frame_h] for apply_camera_params
    img_size[0] = (cw, ch)

    # ── Build Tk window ───────────────────────────────────────
    root = tk.Tk()
    root.title("Calibration")
    root.resizable(False, False)
    root.geometry(f"{cw + SIDEBAR_W}x{ch}")

# =============================================================================
# Canvas
# =============================================================================
    canvas = tk.Canvas(root, width=cw, height=ch,
                       bg="black", highlightthickness=0, cursor="crosshair")
    canvas.grid(row=0, column=0)
    photo_ref = [None]

    def render_frame():
        f = display_frame.copy()
        # Draw existing points
        for i, pt in enumerate(clicked):
            cv2.circle(f, (pt[0],pt[1]), POINT_RADIUS, POINT_COLOR, -1)
            cv2.putText(f, str(i+1), (pt[0]+TEXT_OFFSET_X, pt[1]+TEXT_OFFSET_Y),
                        TEXT_FONT, TEXT_SCALE_MAIN, POINT_COLOR, TEXT_THICKNESS_MAIN)
            if i > 0:
                cv2.line(f, tuple(clicked[i-1]), tuple(pt), LINE_COLOR, LINE_THICKNESS)
                
        if len(clicked) == NEEDED_POINT:
            cv2.line(f, tuple(clicked[3]), tuple(clicked[0]), LINE_COLOR, LINE_THICKNESS)
        # Status text top-left
        status = f"Points: {len(clicked)} / {NEEDED_POINT}"
        cv2.putText(f, status, (STATUS_POS_X, ch-STATUS_MARGIN_BOTTOM),
                    TEXT_FONT, TEXT_SCALE_STATUS, STATUS_TEXT_COLOR, TEXT_THICKNESS_STATUS, cv2.LINE_AA)
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        photo_ref[0] = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(rgb))
        canvas.create_image(0, 0, anchor="nw", image=photo_ref[0])
        _update_sidebar_state()

    def on_canvas_click(event):
        if len(clicked) < NEEDED_POINT:
            clicked.append([event.x, event.y])
            render_frame()

    def on_canvas_rclick(event):
        if clicked:
            clicked.pop()
            render_frame()

    canvas.bind("<Button-1>", on_canvas_click)
    canvas.bind("<Button-3>", on_canvas_rclick)

# =============================================================================
# Sidebar
# =============================================================================
    sb = tk.Frame(root, bg=COLOR_SIDEBAR_BG, width=SIDEBAR_W)
    sb.grid(row=0, column=1, sticky="nsew")
    sb.grid_propagate(False)

    create_label_title(sb, "CALIBRATION")


# =============================================================================
# Instructions
# =============================================================================
    inf = create_label_frame(sb, "Instructions")
    create_label(inf, (
        f"Click {NEEDED_POINT} road points:\n"
        "  1 = Top-Left\n"
        "  2 = Top-Right\n"
        "  3 = Bot-Left\n"
        "  4 = Bot-Right\n\n"
        "Pick a flat rectangle\n"
        "with known real size.\n"
        "e.g. lane lines,\n"
        "crosswalk, parking bay."
    ))

# =============================================================================
# Dimension inputs
# =============================================================================
    dimf = create_label_frame(sb, "Real-world size")
    
    # Width
    create_label(dimf, "Width (top/bot edge) m:")
    width_var = tk.DoubleVar(value=ROAD_WIDTH_M)
    create_entry(dimf, width_var)
    
    # Length
    create_label(dimf, "Length (left/right edge) m:")
    length_var = tk.DoubleVar(value=ROAD_LENGTH_M)
    create_entry(dimf, length_var)

# =============================================================================
# Buttons
# =============================================================================
    def do_confirm():
        if mode_var.get() == "auto":
            try:
                h  = float(cam_h_var.get())
                td = float(cam_tilt_var.get())
                fv = float(cam_fov_var.get())
                sl = float(road_slope_var.get())
            except Exception:
                status_var.set("Invalid camera parameter value.")
                return
            if h <= 0:
                status_var.set("Camera height must be > 0.")
                return
            if not (0 < fv < 180):
                status_var.set("FOV must be 0–180 deg.")
                return
            if not (0 < td < 90):
                status_var.set("Tilt must be 0–90 deg.")
                return
            result[0] = "auto"
            root.quit()
        else:
            if len(clicked) != NEEDED_POINT:
                status_var.set(f"Need exactly {NEEDED_POINT} points!")
                return
            try:
                w = float(width_var.get())
                l = float(length_var.get())
            except Exception:
                status_var.set("Invalid dimensions.")
                return
            if w <= 0 or l <= 0:
                status_var.set("Dimensions must be > 0.")
                return
            result[0] = "new"
            root.quit()
        
    def undo_last_point():
        if clicked:
            clicked.pop()
        render_frame()
        
    def reset_all_points():
        clicked.clear()
        render_frame()
        
    def use_default():
        result[0] = "default"
        root.quit()
        

# ── Calibration mode toggle ────────────────────────────────────
    mode_var = tk.StringVar(value="manual")

    def _refresh_mode(*_):
        if mode_var.get() == "auto":
            dimf.pack_forget()
            camf.pack(fill="x", padx=8, pady=4)
            canvas.config(cursor="arrow")
            canvas.unbind("<Button-1>")
            canvas.unbind("<Button-3>")
        else:
            camf.pack_forget()
            dimf.pack(fill="x", padx=8, pady=4)
            canvas.config(cursor="crosshair")
            canvas.bind("<Button-1>", on_canvas_click)
            canvas.bind("<Button-3>", on_canvas_rclick)
        _update_sidebar_state()

    modef = create_label_frame(sb, "Calibration Mode")
    tk.Radiobutton(modef, text="Manual (click 4 pts)",
                   variable=mode_var, value="manual", command=_refresh_mode,
                   bg=COLOR_SIDEBAR_BG, fg="#ccc", selectcolor="#2c3e50",
                   activebackground=COLOR_SIDEBAR_BG, font=("Consolas", 9)
                   ).pack(anchor="w")
    tk.Radiobutton(modef, text="Auto (camera params)",
                   variable=mode_var, value="auto", command=_refresh_mode,
                   bg=COLOR_SIDEBAR_BG, fg="#ccc", selectcolor="#2c3e50",
                   activebackground=COLOR_SIDEBAR_BG, font=("Consolas", 9)
                   ).pack(anchor="w")

    # ── Camera params panel (hidden until Auto mode selected) ──
    camf = create_label_frame(sb, "Camera Parameters")
    camf.pack_forget()   # hidden by default; shown when mode == "auto"

    create_label(camf, "Height above road (m):")
    cam_h_var = tk.DoubleVar(value=DEFAULT_CAM_HEIGHT_M)
    create_entry(camf, cam_h_var)

    create_label(camf, "Tilt/depression angle (deg):")
    cam_tilt_var = tk.DoubleVar(value=DEFAULT_CAM_TILT_DEG)
    create_entry(camf, cam_tilt_var)

    create_label(camf, "Horizontal FOV (deg):")
    cam_fov_var = tk.DoubleVar(value=DEFAULT_CAM_FOV_H_DEG)
    create_entry(camf, cam_fov_var)

    create_label(camf, "Road slope (deg, uphill+):")
    road_slope_var = tk.DoubleVar(value=DEFAULT_ROAD_SLOPE_DEG)
    create_entry(camf, road_slope_var)

    # Live preview of computed W×L
    preview_var = tk.StringVar(value="W: --  L: --")
    tk.Label(camf, textvariable=preview_var, anchor="w",
             bg=COLOR_SIDEBAR_BG, fg="#7dffb3",
             font=("Consolas", 8), wraplength=220).pack(fill="x", pady=(4,0))

    def _preview_wl(*_):
        """Compute and display projected W×L without closing the window."""
        import math, numpy as np
        try:
            h  = float(cam_h_var.get())
            td = float(cam_tilt_var.get())
            fv = float(cam_fov_var.get())
            sl = float(road_slope_var.get())
            fw, fh = img_size[0]
        except Exception:
            preview_var.set("W: ?  L: ?")
            return
        try:
            theta = math.radians(td);  alpha = math.radians(sl)
            f  = (fw / 2.0) / math.tan(math.radians(fv) / 2.0)
            cx, cy = fw / 2.0, fh / 2.0
            R = [[1, 0, 0],
                 [0, -math.sin(theta),  math.cos(theta)],
                 [0, -math.cos(theta), -math.sin(theta)]]
            C = [0.0, 0.0, h]
            n = [0.0, -math.sin(alpha), math.cos(alpha)]
            nC = n[1]*C[1] + n[2]*C[2]
            corners = [
                (int(fw*0.10), int(fh*0.35)), (int(fw*0.90), int(fh*0.35)),
                (int(fw*0.10), int(fh*0.90)), (int(fw*0.90), int(fh*0.90)),
            ]
            pts = []
            for u, v in corners:
                dc = [(u-cx)/f, (v-cy)/f, 1.0]
                dw = [R[r][0]*dc[0]+R[r][1]*dc[1]+R[r][2]*dc[2] for r in range(3)]
                denom = n[0]*dw[0]+n[1]*dw[1]+n[2]*dw[2]
                if abs(denom) < 1e-9: raise ValueError
                t = -nC / denom
                if t < 0: raise ValueError
                pts.append((C[0]+t*dw[0], C[1]+t*dw[1]))
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            W = max(xs)-min(xs);  L = max(ys)-min(ys)
            preview_var.set(f"ROI → W:{W:.1f}m  L:{L:.1f}m")
        except Exception:
            preview_var.set("W: err  L: err (check params)")

    for var in (cam_h_var, cam_tilt_var, cam_fov_var, road_slope_var):
        var.trace_add("write", _preview_wl)

    def _show_3d_preview(*_):
        """Open a Tk toplevel with an embedded 3-D calibration diagram.

        Left panel  – isometric 3-D: camera position, frustum rays to the
                      four ROI ground corners, ROI footprint, W/L labels.
        Right panel – 2-D side view: camera height h, tilt angle θ, near/
                      far distances on the ground plane.
        """
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
            from mpl_toolkits.mplot3d import Axes3D            # registers '3d'
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        except ImportError:
            status_var.set("pip install matplotlib")
            return

        import math as _m, numpy as _np

        # ── gather & validate params ──────────────────────────────────────
        try:
            h  = float(cam_h_var.get())
            td = float(cam_tilt_var.get())
            fv = float(cam_fov_var.get())
            sl = float(road_slope_var.get())
            fw, fh = img_size[0]
        except Exception:
            status_var.set("Set valid params first.")
            return
        if not (h > 0 and 0 < td < 90 and 0 < fv < 180):
            status_var.set("Need h>0, tilt 1–89°, FOV 1–179°.")
            return

        # ── project ROI corners onto the (possibly sloped) road plane ─────
        theta = _m.radians(td);  alpha = _m.radians(sl)
        f_px  = (fw / 2.0) / _m.tan(_m.radians(fv) / 2.0)
        cx, cy = fw / 2.0, fh / 2.0
        R = _np.array([[1,  0,              0            ],
                       [0, -_m.sin(theta),  _m.cos(theta)],
                       [0, -_m.cos(theta), -_m.sin(theta)]])
        C   = _np.array([0.0, 0.0, h])
        n_p = _np.array([0.0, -_m.sin(alpha), _m.cos(alpha)])
        nC  = float(n_p @ C)

        img_corners = [
            (int(fw * 0.10), int(fh * 0.35)),   # TL
            (int(fw * 0.90), int(fh * 0.35)),   # TR
            (int(fw * 0.10), int(fh * 0.90)),   # BL
            (int(fw * 0.90), int(fh * 0.90)),   # BR
        ]
        gpts = []
        for u, v in img_corners:
            dc  = _np.array([(u - cx) / f_px, (v - cy) / f_px, 1.0])
            dw  = R @ dc
            den = float(n_p @ dw)
            if abs(den) < 1e-9:
                status_var.set("Ray parallel to road — adjust tilt."); return
            t = -nC / den
            if t < 0:
                status_var.set("Ray behind camera — increase tilt."); return
            gpts.append(C + t * dw)        # 3-D point on ground

        tl3, tr3, bl3, br3 = gpts
        all_x = [p[0] for p in gpts];  all_y = [p[1] for p in gpts]
        W = max(all_x) - min(all_x)
        L = max(all_y) - min(all_y)
        near_y = min(all_y);  far_y = max(all_y)
        gx0, gx1 = min(all_x) - 1, max(all_x) + 1
        gy0, gy1 = near_y - 2,     far_y  + 2

        # ── Tk toplevel ───────────────────────────────────────────────────
        win = tk.Toplevel(root)
        win.title(f"3D Calibration  h={h}m  tilt={td}°  FOV={fv}°  slope={sl}°  "
                  f"→  W={W:.1f}m  L={L:.1f}m")
        win.geometry("980x530")
        win.configure(bg="#1a1a2e")

        BG = "#1a1a2e"
        fig = Figure(figsize=(9.8, 5.3), facecolor=BG)

        # ════════════════════════════════════════════════════════════════
        # LEFT — 3-D perspective view (road narrows to horizon)
        # ════════════════════════════════════════════════════════════════
        ax3 = fig.add_subplot(121, projection="3d")
        ax3.set_facecolor(BG)
        ax3.tick_params(colors="#666", labelsize=6)
        ax3.xaxis.pane.fill = ax3.yaxis.pane.fill = ax3.zaxis.pane.fill = False
        ax3.xaxis.pane.set_edgecolor("#333")
        ax3.yaxis.pane.set_edgecolor("#333")
        ax3.zaxis.pane.set_edgecolor("#333")
        # Perspective projection — road narrows toward horizon
        try:
            ax3.set_proj_type('persp', focal_length=0.35)
        except TypeError:
            ax3.set_proj_type('persp')
        # View from behind camera looking forward (+Y); elevation matches tilt
        ax3.view_init(elev=max(td + 5, 15), azim=-90)

        # Ground grid
        for gy in _np.arange(_m.floor(gy0), _m.ceil(gy1) + 1, 2):
            ax3.plot([gx0, gx1], [gy, gy], [0, 0], color="#2a2a2a", lw=0.6)
        for gx in _np.arange(_m.floor(gx0), _m.ceil(gx1) + 1, 2):
            ax3.plot([gx, gx], [gy0, gy1], [0, 0], color="#2a2a2a", lw=0.6)

        # ROI footprint — filled polygon on ground
        roi_v = [[[p[0], p[1], 0] for p in [tl3, tr3, br3, bl3]]]
        ax3.add_collection3d(Poly3DCollection(
            roi_v, alpha=0.25, facecolor="#00d4ff", edgecolor="#00d4ff", lw=2))

        # Frustum rays: camera → each ground corner
        ray_colors = ["#ff6b6b", "#4ecdc4", "#ff9f43", "#a29bfe"]
        ray_labels = ["TL", "TR", "BL", "BR"]
        for P, col, lbl in zip(gpts, ray_colors, ray_labels):
            ax3.plot([C[0], P[0]], [C[1], P[1]], [C[2], P[2]],
                     color=col, lw=1.3, alpha=0.75)
            ax3.scatter(P[0], P[1], 0, color=col, s=30, zorder=6)
            ax3.text(P[0], P[1], 0.4, lbl, color=col, fontsize=7)

        # Camera position
        ax3.scatter(*C, color="white", s=90, zorder=10)
        ax3.text(C[0] + 0.2, C[1], C[2] + 0.5,
                 f"Camera\n({h}m)", color="white", fontsize=7, ha="left")
        # Nadir — vertical pole from ground to camera (solid, clearly visible)
        ax3.plot([0, 0], [0, 0], [0, h], color="#a29bfe", lw=2.5, ls="-",
                 zorder=9, solid_capstyle='round')
        # Principal axis — line-of-sight (θ below horizontal, in Y-Z plane)
        pa_3d = min(h / max(_m.sin(theta), 0.05) * 0.55, far_y * 0.6)
        ax3.quiver(C[0], C[1], C[2],
                   0, _m.cos(theta) * pa_3d, -_m.sin(theta) * pa_3d,
                   color="#ffd32a", lw=1.8, arrow_length_ratio=0.12, alpha=0.75)

        # W annotation (near edge)
        mid_near_y = (bl3[1] + br3[1]) / 2
        ax3.plot([bl3[0], br3[0]], [mid_near_y] * 2, [0.35] * 2,
                 color="#f9ca24", lw=1.5)
        ax3.text((bl3[0] + br3[0]) / 2, mid_near_y, 0.8,
                 f"W = {W:.1f} m", color="#f9ca24", fontsize=8, ha="center")
        # L annotation (left edge)
        ax3.plot([tl3[0] - 0.4] * 2, [tl3[1], bl3[1]], [0.35] * 2,
                 color="#6ab04c", lw=1.5)
        ax3.text(tl3[0] - 0.5, (tl3[1] + bl3[1]) / 2, 0.8,
                 f"L={L:.1f}m", color="#6ab04c", fontsize=8, ha="right")

        ax3.set_xlabel("X  right (m)",   color="#888", fontsize=7, labelpad=3)
        ax3.set_ylabel("Y  forward (m)", color="#888", fontsize=7, labelpad=3)
        ax3.set_zlabel("Z  up (m)",      color="#888", fontsize=7, labelpad=3)
        ax3.set_title(f"3D View   W={W:.1f}m  L={L:.1f}m",
                      color="white", fontsize=9, pad=5)

        # ════════════════════════════════════════════════════════════════
        # RIGHT — 2-D side view (Y–Z plane)
        # ════════════════════════════════════════════════════════════════
        ax2 = fig.add_subplot(122, facecolor=BG)
        ax2.tick_params(colors="#888", labelsize=7)
        for sp in ("bottom", "left"):   ax2.spines[sp].set_color("#555")
        for sp in ("top",    "right"):  ax2.spines[sp].set_visible(False)
        ax2.grid(True, color="#252535", lw=0.7, zorder=0)

        # Ground
        ax2.axhline(0, color="#555", lw=1.5, zorder=2)
        ax2.fill_between([gy0, gy1], -0.9, 0, color="#1e3a10", alpha=0.5, zorder=1)

        # ROI band on ground
        ax2.fill_between([near_y, far_y], 0, 0.18,
                         color="#00d4ff", alpha=0.4, zorder=3)
        ax2.text((near_y + far_y) / 2, 0.28,
                 f"ROI  L = {L:.1f} m", color="#00d4ff", fontsize=8, ha="center")

        # Frustum rays (projected to YZ)
        for P, col in zip(gpts, ray_colors):
            ax2.plot([0, P[1]], [h, 0], color=col, lw=1.2, alpha=0.65, zorder=4)

        # Camera dot + label
        ax2.scatter(0, h, color="white", s=80, zorder=10)
        ax2.text(0.4, h + 0.3, f"h = {h} m", color="white", fontsize=9)
        # Nadir (vertical dashed — height reference, NOT the angle reference)
        ax2.plot([0, 0], [0, h], color="#555", lw=1, ls="--", alpha=0.5, zorder=5)

        # ── Tilt angle: measured from HORIZONTAL to line-of-sight ──────
        # (depression angle below horizontal plane)
        arc_r = h * 0.38

        # Horizontal reference line (the zero-angle baseline)
        ax2.plot([0, arc_r * 1.7], [h, h],
                 color="#aaa", lw=1.2, ls="--", alpha=0.7, zorder=5,
                 label="horizontal")

        # Principal axis arrow (camera line-of-sight at angle θ below horizontal)
        pa_len = min(h / max(_m.sin(theta), 0.05) * 0.55, far_y * 0.6)
        ax2.annotate("",
                     xy=(pa_len * _m.cos(theta), h - pa_len * _m.sin(theta)),
                     xytext=(0, h),
                     arrowprops=dict(arrowstyle="-|>", color="#ffd32a",
                                     lw=1.8, mutation_scale=12),
                     zorder=7)

        # Arc from 0 (horizontal) to -θ (depression below horizontal)
        arc_a = _np.linspace(0, -theta, 40)
        ax2.plot(arc_r * _np.cos(arc_a),
                 h  + arc_r * _np.sin(arc_a),
                 color="#ffd32a", lw=2, zorder=6)

        # Label at midpoint of arc
        mid_a = -theta / 2
        lx = arc_r * _m.cos(mid_a) + 0.3
        lz = h + arc_r * _m.sin(mid_a)
        ax2.text(lx, lz, f"θ = {td}°\n(from horiz.)",
                 color="#ffd32a", fontsize=8, va="center")

        # Near / far distance ticks
        for dist, col, va in [(near_y, "#ff9f43", "top"), (far_y, "#ff6b6b", "top")]:
            ax2.plot([dist, dist], [0, -0.55], color=col, lw=1, ls=":", zorder=5)
            ax2.text(dist, -0.65, f"{dist:.1f} m",
                     color=col, fontsize=8, ha="center", va=va)

        ax2.set_xlim(gy0 - 0.5, gy1 + 0.5)
        ax2.set_ylim(-1.2, h + 1.8)
        ax2.set_xlabel("Y — forward distance (m)", color="#888", fontsize=8)
        ax2.set_ylabel("Z — height (m)",           color="#888", fontsize=8)
        ax2.set_title("Side View  (camera profile)", color="white", fontsize=9)

        fig.tight_layout(pad=1.5)

        cv = FigureCanvasTkAgg(fig, master=win)
        cv.draw()
        cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    create_button(camf, "Preview 3D Geometry  [P]", _show_3d_preview)

    bf = create_label_frame(sb, "Actions")

    status_var = tk.StringVar(value="")
    status_lbl = tk.Label(bf, textvariable=status_var, anchor="w",
                          bg=COLOR_SIDEBAR_BG, fg=COLOR_STATUS_WARN,
                          font=("Consolas",8), wraplength=220)
    status_lbl.pack(fill="x", pady=(0,4))
    
    confirm_btn = create_button(bf, "Confirm Points  [ENTER]", do_confirm)
    confirm_btn.config(state="disabled", bg=COLOR_CONFIRM_DISABLED)
    
    create_button(bf, "Undo Last Point  [Z]", undo_last_point)
    create_button(bf, "Reset All Points  [R]", reset_all_points)

    if has_saved:
        def do_load_prev():
            result[0] = "load"
            root.quit()

        create_button(bf, "Use Previous Calibration", do_load_prev)
        
        status_var.set(f"Saved calibration found:\n{os.path.basename(cal_path)}")
    else:
        status_var.set("No saved calibration found.")
        
    create_button(bf, "Use Default  [ESC]", use_default)

    def _update_sidebar_state():
        if mode_var.get() == "auto" or len(clicked) == NEEDED_POINT:
            confirm_btn.config(state="normal", bg=COLOR_CONFIRM_ENABLED)
        else:
            confirm_btn.config(state="disabled", bg=COLOR_CONFIRM_DISABLED)

# =============================================================================
# Hotkeys
# =============================================================================
    def on_key(event):
        k = event.keysym.lower()
        if k == "return":
            do_confirm()
        elif k == "z":
            undo_last_point()
        elif k == "r":
            reset_all_points()
        elif k == "escape":
            use_default()
        elif k == "p" and mode_var.get() == "auto":
            _show_3d_preview()

    root.bind("<KeyPress>", on_key)
    root.protocol("WM_DELETE_WINDOW", lambda: (result.__setitem__(0,"default"), root.quit()))

    render_frame()
    root.mainloop()
    root.destroy()

# =============================================================================
# Apply result
# =============================================================================
    if result[0] == "new":
        try:
            w_m = float(width_var.get())
            l_m = float(length_var.get())
        except Exception:
            w_m, l_m = ROAD_WIDTH_M, ROAD_LENGTH_M
        calibrator.apply_points(clicked, w_m, l_m)
        calibrator.save(cal_path)
        return True

    elif result[0] == "auto":
        try:
            h  = float(cam_h_var.get())
            td = float(cam_tilt_var.get())
            fv = float(cam_fov_var.get())
            sl = float(road_slope_var.get())
            fw, fh = img_size[0]
        except Exception as e:
            print(f"[AutoCal] Error reading params: {e}. Using default.")
            calibrator.set_default()
            return True
        calibrator.apply_camera_params(fw, fh, h, td, fv, sl)
        calibrator.save(cal_path)
        return True

    elif result[0] == "load":
        if calibrator.load(cal_path):
            return True
        print("[WARN] Load failed. Using default.")
        calibrator.set_default()
        return True

    else:  # "default" or None
        loaded = calibrator.load(cal_path) if has_saved else False
        if not loaded:
            print("[WARN] Using default homography.")
            calibrator.set_default()
        return True