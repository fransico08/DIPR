import cv2
import os

import tkinter as tk
from utils.screen import get_screen_size, fit_to_screen
from config import ROAD_WIDTH_M, ROAD_LENGTH_M
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
    result = [None]   # "new", "load", "default", "cancel"

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
        if len(clicked) == NEEDED_POINT:
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