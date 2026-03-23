import cv2
import os

import tkinter as tk
from utils.screen import get_screen_size, fit_to_screen
from config import ROAD_WIDTH_M, ROAD_LENGTH_M

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
    ret, raw_frame = cap.read(); cap.release()
    if not ret:
        print("[ERROR] Cannot read first frame."); return False

    scr_w, scr_h = get_screen_size()
    cal_sidebar  = 260
    cw, ch = fit_to_screen(raw_frame.shape[1], raw_frame.shape[0],
                            scr_w - cal_sidebar, scr_h, margin=80)
    display_frame = cv2.resize(raw_frame, (cw, ch))

    # Check for existing calibration file
    cal_path = os.path.join(os.path.dirname(video_path), "calibration.npz")
    has_saved = os.path.exists(cal_path)

    clicked   = []
    result    = [None]   # "new", "load", "default", "cancel"

    # ── Build Tk window ───────────────────────────────────────
    root = tk.Tk()
    root.title("Calibration")
    root.resizable(False, False)
    root.geometry(f"{cw + cal_sidebar}x{ch}")

    # Canvas (left)
    canvas = tk.Canvas(root, width=cw, height=ch,
                       bg="black", highlightthickness=0, cursor="crosshair")
    canvas.grid(row=0, column=0)
    photo_ref = [None]

    def render_frame():
        f = display_frame.copy()
        # Draw existing points
        for i, pt in enumerate(clicked):
            cv2.circle(f, (pt[0],pt[1]), 8, (0,255,100), -1)
            cv2.putText(f, str(i+1), (pt[0]+10, pt[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,100), 2)
            if i > 0:
                cv2.line(f, tuple(clicked[i-1]), tuple(pt), (0,210,255), 2)
        if len(clicked) == 4:
            cv2.line(f, tuple(clicked[3]), tuple(clicked[0]), (0,210,255), 2)
        # Status text top-left
        status = f"Points: {len(clicked)} / 4"
        cv2.putText(f, status, (10, ch-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100,255,180), 1, cv2.LINE_AA)
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        photo_ref[0] = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(rgb))
        canvas.create_image(0, 0, anchor="nw", image=photo_ref[0])
        _update_sidebar_state()

    def on_canvas_click(event):
        if len(clicked) < 4:
            clicked.append([event.x, event.y])
            render_frame()

    def on_canvas_rclick(event):
        if clicked:
            clicked.pop(); render_frame()

    canvas.bind("<Button-1>", on_canvas_click)
    canvas.bind("<Button-3>", on_canvas_rclick)

    # Sidebar (right)
    sb = tk.Frame(root, bg="#1a1a2e", width=cal_sidebar)
    sb.grid(row=0, column=1, sticky="nsew")
    sb.grid_propagate(False)

    tk.Label(sb, text="CALIBRATION", bg="#16213e", fg="#80d2ff",
             font=("Consolas",11,"bold"), pady=8).pack(fill="x")

    # Instructions
    inf = tk.LabelFrame(sb, text="Instructions", bg="#1a1a2e", fg="#aaa",
                        font=("Consolas",9), padx=6, pady=4)
    inf.pack(fill="x", padx=8, pady=(8,4))
    tk.Label(inf, text=(
        "Click 4 road points:\n"
        "  1 = Top-Left\n"
        "  2 = Top-Right\n"
        "  3 = Bot-Left\n"
        "  4 = Bot-Right\n\n"
        "Pick a flat rectangle\n"
        "with known real size.\n"
        "e.g. lane lines,\n"
        "crosswalk, parking bay."
    ), anchor="w", justify="left", bg="#1a1a2e",
       fg="#ccc", font=("Consolas",8)).pack(fill="x")

    # Dimension inputs
    dimf = tk.LabelFrame(sb, text="Real-world size", bg="#1a1a2e", fg="#aaa",
                         font=("Consolas",9), padx=6, pady=4)
    dimf.pack(fill="x", padx=8, pady=4)
    tk.Label(dimf, text="Width (top/bot edge) m:", anchor="w",
             bg="#1a1a2e", fg="#ccc", font=("Consolas",8)).pack(fill="x")
    width_var = tk.DoubleVar(value=ROAD_WIDTH_M)
    tk.Entry(dimf, textvariable=width_var, font=("Consolas",9),
             bg="#2c3e50", fg="white", insertbackground="white"
             ).pack(fill="x", pady=2)
    tk.Label(dimf, text="Length (left/right edge) m:", anchor="w",
             bg="#1a1a2e", fg="#ccc", font=("Consolas",8)).pack(fill="x")
    length_var = tk.DoubleVar(value=ROAD_LENGTH_M)
    tk.Entry(dimf, textvariable=length_var, font=("Consolas",9),
             bg="#2c3e50", fg="white", insertbackground="white"
             ).pack(fill="x", pady=2)

    # Action buttons
    bf = tk.LabelFrame(sb, text="Actions", bg="#1a1a2e", fg="#aaa",
                       font=("Consolas",9), padx=6, pady=4)
    bf.pack(fill="x", padx=8, pady=4)

    status_var = tk.StringVar(value="")
    status_lbl = tk.Label(bf, textvariable=status_var, anchor="w",
                          bg="#1a1a2e", fg="#f4a261",
                          font=("Consolas",8), wraplength=220)
    status_lbl.pack(fill="x", pady=(0,4))

    confirm_btn = tk.Button(bf, text="Confirm Points  [ENTER]",
                            bg="#2d6a4f", fg="white",
                            font=("Consolas",9,"bold"), relief="flat",
                            padx=4, pady=5, cursor="hand2", anchor="w")
    confirm_btn.pack(fill="x", pady=2)

    tk.Button(bf, text="Undo Last Point  [Z]",
              command=lambda: (clicked.pop() if clicked else None, render_frame()),
              bg="#1d3557", fg="white", font=("Consolas",9,"bold"),
              relief="flat", padx=4, pady=5, cursor="hand2", anchor="w"
              ).pack(fill="x", pady=2)

    tk.Button(bf, text="Reset All Points  [R]",
              command=lambda: (clicked.clear(), render_frame()),
              bg="#374151", fg="white", font=("Consolas",9,"bold"),
              relief="flat", padx=4, pady=5, cursor="hand2", anchor="w"
              ).pack(fill="x", pady=2)

    if has_saved:
        def do_load_prev():
            result[0] = "load"; root.quit()

        tk.Button(bf, text="Use Previous Calibration",
                  command=do_load_prev,
                  bg="#457b9d", fg="white", font=("Consolas",9,"bold"),
                  relief="flat", padx=4, pady=5, cursor="hand2", anchor="w"
                  ).pack(fill="x", pady=2)
        status_var.set(f"Saved calibration found:\n{os.path.basename(cal_path)}")
    else:
        status_var.set("No saved calibration found.")

    tk.Button(bf, text="Use Default  [ESC]",
              command=lambda: (result.__setitem__(0, "default"), root.quit()),
              bg="#5c4033", fg="white", font=("Consolas",9,"bold"),
              relief="flat", padx=4, pady=5, cursor="hand2", anchor="w"
              ).pack(fill="x", pady=2)

    def _update_sidebar_state():
        if len(clicked) == 4:
            confirm_btn.config(state="normal", bg="#1d9e75")
        else:
            confirm_btn.config(state="disabled", bg="#444")

    def do_confirm():
        if len(clicked) != 4:
            status_var.set("Need exactly 4 points!"); return
        try:
            w = float(width_var.get())
            l = float(length_var.get())
        except Exception:
            status_var.set("Invalid dimensions."); return
        if w <= 0 or l <= 0:
            status_var.set("Dimensions must be > 0."); return
        result[0] = "new"; root.quit()

    confirm_btn.config(command=do_confirm, state="disabled", bg="#444")

    def on_key(event):
        k = event.keysym.lower()
        if k == "return":   do_confirm()
        elif k == "z" and clicked: clicked.pop(); render_frame()
        elif k == "r":      clicked.clear(); render_frame()
        elif k == "escape": result[0] = "default"; root.quit()

    root.bind("<KeyPress>", on_key)
    root.protocol("WM_DELETE_WINDOW", lambda: (result.__setitem__(0,"default"), root.quit()))

    render_frame()
    root.mainloop()
    root.destroy()

    # ── Apply result ──────────────────────────────────────────
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
