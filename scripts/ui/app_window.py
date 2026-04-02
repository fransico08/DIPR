import cv2
import tkinter as tk
from tkinter import ttk
from utils.ui_helpers import SIDEBAR_W, COLOR_SIDEBAR_BG, create_label_title, create_label_frame, create_label_variable, create_button, create_button_inline

SEEK_STEP_SHORT = 5
SEEK_STEP_LONG = 10

class AppWindow:
    def __init__(self, video_w, video_h,
                 on_reset, on_pause, on_quit,
                 on_screenshot, on_load_video, on_seek,
                 on_scale, scale_factor,
                 total_frames, fps_src):

        self.video_w  = video_w
        self.video_h  = video_h
        self._on_quit = on_quit
        self._on_seek = on_seek
        self._total   = total_frames
        self._alive   = True

        self.root = tk.Tk()
        self.root.title("Vehicle Speed Estimation")
        self.root.protocol("WM_DELETE_WINDOW", self._hard_quit)
        self.root.resizable(False, False)
        self.root.geometry(f"{video_w + SIDEBAR_W}x{video_h}")

# =============================================================================
# Canvas
# =============================================================================
        self.canvas = tk.Canvas(self.root, width=video_w, height=video_h,
                                bg="black", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self._photo = None

# =============================================================================
# Sidebar
# =============================================================================
        sb = tk.Frame(self.root, bg=COLOR_SIDEBAR_BG, width=SIDEBAR_W)
        sb.grid(row=0, column=1, sticky="nsew")
        sb.grid_propagate(False)
        self.root.columnconfigure(1, minsize=SIDEBAR_W)

        create_label_title(sb, "CONTROLS")


# =============================================================================
# Stats
# =============================================================================
        sf = create_label_frame(sb, "Live Stats")
        self._fv = tk.StringVar(value="FPS: --")
        self._tv = tk.StringVar(value="Tracked: --")
        self._cv = tk.StringVar(value="Homography: --")
        self._sv = tk.StringVar(value="Status: starting")
        for v in [self._fv, self._tv, self._cv, self._sv]:
            create_label_variable(sf, v)

# =============================================================================
# Seekbar
# =============================================================================
        vf = create_label_frame(sb, "Video Position")
        self._pos_v  = tk.StringVar(value="Frame: 0 / 0")
        tk.Label(vf, textvariable=self._pos_v, anchor="w", bg="#1a1a2e",
                 fg="#ccc", font=("Consolas",9)).pack(fill="x")
        self._seek_v = tk.IntVar(value=0)
        ttk.Scale(vf, from_=0, to=max(total_frames-1,1),
                  orient="horizontal", variable=self._seek_v,
                  command=lambda v: self._on_seek(int(float(v)))
                  ).pack(fill="x", pady=(4,2))
        ff = tk.Frame(vf, bg="#1a1a2e")
        ff.pack(fill="x")
        
        for lbl, sec in [
            (f"<<{SEEK_STEP_LONG}s", -SEEK_STEP_LONG),
            (f"<{SEEK_STEP_SHORT}s", -SEEK_STEP_SHORT),
            (f"+{SEEK_STEP_SHORT}s>", SEEK_STEP_SHORT),
            (f"+{SEEK_STEP_LONG}s>>", SEEK_STEP_LONG)
        ]:
            df = int(sec * fps_src)
            create_button_inline(ff, text=lbl, command=lambda d=df: self._seek_rel(d))

# =============================================================================
# Speed Scale
# =============================================================================
        xf = create_label_frame(sb, "Speed Scale Factor")
        tk.Label(xf, text="Nhân tốc độ ×:", anchor="w",
                 bg="#1a1a2e", fg="#ccc", font=("Consolas", 9)).pack(fill="x")
        self._scale_v = tk.DoubleVar(value=scale_factor)
        row = tk.Frame(xf, bg="#1a1a2e"); row.pack(fill="x", pady=(2, 4))
        tk.Entry(row, textvariable=self._scale_v, width=8,
                 bg="#2d2d44", fg="white", insertbackground="white",
                 font=("Consolas", 10)).pack(side="left", padx=(0, 6))
        create_button_inline(row, text="Áp dụng",
                             command=lambda: on_scale(self._scale_v.get()))

# =============================================================================
# Buttons
# =============================================================================
        bf = create_label_frame(sb, "Actions")
        for text, cmd in [
            ("Pause / Resume  [SPACE]", on_pause),
            ("Reset Tracker   [R]", on_reset),
            ("Screenshot      [S]", on_screenshot),
            ("Replay Video    ", self._replay),
            ("Load New Video  [L]", on_load_video),
            ("Quit            [ESC]", self._hard_quit),
        ]:
            create_button(bf, text=text, command=cmd)
            
    def _seek_rel(self, delta):
        t = max(0, min(self._total-1, self._seek_v.get() + delta))
        self._seek_v.set(t)
        self._on_seek(t)

    def _replay(self):
        self._seek_v.set(0)
        self._on_seek(0)

    def _hard_quit(self):
        """Immediately destroy window and signal quit — no confirm dialog."""
        self._alive = False
        self._on_quit()
        try: self.root.destroy()
        except Exception: pass

    def update_frame(self, bgr_frame):
        import PIL.Image, PIL.ImageTk
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        self._photo = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(rgb))
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

    def update_stats(self, fps, tracked, cal, paused, stopped, frame_no):
        if not self._alive: return
        self._fv.set(f"FPS: {fps:.1f}")
        self._tv.set(f"Tracked: {tracked}")
        self._cv.set(f"Homography: {'Calibrated' if cal else 'Default'}")
        self._sv.set("Status: " + (
            "Finished" if stopped else "PAUSED" if paused else "Running"))
        self._seek_v.set(frame_no)
        self._pos_v.set(f"Frame: {frame_no} / {self._total}")
        self.root.update()

    def destroy(self):
        self._alive = False
        try: self.root.destroy()
        except Exception: pass
