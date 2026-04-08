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
                 total_frames, fps_src,
                 on_replay=None):

        self.video_w  = video_w
        self.video_h  = video_h
        self._on_quit   = on_quit
        self._on_seek   = on_seek
        self._on_replay = on_replay
        self._total     = total_frames
        self._alive     = True
        self._seek_lock = False   # prevent update_stats from triggering seek

        self.root = tk.Tk()
        self.root.title("Vehicle Speed Estimation")
        self.root.protocol("WM_DELETE_WINDOW", self._hard_quit)
        self.root.resizable(True, True)
        self.root.configure(bg="black")
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", lambda event: self._hard_quit())
        self.root.update_idletasks()   # flush geometry before first frame

        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, minsize=SIDEBAR_W, weight=0)

# =============================================================================
# Banner
# =============================================================================
        header = tk.Frame(self.root, bg=COLOR_HEADER_BG)
        header.grid(row=0, column=0, columnspan=2, sticky="nsew")
        tk.Label(header,
                 text="Bùi Duy Phong - 19110131   |   Huỳnh Minh Tài - 22110068   |   Lê Minh Ngọc - 22110056",
                 bg=COLOR_HEADER_BG, fg=COLOR_HEADER_FG,
                 font=("Consolas", 12, "bold"), pady=10).pack(fill="x")

# =============================================================================
# Canvas
# =============================================================================
        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self._photo = None

# =============================================================================
# Sidebar
# =============================================================================
        sb = tk.Frame(self.root, bg=COLOR_SIDEBAR_BG, width=SIDEBAR_W)
        sb.grid(row=1, column=1, sticky="nsew")
        sb.grid_propagate(False)

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
                  command=lambda v: (None if self._seek_lock
                                     else self._on_seek(int(float(v))))
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
# Buttons
# =============================================================================
        bf = create_label_frame(sb, "Actions")
        for text, cmd in [
            ("Pause / Resume  [SPACE]", on_pause),
            ("Reset Tracker   [R]",     on_reset),
            ("Screenshot      [S]",     on_screenshot),
            ("Replay Video    [R]",     self._replay),
            ("Load New Video  [L]",     on_load_video),
            ("Quit            [ESC]",   self._hard_quit),
        ]:
            create_button(bf, text=text, command=cmd)

        # Screenshot status notification
        self._status_v = tk.StringVar(value="")
        tk.Label(bf, textvariable=self._status_v, anchor="w",
                 bg=COLOR_SIDEBAR_BG, fg="#7dffb3",
                 font=("Consolas", 9), wraplength=260
                 ).pack(fill="x", pady=(4, 0))

        # Spacer — push content to top in tall windows
        tk.Frame(sb, bg=COLOR_SIDEBAR_BG).pack(fill="both", expand=True)
            
    def _seek_rel(self, delta):
        t = max(0, min(self._total-1, self._seek_v.get() + delta))
        self._seek_v.set(t)
        self._on_seek(t)

    def _replay(self):
        self._seek_lock = True
        self._seek_v.set(0)
        self._seek_lock = False
        if self._on_replay:
            self._on_replay()   # seek to 0 + unpause
        else:
            self._on_seek(0)

    def show_status(self, msg, ms=3000):
        """Show a temporary status message in the sidebar."""
        if not self._alive: return
        self._status_v.set(msg)
        self.root.after(ms, lambda: self._status_v.set("") if self._alive else None)

    def _hard_quit(self):
        """Immediately destroy window and signal quit — no confirm dialog."""
        self._alive = False
        self._on_quit()
        try: self.root.destroy()
        except Exception: pass

    def update_frame(self, bgr_frame):
        import PIL.Image, PIL.ImageTk
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return
        bgr_frame = cv2.resize(bgr_frame, (cw, ch), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        self._photo = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(rgb))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

    def update_stats(self, fps, tracked, cal, paused, stopped, frame_no):
        if not self._alive: return
        self._fv.set(f"FPS: {fps:.1f}")
        self._tv.set(f"Tracked: {tracked}")
        self._cv.set(f"Homography: {'Calibrated' if cal else 'Default'}")
        self._sv.set("Status: " + (
            "Finished" if stopped else "PAUSED" if paused else "Running"))
        self._seek_lock = True
        self._seek_v.set(frame_no)
        self._seek_lock = False
        self._pos_v.set(f"Frame: {frame_no} / {self._total}")
        self.root.update()

    def destroy(self):
        self._alive = False
        try: self.root.destroy()
        except Exception: pass
