import cv2
import tkinter as tk
from tkinter import ttk
from config import SIDEBAR_W

class AppWindow:
    def __init__(self, video_w, video_h,
                 on_reset, on_pause, on_quit,
                 on_screenshot, on_load_video, on_seek,
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

        # Canvas
        self.canvas = tk.Canvas(self.root, width=video_w, height=video_h,
                                bg="black", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self._photo = None

        # Sidebar
        sb = tk.Frame(self.root, bg="#1a1a2e", width=SIDEBAR_W)
        sb.grid(row=0, column=1, sticky="nsew")
        sb.grid_propagate(False)
        self.root.columnconfigure(1, minsize=SIDEBAR_W)

        tk.Label(sb, text="CONTROLS", bg="#16213e", fg="#80d2ff",
                 font=("Consolas",11,"bold"), pady=8).pack(fill="x")

        # Stats
        sf = tk.LabelFrame(sb, text="Live Stats", bg="#1a1a2e", fg="#aaa",
                           font=("Consolas",9), padx=6, pady=4)
        sf.pack(fill="x", padx=8, pady=(8,4))
        self._fv = tk.StringVar(value="FPS: --")
        self._tv = tk.StringVar(value="Tracked: --")
        self._cv = tk.StringVar(value="Homography: --")
        self._sv = tk.StringVar(value="Status: starting")
        for v in [self._fv, self._tv, self._cv, self._sv]:
            tk.Label(sf, textvariable=v, anchor="w", bg="#1a1a2e",
                     fg="#ddd", font=("Consolas",9)).pack(fill="x")

        # Seekbar
        vf = tk.LabelFrame(sb, text="Video Position", bg="#1a1a2e", fg="#aaa",
                            font=("Consolas",9), padx=6, pady=4)
        vf.pack(fill="x", padx=8, pady=4)
        self._pos_v  = tk.StringVar(value="Frame: 0 / 0")
        tk.Label(vf, textvariable=self._pos_v, anchor="w", bg="#1a1a2e",
                 fg="#ccc", font=("Consolas",9)).pack(fill="x")
        self._seek_v = tk.IntVar(value=0)
        ttk.Scale(vf, from_=0, to=max(total_frames-1,1),
                  orient="horizontal", variable=self._seek_v,
                  command=lambda v: self._on_seek(int(float(v)))
                  ).pack(fill="x", pady=(4,2))
        ff = tk.Frame(vf, bg="#1a1a2e"); ff.pack(fill="x")
        for lbl, sec in [("<<10s",-10),("<5s",-5),("+5s>",5),("+10s>>",10)]:
            df = int(sec * fps_src)
            tk.Button(ff, text=lbl, bg="#2c3e50", fg="white",
                      font=("Consolas",8), relief="flat", padx=2,
                      command=lambda d=df: self._seek_rel(d)
                      ).pack(side="left", expand=True, fill="x", padx=1, pady=2)

        # Buttons
        bf = tk.LabelFrame(sb, text="Controls", bg="#1a1a2e", fg="#aaa",
                           font=("Consolas",9), padx=6, pady=4)
        bf.pack(fill="x", padx=8, pady=4)
        for text, cmd, bg in [
            ("Pause / Resume  [SPACE]", on_pause,          "#2d6a4f"),
            ("Reset Tracker   [R]",     on_reset,           "#1d3557"),
            ("Screenshot      [S]",     on_screenshot,      "#457b9d"),
            ("Replay Video",            self._replay,       "#374151"),
            ("Load New Video  [L]",     on_load_video,      "#5c4033"),
        ]:
            tk.Button(bf, text=text, command=cmd, bg=bg, fg="white",
                      font=("Consolas",9,"bold"), relief="flat",
                      padx=4, pady=5, cursor="hand2", anchor="w"
                      ).pack(fill="x", pady=2)

        # Quit — calls _hard_quit directly, no confirm dialog
        tk.Button(bf, text="Quit  [ESC]",
                  command=self._hard_quit,
                  bg="#9b2226", fg="white",
                  font=("Consolas",9,"bold"), relief="flat",
                  padx=4, pady=5, cursor="hand2", anchor="w"
                  ).pack(fill="x", pady=2)

        # Calibration guide
        gf = tk.LabelFrame(sb, text="Calibration Guide", bg="#1a1a2e", fg="#aaa",
                           font=("Consolas",9), padx=6, pady=4)
        gf.pack(fill="x", padx=8, pady=(4,8))
        tk.Label(gf, text=(
            "Click 4 road points:\n"
            "  1=Top-Left  2=Top-Right\n"
            "  3=Bot-Left  4=Bot-Right\n\n"
            "Right-click / Z = undo\n"
            "R = reset all\n"
            "ENTER = confirm\n\n"
            "Good refs: lane lines,\n"
            "crosswalk, parking bay."
        ), anchor="w", justify="left", bg="#1a1a2e",
           fg="#bbb", font=("Consolas",8)).pack(fill="x")

    def _seek_rel(self, delta):
        t = max(0, min(self._total-1, self._seek_v.get() + delta))
        self._seek_v.set(t); self._on_seek(t)

    def _replay(self):
        self._seek_v.set(0); self._on_seek(0)

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
