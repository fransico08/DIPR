import tkinter as tk

def get_screen_size():
    r = tk.Tk();
    r.withdraw()
    w, h = r.winfo_screenwidth(), r.winfo_screenheight()
    r.destroy()
    return w, h

def fit_to_screen(img_w, img_h, max_w, max_h, margin=80):
    scale = min((max_w - margin) / img_w, (max_h - margin) / img_h, 1.0)
    return int(img_w * scale), int(img_h * scale)