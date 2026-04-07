# utils/screen.py — Screen size detection and image fitting helpers

import tkinter as tk


def get_screen_size() -> tuple[int, int]:
    """Return (screen_width, screen_height) in pixels."""
    root = tk.Tk()
    root.withdraw()
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    root.destroy()
    return w, h


def fit_to_screen(img_w: int, img_h: int,
                  max_w: int, max_h: int,
                  margin: int = 80) -> tuple[int, int]:
    """Resize giữ nguyên tỷ lệ, ưu tiên vừa khung mà không cắt mất nội dung."""
    available_w = max(100, max_w - margin)
    available_h = max(100, max_h - margin)

    scale_w = available_w / img_w
    scale_h = available_h / img_h
    scale = min(scale_w, scale_h, 1.0)

    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    return new_w, new_h
