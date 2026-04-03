# ui/ui_helpers.py — Tkinter widget factory (no business logic)

import tkinter as tk
from tkinter import ttk
from typing import Callable

from config import (
    FONT_TITLE, FONT_NORMAL, FONT_BOLD, FONT_SMALL,
    COLOR_SIDEBAR_BG, COLOR_HEADER_BG, COLOR_HEADER_FG,
    COLOR_FRAME_FG, COLOR_LABEL_FG, COLOR_ENTRY_BG,
    COLOR_BTN_DEFAULT, COLOR_BTN_SEEK,
    UI_HEADER_PADY, UI_SECTION_PADX, UI_SECTION_PADY,
    UI_SECTION_PACK_PADX, UI_SECTION_PACK_PADY,
    UI_BTN_PADX, UI_BTN_PADY, UI_BTN_PACK_PADY,
    UI_SEEKBAR_PADY,
)


def create_header_label(parent: tk.Widget, text: str) -> tk.Label:
    label = tk.Label(parent, text=text,
                     bg=COLOR_HEADER_BG, fg=COLOR_HEADER_FG,
                     font=FONT_TITLE, pady=UI_HEADER_PADY)
    label.pack(fill="x")
    return label


def create_section(parent: tk.Widget, title: str) -> tk.LabelFrame:
    frame = tk.LabelFrame(parent, text=title,
                          bg=COLOR_SIDEBAR_BG, fg=COLOR_FRAME_FG,
                          font=FONT_NORMAL,
                          padx=UI_SECTION_PADX, pady=UI_SECTION_PADY)
    frame.pack(fill="x", padx=UI_SECTION_PACK_PADX, pady=UI_SECTION_PACK_PADY)
    return frame


def create_static_label(parent: tk.Widget, text: str) -> tk.Label:
    label = tk.Label(parent, text=text,
                     anchor="w", justify="left",
                     bg=COLOR_SIDEBAR_BG, fg=COLOR_LABEL_FG,
                     font=FONT_SMALL)
    label.pack(fill="x")
    return label


def create_var_label(parent: tk.Widget, variable: tk.StringVar) -> tk.Label:
    label = tk.Label(parent, textvariable=variable,
                     anchor="w", justify="left",
                     bg=COLOR_SIDEBAR_BG, fg=COLOR_LABEL_FG,
                     font=FONT_SMALL)
    label.pack(fill="x")
    return label


def create_entry(parent: tk.Widget, variable: tk.Variable) -> tk.Entry:
    entry = tk.Entry(parent, textvariable=variable,
                     font=FONT_NORMAL,
                     bg=COLOR_ENTRY_BG, fg="white",
                     insertbackground="white")
    entry.pack(fill="x", pady=UI_BTN_PACK_PADY)
    return entry


def create_seekbar(parent: tk.Widget,
                   variable: tk.IntVar,
                   total_frames: int,
                   on_change: Callable) -> ttk.Scale:
    bar = ttk.Scale(parent,
                    from_=0, to=max(total_frames - 1, 1),
                    orient="horizontal",
                    variable=variable,
                    command=lambda v: on_change(int(float(v))))
    bar.pack(fill="x", pady=UI_SEEKBAR_PADY)
    return bar


def create_button(parent: tk.Widget, text: str,
                  command: Callable,
                  bg: str = COLOR_BTN_DEFAULT) -> tk.Button:
    btn = tk.Button(parent, text=text, command=command,
                    bg=bg, fg="white",
                    font=FONT_BOLD, relief="flat",
                    padx=UI_BTN_PADX, pady=UI_BTN_PADY,
                    cursor="hand2", anchor="w")
    btn.pack(fill="x", pady=UI_BTN_PACK_PADY)
    return btn


def create_inline_button(parent: tk.Widget, text: str,
                         command: Callable,
                         bg: str = COLOR_BTN_SEEK) -> tk.Button:
    btn = tk.Button(parent, text=text, command=command,
                    bg=bg, fg="white",
                    font=FONT_SMALL, relief="flat",
                    padx=UI_BTN_PACK_PADY, cursor="hand2", anchor="w")
    btn.pack(side="left", expand=True, fill="x",
             padx=UI_BTN_PACK_PADY, pady=UI_BTN_PACK_PADY)
    return btn
