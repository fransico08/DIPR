import tkinter as tk

FONT_TITLE = ("Consolas", 11, "bold")
FONT_NORMAL = ("Consolas", 9)
FONT_SMALL = ("Consolas", 8)

SIDEBAR_W = 280
COLOR_SIDEBAR_BG = "#1a1a2e"
COLOR_HEADER_BG = "#16213e"
COLOR_HEADER_FG = "#80d2ff"
COLOR_FRAME_FG = "#aaa"
COLOR_LABEL_FG = "#ccc"
COLOR_ENTRY_BG = "#2c3e50"
COLOR_BUTTON = "#457b9d"

def create_label_title(parent, text):
    label = tk.Label(
        parent,
        text="CONTROLS",
        bg=COLOR_HEADER_BG,
        fg=COLOR_HEADER_FG,
        font=FONT_TITLE,
        pady=8
    )
    label.pack(fill="x")
    return label

def create_label_frame(parent, text):
    label_frame = tk.LabelFrame(
        parent,
        text=text,
        bg=COLOR_SIDEBAR_BG,
        fg=COLOR_FRAME_FG,
        font=FONT_NORMAL,
        padx=6,
        pady=4
    )
    label_frame.pack(fill="x", padx=8, pady=4)
    return label_frame

def create_label(parent, text):
    label = tk.Label(
        parent,
        text=text,
        anchor="w",
        justify="left",
        bg=COLOR_SIDEBAR_BG,
        fg=COLOR_LABEL_FG,
        font=FONT_SMALL
    )
    label.pack(fill="x")
    return label

def create_label_variable(parent, text):
    label = tk.Label(
        parent,
        textvariable=text,
        anchor="w",
        justify="left",
        bg=COLOR_SIDEBAR_BG,
        fg=COLOR_LABEL_FG,
        font=FONT_SMALL
    )
    label.pack(fill="x")
    return label

def create_entry(parent, variable):
    entry = tk.Entry(
        parent,
        textvariable=variable,
        font=FONT_NORMAL,
        bg=COLOR_ENTRY_BG,
        fg="white",
        insertbackground="white"
    )

    entry.pack(fill="x", pady=2)
    return entry

def create_button(parent, text, command):
    button = tk.Button(
        parent,
        text=text,
        command=command,
        bg=COLOR_BUTTON,
        fg="white",
        font=("Consolas", 9, "bold"),
        relief="flat",
        padx=4,
        pady=5,
        cursor="hand2",
        anchor="w"
    )
    button.pack(fill="x", pady=2)
    return button

def create_button_inline(parent, text, command):
    button = tk.Button(
        parent,
        text=text,
        command=command,
        bg=COLOR_BUTTON,
        fg="white",
        font=("Consolas", 9, "bold"),
        relief="flat",
        padx=2,
        cursor="hand2",
        anchor="w",
    )
    button.pack(side="left", expand=True, fill="x", padx=1, pady=2)
    return button