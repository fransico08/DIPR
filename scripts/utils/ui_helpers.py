import tkinter as tk

FONT_TITLE = ("Consolas", 13, "bold")
FONT_NORMAL = ("Consolas", 10)
FONT_SMALL = ("Consolas", 9)

SIDEBAR_W = 290
COLOR_SIDEBAR_BG = "#1a1a2e"
COLOR_HEADER_BG = "#0f3460"
COLOR_HEADER_FG = "#00d4ff"
COLOR_FRAME_FG  = "#c0c0c0"
COLOR_LABEL_FG  = "#e8e8e8"
COLOR_ENTRY_BG  = "#2c3e50"
COLOR_BUTTON    = "#457b9d"

def create_label_title(parent, text):
    label = tk.Label(
        parent,
        text=text,
        bg=COLOR_HEADER_BG,
        fg=COLOR_HEADER_FG,
        font=FONT_TITLE,
        pady=10
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

def create_param_row(parent, label_text, var, from_, to, step=0.1):
    """Compact label + [entry | slider] row for numeric parameters."""
    tk.Label(parent, text=label_text, anchor="w",
             bg=COLOR_SIDEBAR_BG, fg=COLOR_LABEL_FG, font=FONT_SMALL
             ).pack(fill="x")
    row = tk.Frame(parent, bg=COLOR_SIDEBAR_BG)
    row.pack(fill="x", pady=(0, 3))
    entry = tk.Entry(row, textvariable=var, width=7, font=FONT_SMALL,
                     bg=COLOR_ENTRY_BG, fg="white", insertbackground="white")
    entry.pack(side="left", padx=(0, 3))
    slider = tk.Scale(row, variable=var, from_=from_, to=to, resolution=step,
                      orient="horizontal", bg=COLOR_SIDEBAR_BG, fg="#888",
                      troughcolor=COLOR_ENTRY_BG, activebackground=COLOR_BUTTON,
                      highlightthickness=0, bd=0, showvalue=False, takefocus=False)
    slider.pack(side="left", fill="x", expand=True)
    return entry, slider

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