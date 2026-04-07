# utils/drawing.py — OpenCV drawing helpers (pure rendering, no business logic)

import cv2
import numpy as np

from config import (
    COLOR_PALETTE,
    HUD_ORIGIN_X, HUD_ORIGIN_Y, HUD_RECT_X2, HUD_RECT_Y2,
    HUD_OVERLAY_ALPHA, HUD_BG_COLOR,
    HUD_TEXT_X, HUD_LINE1_Y, HUD_LINE2_Y, HUD_LINE3_Y, HUD_LINE4_Y,
    HUD_TEXT_SCALE_TITLE, HUD_TEXT_SCALE_BODY,
    HUD_TITLE_COLOR, HUD_BODY_COLOR,
    HUD_CAL_OK_COLOR, HUD_CAL_NO_COLOR,
    HUD_RUNNING_COLOR, HUD_PAUSED_COLOR,
    ROI_COLOR, ROI_DOT_RADIUS, ROI_LABEL_OFFSET_X, ROI_LABEL_OFFSET_Y,
    ROI_LABEL_SCALE, ROI_QUAD_ORDER, ROI_LABEL_NAMES,
    ROI_DOT_RADIUS_CAL, ROI_LABEL_FONT_SCALE_CAL,
    CAL_POINT_COLOR, CAL_LINE_COLOR, CAL_LINE_THICKNESS,
    TRACK_LABEL_FONT_SCALE, TRACK_LABEL_PAD, TRACK_LABEL_TEXT_COLOR,
    TRACK_DOT_RADIUS, TRACK_DOT_BORDER,
)

_color_cache: dict[int, tuple] = {}


def get_color(tid) -> tuple:
    k = int(tid)
    if k not in _color_cache:
        _color_cache[k] = COLOR_PALETTE[k % len(COLOR_PALETTE)]
    return _color_cache[k]


def clear_color_cache() -> None:
    _color_cache.clear()


def draw_track(frame: np.ndarray,
               l: int, t: int, r: int, b: int,
               tid, cls_name: str, speed,
               color: tuple,
               in_zone: bool = True) -> None:
    """
    Draw a tracked vehicle bounding box + label.

    in_zone=True  → full-color box + speed label
    in_zone=False → dim/grey box only, no speed label (out-of-calibration)
    speed=None    → box with "---" label (zone but no speed yet)
    """
    if not in_zone:
        # Draw dimmed grey box for out-of-zone vehicles
        dim_color = (80, 80, 80)
        cv2.rectangle(frame, (l, t), (r, b), dim_color, 1)
        # Small label, no speed
        label = f"ID{int(tid)}"
        cv2.putText(frame, label, (l + 2, t - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (120, 120, 120), 1, cv2.LINE_AA)
        return

    cv2.rectangle(frame, (l, t), (r, b), color, 2)

    if speed is None:
        speed_str = "---"
    else:
        speed_str = f"{speed:.1f}km/h"

    label = f"ID{int(tid)} {cls_name} {speed_str}"
    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_DUPLEX, TRACK_LABEL_FONT_SCALE, 1)
    p = TRACK_LABEL_PAD
    cv2.rectangle(frame,
                  (l, t - th - 2 * p - baseline),
                  (l + tw + 2 * p, t),
                  color, -1)
    cv2.putText(frame, label,
                (l + p, t - p - baseline),
                cv2.FONT_HERSHEY_DUPLEX, TRACK_LABEL_FONT_SCALE,
                TRACK_LABEL_TEXT_COLOR, 1, cv2.LINE_AA)
    cx, cy = (l + r) // 2, b
    cv2.circle(frame, (cx, cy), TRACK_DOT_RADIUS, color, -1)
    cv2.circle(frame, (cx, cy), TRACK_DOT_RADIUS, TRACK_DOT_BORDER, 1)

    # Draw vehicle center cross
    cx_box = (l + r) // 2
    cy_box = (t + b) // 2
    cross_size = 6
    cv2.line(frame, (cx_box - cross_size, cy_box), (cx_box + cross_size, cy_box),
             (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(frame, (cx_box, cy_box - cross_size), (cx_box, cy_box + cross_size),
             (255, 255, 255), 1, cv2.LINE_AA)


# ── Shared ROI drawing — used by BOTH live HUD and calibration preview ────────

def draw_roi_quad(frame: np.ndarray, pts: list,
                  line_color: tuple = CAL_LINE_COLOR,
                  dot_color:  tuple = CAL_POINT_COLOR,
                  line_thickness: int = CAL_LINE_THICKNESS,
                  dot_radius: int = ROI_DOT_RADIUS_CAL,
                  label_scale: float = ROI_LABEL_FONT_SCALE_CAL,
                  labels: tuple = ROI_LABEL_NAMES) -> None:
    """
    Draw a 4-point calibration quadrilateral with corner labels.
    Point order expected: [TL, TR, BL, BR]  (same as apply_camera_params output).
    Polygon closed as TL→TR→BR→BL→TL.
    """
    if not pts or len(pts) < 4:
        return
    ip = [tuple(map(int, p)) for p in pts]
    order = ROI_QUAD_ORDER   # (0,1,3,2) = TL,TR,BR,BL
    for i in range(4):
        a = ip[order[i]]
        b = ip[order[(i + 1) % 4]]
        cv2.line(frame, a, b, line_color, line_thickness, cv2.LINE_AA)
    for i, pt in enumerate(ip):
        cv2.circle(frame, pt, dot_radius, dot_color, -1)
        cv2.putText(frame, labels[i],
                    (pt[0] + 8, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, label_scale,
                    dot_color, 1, cv2.LINE_AA)


def draw_roi(frame: np.ndarray, pts: list | None) -> None:
    """
    Live HUD version: enhanced ROI with depth/perspective gradient cues.
    """
    if not pts or len(pts) < 4:
        return
    _draw_roi_live(frame, pts)


def _draw_roi_live(frame: np.ndarray, pts: list) -> None:
    """
    Live tracking overlay of ROI — lighter version of the calibration enhanced draw.
    Semi-transparent fill + depth-coded edges.
    """
    tl, tr, bl, br = [tuple(map(int, p)) for p in pts]

    # Semi-transparent fill
    overlay = frame.copy()
    poly = np.array([tl, tr, br, bl], dtype=np.int32)
    cv2.fillPoly(overlay, [poly], (0, 60, 40))
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

    # Far edge: thin cyan
    cv2.line(frame, tl, tr, (0, 220, 255), 1, cv2.LINE_AA)
    # Near edge: thick amber
    cv2.line(frame, bl, br, (255, 160, 0), 3, cv2.LINE_AA)
    # Sides
    cv2.line(frame, tl, bl, (80, 200, 160), 2, cv2.LINE_AA)
    cv2.line(frame, tr, br, (80, 200, 160), 2, cv2.LINE_AA)

    # Corner dots
    for pt, col, r in [(tl, (0, 220, 255), 4), (tr, (0, 220, 255), 4),
                        (bl, (255, 160, 0), 7), (br, (255, 160, 0), 7)]:
        cv2.circle(frame, pt, r + 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(frame, pt, r,     col,             -1, cv2.LINE_AA)


def draw_hud(frame: np.ndarray,
             fps: float, tracked: int,
             calibrated: bool, paused: bool) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (HUD_ORIGIN_X, HUD_ORIGIN_Y),
                  (HUD_RECT_X2, HUD_RECT_Y2),
                  HUD_BG_COLOR, -1)
    cv2.addWeighted(overlay, HUD_OVERLAY_ALPHA,
                    frame,   1 - HUD_OVERLAY_ALPHA, 0, frame)
    cv2.putText(frame, "VEHICLE SPEED ESTIMATION",
                (HUD_TEXT_X, HUD_LINE1_Y),
                cv2.FONT_HERSHEY_DUPLEX, HUD_TEXT_SCALE_TITLE,
                HUD_TITLE_COLOR, 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}   Tracked: {tracked}",
                (HUD_TEXT_X, HUD_LINE2_Y),
                cv2.FONT_HERSHEY_SIMPLEX, HUD_TEXT_SCALE_BODY,
                HUD_BODY_COLOR, 1, cv2.LINE_AA)
    hom_color = HUD_CAL_OK_COLOR if calibrated else HUD_CAL_NO_COLOR
    cv2.putText(frame, f"Homography: {'Calibrated' if calibrated else 'Default'}",
                (HUD_TEXT_X, HUD_LINE3_Y),
                cv2.FONT_HERSHEY_SIMPLEX, HUD_TEXT_SCALE_BODY,
                hom_color, 1, cv2.LINE_AA)
    sc = HUD_PAUSED_COLOR if paused else HUD_RUNNING_COLOR
    cv2.putText(frame, "PAUSED" if paused else "RUNNING",
                (HUD_TEXT_X, HUD_LINE4_Y),
                cv2.FONT_HERSHEY_SIMPLEX, HUD_TEXT_SCALE_BODY,
                sc, 1, cv2.LINE_AA)
