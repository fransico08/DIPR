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
               tid, cls_name: str, speed: float,
               color: tuple) -> None:
    cv2.rectangle(frame, (l, t), (r, b), color, 2)
    label = f"ID{int(tid)} {cls_name} {speed:.1f}km/h"
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
    """Live HUD version: same quad style as calibration preview."""
    if not pts or len(pts) < 4:
        return
    draw_roi_quad(frame, pts)


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
