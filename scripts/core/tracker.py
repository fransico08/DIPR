# core/tracker.py — DeepSORT factory and HSV histogram embedder

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from config import (
    MAX_AGE, N_INIT, MAX_IOU_DIST, MAX_COSINE_DIST,
    EMBED_BINS_H, EMBED_BINS_S, EMBED_BINS_V, EMBED_DIM,
    HSV_HUE_MAX, HSV_SAT_MAX, HSV_VAL_MAX,
)


def make_tracker() -> DeepSort:
    return DeepSort(
        max_age             = MAX_AGE,
        n_init              = N_INIT,
        max_iou_distance    = MAX_IOU_DIST,
        max_cosine_distance = MAX_COSINE_DIST,
        embedder            = None,
        half                = False,
        bgr                 = True,
    )


def compute_embeddings(frame: np.ndarray, detections: list) -> list[np.ndarray]:
    H, W = frame.shape[:2]
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    out: list[np.ndarray] = []
    for det in detections:
        x, y, w, h = [int(v) for v in det[0]]
        x1, y1 = max(0, x),     max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        if x2 <= x1 or y2 <= y1:
            out.append(np.zeros(EMBED_DIM, dtype=np.float32))
            continue
        crop = hsv[y1:y2, x1:x2]
        feat = np.concatenate([
            cv2.calcHist([crop], [0], None, [EMBED_BINS_H], [0, HSV_HUE_MAX]).flatten(),
            cv2.calcHist([crop], [1], None, [EMBED_BINS_S], [0, HSV_SAT_MAX]).flatten(),
            cv2.calcHist([crop], [2], None, [EMBED_BINS_V], [0, HSV_VAL_MAX]).flatten(),
        ])
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat /= norm
        out.append(feat)
    return out
