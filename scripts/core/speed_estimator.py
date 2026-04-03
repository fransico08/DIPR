# core/speed_estimator.py — Speed estimation with multi-layer stabilisation
#
# Root cause of speed oscillation:
#   DeepSORT's Kalman filter smooths position BUT its bbox still jitters
#   ~2-5 pixels per frame. After homography this becomes ~0.1-0.3 m/frame
#   of positional noise. At 30 fps, 0.3 m/frame = 10.8 km/h of noise.
#
# Layered solution (outermost = last defence):
#   Layer 0 — Pixel smoothing: EMA on the raw pixel coordinate BEFORE
#             homography. This kills bbox jitter at the source.
#   Layer 1 — Long window path speed: use SPEED_WINDOW frames, sum segment
#             distances / total time. Longer window = more averaging.
#   Layer 2 — Spike rejection: if raw > SPIKE_RATIO * displayed, discard.
#   Layer 3 — Heavy EMA (alpha=0.08): very slow response but silky smooth.
#   Layer 4 — Dead-band: freeze display if change < DEADBAND_KMH.
#             This is the key to stopping the number "dancing" when speed
#             is constant — only update when the change is real.

import math
from collections import defaultdict, deque
from config import (
    SPEED_WINDOW, MIN_HISTORY,
    MS_TO_KMH, TS_MS_TO_S,
    SPEED_EMA_ALPHA,
    SPEED_DEADBAND_KMH,
    SPEED_SPIKE_RATIO,
    SPEED_MIN_VALID,
)

# Pixel-space EMA alpha — tighter than world-space EMA
# Lower value = smoother position = less homography noise
_PX_EMA = 0.15


class SpeedEstimator:
    def __init__(self, calibrator):
        self._cal        = calibrator
        self._hist       = defaultdict(lambda: deque(maxlen=SPEED_WINDOW + 1))
        self._display:   dict[int, float] = {}   # last shown speed
        self._px_smooth: dict[int, tuple] = {}   # smoothed pixel position

    def update(self, tid: int, pixel_pt: tuple, ts_ms: float) -> float:
        if not self._cal.is_calibrated():
            return 0.0

        # ── Layer 0: smooth pixel position before homography ──
        px, py = float(pixel_pt[0]), float(pixel_pt[1])
        prev_px, prev_py = self._px_smooth.get(tid, (px, py))
        spx = _PX_EMA * px + (1.0 - _PX_EMA) * prev_px
        spy = _PX_EMA * py + (1.0 - _PX_EMA) * prev_py
        self._px_smooth[tid] = (spx, spy)

        # ── World transform on smoothed pixel ─────────────────
        wx, wy = self._cal.pixel_to_world((spx, spy))
        self._hist[tid].append((ts_ms, wx, wy))
        h = self._hist[tid]
        if len(h) < MIN_HISTORY:
            return 0.0

        # ── Layer 1: path speed over full window ───────────────
        raw = _path_speed(h)

        prev = self._display.get(tid, raw)

        # ── Layer 2: spike rejection ───────────────────────────
        if prev > SPEED_MIN_VALID and raw > prev * SPEED_SPIKE_RATIO:
            return prev

        # ── Layer 3: heavy EMA ─────────────────────────────────
        ema = SPEED_EMA_ALPHA * raw + (1.0 - SPEED_EMA_ALPHA) * prev

        # ── Layer 4: dead-band ─────────────────────────────────
        if abs(ema - prev) < SPEED_DEADBAND_KMH:
            return prev

        self._display[tid] = ema
        return ema

    def get_display_speed(self, tid: int) -> float:
        return self._display.get(tid, 0.0)

    def reset(self) -> None:
        self._hist.clear()
        self._display.clear()
        self._px_smooth.clear()


def _path_speed(h: deque) -> float:
    """Total path length / total elapsed time → km/h."""
    total_dist = sum(
        math.hypot(h[i][1] - h[i-1][1], h[i][2] - h[i-1][2])
        for i in range(1, len(h))
    )
    dt = (h[-1][0] - h[0][0]) / TS_MS_TO_S
    if dt <= 0:
        return 0.0
    return total_dist / dt * MS_TO_KMH
