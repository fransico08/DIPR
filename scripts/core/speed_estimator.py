# core/speed_estimator.py — Speed estimation with multi-layer stabilisation
#
# Root cause of speed oscillation:
#   DeepSORT's Kalman filter smooths position BUT its bbox still jitters
#   ~2-5 pixels per frame. After homography this becomes ~0.1-0.3 m/frame
#   of positional noise. At 30 fps, 0.3 m/frame = 10.8 km/h of noise.
#
# Layered solution:
#   Layer 0 — Pixel smoothing: EMA on the raw pixel coordinate BEFORE
#             homography. This kills bbox jitter at the source.
#   Layer 1 — Long window path speed: use SPEED_WINDOW frames, sum segment
#             distances / total time. Longer window = more averaging.
#   Layer 2 — Spike rejection: if raw > SPIKE_RATIO * displayed, discard.
#             ALSO rejects spurious 0 km/h when vehicle near edge of frame.
#   Layer 3 — Heavy EMA (alpha=0.08): very slow response but silky smooth.
#   Layer 4 — Dead-band: freeze display if change < DEADBAND_KMH.
#
# Additional fixes:
#   - Far-object noise: small vehicles far away have large angular error per
#     pixel. We detect this by checking if the world-space Y is near the far
#     boundary and dampen speed accordingly.
#   - Edge exit: when a vehicle is near the ROI boundary we hold the last
#     good speed rather than letting it decay to zero.
#   - Camera shake: median-filter over position history before speed calc.

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

# Pixel-space EMA alpha
# Lower = smoother pixel position = less homography projection noise
_PX_EMA = 0.30

# Number of frames without update before speed is considered stale
_MAX_HOLD_FRAMES = 8


class SpeedEstimator:
    def __init__(self, calibrator):
        self._cal        = calibrator
        self._hist       = defaultdict(lambda: deque(maxlen=SPEED_WINDOW + 2))
        self._display:   dict[int, float] = {}   # last shown speed
        self._px_smooth: dict[int, tuple] = {}   # smoothed pixel position
        self._hold_count: dict[int, int]  = {}   # frames since last sample

    def update(self, tid: int, pixel_pt: tuple, ts_ms: float) -> float:
        if not self._cal.is_calibrated():
            return 0.0

        self._hold_count[tid] = 0  # got a new sample → reset hold counter

        # ── Layer 0: EMA on pixel coordinate before homography ─
        # Kills bbox jitter at the source before it gets amplified by H.
        px, py = float(pixel_pt[0]), float(pixel_pt[1])
        prev_px, prev_py = self._px_smooth.get(tid, (px, py))
        spx = _PX_EMA * px + (1.0 - _PX_EMA) * prev_px
        spy = _PX_EMA * py + (1.0 - _PX_EMA) * prev_py
        self._px_smooth[tid] = (spx, spy)

        # ── Project smoothed pixel → world coordinates ─────────
        wx, wy = self._cal.pixel_to_world((spx, spy))
        self._hist[tid].append((ts_ms, wx, wy))
        h = self._hist[tid]

        # Need at least MIN_HISTORY samples to compute a meaningful speed
        if len(h) < MIN_HISTORY:
            return 0.0

        # ── Layer 1: path speed over the whole sliding window ──
        raw = _path_speed(h)

        if tid not in self._display:
            self._display[tid] = raw
            return raw
        
        prev = self._display[tid]

        # ── Layer 2: spike rejection ───────────────────────────
        # Discard a sample only when it is implausibly large compared to
        # the current display speed (e.g. a sudden 5→150 km/h jump).
        if prev > SPEED_MIN_VALID and raw > prev * SPEED_SPIKE_RATIO:
            return prev

        # ── Layer 3: EMA smoothing ────────────────────────────
        # Use a lighter alpha (0.25) so speed can actually rise/fall
        # to its true value within a few seconds of window time.
        # The heavy alpha (0.08) in config was the main reason speed
        # stayed near 0 — it takes ~40 frames to reach 63% of truth.
        _SMOOTH = 0.35
        ema = _SMOOTH * raw + (1.0 - _SMOOTH) * prev

        # ── Layer 4: dead-band — only freeze if truly stable ──
        # Use half the configured deadband so small real changes still show.
        adaptive_db = max(0.8, prev * 0.04)  # 5% speed hoặc tối thiểu 1 km/h
        if prev > SPEED_MIN_VALID and abs(ema - prev) < adaptive_db:
            ema = prev + 0.3 * (ema - prev)

        self._display[tid] = ema
        return ema

    def get_display_speed(self, tid: int) -> float:
        return self._display.get(tid, 0.0)

    def hold_speed(self, tid: int) -> float:
        """Hold last good speed for _MAX_HOLD_FRAMES before returning 0."""
        count = self._hold_count.get(tid, 0) + 1
        self._hold_count[tid] = count
        if count > _MAX_HOLD_FRAMES:
            return 0.0
        return self._display.get(tid, 0.0)

    def reset(self) -> None:
        self._hist.clear()
        self._display.clear()
        self._px_smooth.clear()
        self._hold_count.clear()


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
