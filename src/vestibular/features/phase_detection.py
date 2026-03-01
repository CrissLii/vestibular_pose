"""Active-phase detection (trim idle noise) and periodic cycle detection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .velocity import speed_2d, smooth_series


# ---------------------------------------------------------------------------
# Active-phase detection — trim idle start/end
# ---------------------------------------------------------------------------

@dataclass
class ActivePhase:
    start_idx: int
    end_idx: int
    total_frames: int

    @property
    def duration_frames(self) -> int:
        return self.end_idx - self.start_idx


def detect_active_phase(
    positions_xy: np.ndarray,
    fps: float,
    idle_speed_thresh: float = 0.05,
    min_active_ratio: float = 0.3,
    smooth_window: int = 7,
) -> ActivePhase:
    """Find the contiguous active segment by trimming idle start/end.

    Args:
        positions_xy: (N, 2) array of a reference keypoint (e.g. mid_hip).
        fps: Video frame rate.
        idle_speed_thresh: Fraction of body-height-normalised speed below
            which frames are considered idle.  When body_height is unavailable,
            pass an absolute pixel threshold.
        min_active_ratio: Fallback — if detected active phase is shorter than
            this fraction of total, return full range.
        smooth_window: Moving-average window for speed smoothing.
    """
    n = len(positions_xy)
    if n < 10:
        return ActivePhase(0, n, n)

    spd = speed_2d(positions_xy, fps)
    spd = smooth_series(spd, smooth_window)

    active_mask = spd > idle_speed_thresh
    # pad to match original length (speed has N-1 elements)
    active_mask = np.append(active_mask, active_mask[-1])

    indices = np.where(active_mask)[0]
    if len(indices) == 0:
        return ActivePhase(0, n, n)

    start = int(indices[0])
    end = int(indices[-1]) + 1

    if (end - start) < min_active_ratio * n:
        return ActivePhase(0, n, n)

    return ActivePhase(start, end, n)


# ---------------------------------------------------------------------------
# Cycle detection for periodic actions (jumps, steps)
# ---------------------------------------------------------------------------

@dataclass
class Cycle:
    """One detected cycle (e.g. one jump: takeoff → peak → landing)."""
    start_idx: int
    peak_idx: int
    end_idx: int
    amplitude: float


def detect_cycles(
    signal: np.ndarray,
    fps: float,
    min_prominence: float | None = None,
    min_distance_sec: float = 0.2,
    invert: bool = False,
) -> List[Cycle]:
    """Detect periodic cycles via peak detection.

    For jump detection, pass the *negative* hip_y (or set invert=True)
    because image Y is downward — peaks in -Y correspond to highest jump.

    Args:
        signal: 1-D time series (e.g. hip_y for jumps).
        fps: Frame rate.
        min_prominence: Minimum peak prominence.  Auto-estimated if None.
        min_distance_sec: Minimum time between consecutive peaks.
        invert: If True, detect valleys instead of peaks (negate signal).

    Returns:
        List of Cycle objects sorted by start_idx.
    """
    from scipy.signal import find_peaks  # lazy import

    s = -signal.copy() if invert else signal.copy()

    min_distance = max(1, int(min_distance_sec * fps))

    if min_prominence is None:
        min_prominence = 0.15 * (np.max(s) - np.min(s))

    peaks, props = find_peaks(s, distance=min_distance, prominence=min_prominence)

    if len(peaks) < 1:
        return []

    # Find valleys between consecutive peaks as cycle boundaries
    cycles: List[Cycle] = []
    for i, pk in enumerate(peaks):
        # Cycle start: valley before this peak
        seg_start = peaks[i - 1] if i > 0 else 0
        left = int(np.argmin(s[seg_start:pk]) + seg_start)

        # Cycle end: valley after this peak
        seg_end = peaks[i + 1] if i < len(peaks) - 1 else len(s) - 1
        right = int(np.argmin(s[pk:seg_end + 1]) + pk)

        amp = float(s[pk] - min(s[left], s[right]))
        cycles.append(Cycle(start_idx=left, peak_idx=pk, end_idx=right, amplitude=amp))

    return cycles


def detect_stop_frame(
    positions_xy: np.ndarray,
    fps: float,
    speed_thresh_px: float = 5.0,
    sustained_frames: int | None = None,
    smooth_window: int = 7,
) -> int | None:
    """Detect the frame where movement stops (speed drops below threshold).

    Requires speed to stay below threshold for `sustained_frames` consecutive
    frames (default: 0.3 seconds).
    """
    n = len(positions_xy)
    if n < 10:
        return None

    if sustained_frames is None:
        sustained_frames = max(3, int(0.3 * fps))

    spd = speed_2d(positions_xy, fps)
    spd = smooth_series(spd, smooth_window)

    below = spd < speed_thresh_px
    count = 0
    for i, b in enumerate(below):
        if b:
            count += 1
            if count >= sustained_frames:
                return i - sustained_frames + 1
        else:
            count = 0
    return None
