"""Body-height normalization using shoulder-to-ankle pixel distance."""
from __future__ import annotations

from typing import List

import numpy as np

from ..pose.keypoints import KeypointsFrame

LSH, RSH = 5, 6
LANK, RANK = 15, 16


def _body_height_one_frame(xy: np.ndarray, conf: np.ndarray, conf_thresh: float) -> float | None:
    """Return shoulder-midpoint to ankle-midpoint pixel distance for one frame."""
    needed = [LSH, RSH, LANK, RANK]
    if max(needed) >= len(conf):
        return None
    if min(conf[i] for i in needed) < conf_thresh:
        return None
    shoulder_mid = (xy[LSH] + xy[RSH]) / 2.0
    ankle_mid = (xy[LANK] + xy[RANK]) / 2.0
    dist = float(np.linalg.norm(shoulder_mid - ankle_mid))
    return dist if dist > 1.0 else None


def estimate_body_height_px(
    kpt_frames: List[KeypointsFrame],
    conf_thresh: float = 0.20,
) -> float:
    """Estimate stable body height (shoulder-to-ankle) in pixels.

    Uses the median across all valid frames to be robust against
    crouching / jumping frames.
    """
    heights: list[float] = []
    for f in kpt_frames:
        h = _body_height_one_frame(f.xy, f.conf, conf_thresh)
        if h is not None:
            heights.append(h)

    if not heights:
        return float("nan")
    return float(np.median(heights))


def normalize_px(value_px: float, body_height_px: float) -> float:
    """Convert a pixel measurement to body-height-normalized units.

    Returns NaN if body_height_px is invalid.
    """
    if np.isnan(body_height_px) or body_height_px < 1.0:
        return float("nan")
    return value_px / body_height_px
