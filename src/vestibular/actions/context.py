"""Evaluation context passed to all action evaluators."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List

import numpy as np

from ..pose.keypoints import KeypointsFrame
from ..features.normalization import estimate_body_height_px


class ViewAngle(str, Enum):
    FRONT = "front"
    SIDE = "side"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    NORMAL = "正常"
    MILD = "轻度偏差"
    MODERATE = "中度偏差"
    SEVERE = "重度偏差"


SEVERITY_ORDER = {
    Severity.NORMAL: 0,
    Severity.MILD: 1,
    Severity.MODERATE: 2,
    Severity.SEVERE: 3,
}


def max_severity(*severities: Severity) -> Severity:
    return max(severities, key=lambda s: SEVERITY_ORDER[s])


@dataclass
class EvalContext:
    """Shared context for all evaluators."""
    kpt_frames: List[KeypointsFrame]
    fps: float
    view: ViewAngle = ViewAngle.UNKNOWN
    conf_thresh: float = 0.20
    body_height_px: float = field(init=False)

    def __post_init__(self):
        self.body_height_px = estimate_body_height_px(
            self.kpt_frames, self.conf_thresh
        )

    def norm(self, value_px: float) -> float:
        """Normalize a pixel value by body height."""
        if np.isnan(self.body_height_px) or self.body_height_px < 1.0:
            return float("nan")
        return value_px / self.body_height_px

    @property
    def duration_sec(self) -> float:
        return len(self.kpt_frames) / self.fps if self.fps > 0 else 0.0
