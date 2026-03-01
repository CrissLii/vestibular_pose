"""Joint angle computation for arbitrary three-point chains (e.g. knee, elbow)."""
from __future__ import annotations

import numpy as np

# COCO-17 keypoint indices
NOSE = 0
LEYE, REYE = 1, 2
LEAR, REAR = 3, 4
LSH, RSH = 5, 6
LELB, RELB = 7, 8
LWR, RWR = 9, 10
LHP, RHP = 11, 12
LKNE, RKNE = 13, 14
LANK, RANK = 15, 16


def three_point_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex *b* formed by segments b→a and b→c, in degrees.

    Returns value in [0, 180].
    """
    ba = (a - b).astype(np.float64)
    bc = (c - b).astype(np.float64)
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cos_val = float(np.dot(ba, bc) / denom)
    cos_val = float(np.clip(cos_val, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_val)))


def knee_angle_deg(hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> float:
    """Knee flexion angle (hip-knee-ankle). 180° = fully extended."""
    return three_point_angle_deg(hip, knee, ankle)


def left_knee_angle(xy: np.ndarray) -> float:
    return knee_angle_deg(xy[LHP], xy[LKNE], xy[LANK])


def right_knee_angle(xy: np.ndarray) -> float:
    return knee_angle_deg(xy[RHP], xy[RKNE], xy[RANK])


def elbow_angle_deg(shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray) -> float:
    """Elbow flexion angle. 180° = fully extended."""
    return three_point_angle_deg(shoulder, elbow, wrist)
