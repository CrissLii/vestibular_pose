from __future__ import annotations
import numpy as np

def angle_deg_between(v: np.ndarray, u: np.ndarray) -> float:
    v = v.astype(np.float64)
    u = u.astype(np.float64)
    denom = (np.linalg.norm(v) * np.linalg.norm(u)) + 1e-8
    cos = float(np.dot(v, u) / denom)
    cos = float(np.clip(cos, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))

def trunk_angle_deg(shoulder_mid: np.ndarray, hip_mid: np.ndarray) -> float:
    """Angle between trunk vector (hip->shoulder) and image vertical axis.
    Image coordinates have y downward. We define 'up' as (0,-1).
    """
    v = shoulder_mid - hip_mid
    vertical_up = np.array([0.0, -1.0], dtype=np.float64)
    return angle_deg_between(v, vertical_up)
