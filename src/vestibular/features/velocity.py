"""Time-series kinematics: velocity, acceleration, jerk from position data."""
from __future__ import annotations

import numpy as np


def velocity_1d(positions: np.ndarray, fps: float) -> np.ndarray:
    """First derivative of 1-D position series → velocity.

    Returns array of length len(positions) - 1.
    """
    dt = 1.0 / fps
    return np.diff(positions) / dt


def acceleration_1d(positions: np.ndarray, fps: float) -> np.ndarray:
    """Second derivative of 1-D position → acceleration.

    Returns array of length len(positions) - 2.
    """
    vel = velocity_1d(positions, fps)
    dt = 1.0 / fps
    return np.diff(vel) / dt


def jerk_1d(positions: np.ndarray, fps: float) -> np.ndarray:
    """Third derivative → jerk. Length = len(positions) - 3."""
    acc = acceleration_1d(positions, fps)
    dt = 1.0 / fps
    return np.diff(acc) / dt


def speed_2d(positions_xy: np.ndarray, fps: float) -> np.ndarray:
    """Frame-to-frame speed from 2-D position array (N, 2).

    Returns array of length N - 1.
    """
    disp = np.diff(positions_xy, axis=0)
    dt = 1.0 / fps
    return np.linalg.norm(disp, axis=1) / dt


def trajectory_length(positions_xy: np.ndarray) -> float:
    """Total path length of a 2-D trajectory (sum of frame-to-frame displacements)."""
    if len(positions_xy) < 2:
        return 0.0
    disp = np.diff(positions_xy, axis=0)
    return float(np.sum(np.linalg.norm(disp, axis=1)))


def smooth_series(series: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving-average smoothing.  Preserves array length via 'same' mode."""
    if len(series) < window:
        return series
    kernel = np.ones(window) / window
    return np.convolve(series, kernel, mode="same")
