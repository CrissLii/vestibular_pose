"""Comprehensive feature extraction for action classification.

Extracts ~35 kinematic and posture features from a keypoint sequence,
suitable for training a classifier (RandomForest, etc.).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.signal import find_peaks

from ..pose.keypoints import KeypointsFrame
from ..features.trunk_angle import trunk_angle_deg
from ..features.phase_detection import detect_active_phase

NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
LSH, RSH = 5, 6
L_ELB, R_ELB = 7, 8
LWR, RWR = 9, 10
LHP, RHP = 11, 12
L_KN, R_KN = 13, 14
LANK, RANK = 15, 16


def _mid(xy: np.ndarray, a: int, b: int) -> np.ndarray:
    return (xy[a] + xy[b]) / 2.0


def _valid(cf: np.ndarray, ids: List[int], thr: float) -> bool:
    return all((i < len(cf) and cf[i] >= thr) for i in ids)


def _autocorr_peak(signal: np.ndarray, min_lag: int = 5) -> float:
    """Normalized autocorrelation peak (excluding lag 0). Measures periodicity."""
    if len(signal) < min_lag * 3:
        return 0.0
    s = signal - np.mean(signal)
    norm = np.sum(s ** 2)
    if norm < 1e-8:
        return 0.0
    acf = np.correlate(s, s, mode="full")
    acf = acf[len(s) - 1:]  # keep positive lags
    acf = acf / norm
    # find first peak after min_lag
    search = acf[min_lag:]
    if len(search) < 3:
        return 0.0
    peaks, props = find_peaks(search, prominence=0.05)
    if len(peaks) == 0:
        return 0.0
    return float(search[peaks[0]])


def _count_peaks(signal: np.ndarray, min_prom_frac: float = 0.15,
                 min_dist: int = 5) -> int:
    """Count prominent peaks in a 1D signal."""
    if len(signal) < 10:
        return 0
    prom = min_prom_frac * (np.max(signal) - np.min(signal))
    if prom < 1e-8:
        return 0
    peaks, _ = find_peaks(signal, prominence=prom, distance=min_dist)
    return len(peaks)


def extract_features(
    kpts: List[KeypointsFrame],
    fps: float = 30.0,
    conf_thr: float = 0.2,
    use_active_filter: bool = True,
) -> Optional[Dict[str, float]]:
    """Extract a fixed-size feature dict from a keypoint sequence.

    Returns None if not enough valid frames.
    """
    hip_list, sh_list, nose_list, wr_list, ank_list, trunk_list = (
        [], [], [], [], [], []
    )
    lwr_list, rwr_list = [], []
    lank_list, rank_list = [], []
    lkn_list, rkn_list = [], []

    required = [NOSE, LSH, RSH, LHP, RHP, LWR, RWR, LANK, RANK]

    for f in kpts:
        xy, cf = f.xy, f.conf
        if not _valid(cf, required, conf_thr):
            continue

        hip_xy = _mid(xy, LHP, RHP)
        sh_xy = _mid(xy, LSH, RSH)
        wr_xy = _mid(xy, LWR, RWR)
        ank_xy = _mid(xy, LANK, RANK)

        hip_list.append(hip_xy)
        sh_list.append(sh_xy)
        nose_list.append(xy[NOSE])
        wr_list.append(wr_xy)
        ank_list.append(ank_xy)
        lwr_list.append(xy[LWR])
        rwr_list.append(xy[RWR])
        lank_list.append(xy[LANK])
        rank_list.append(xy[RANK])
        if _valid(cf, [L_KN, R_KN], conf_thr):
            lkn_list.append(xy[L_KN])
            rkn_list.append(xy[R_KN])
        else:
            lkn_list.append(np.full(2, np.nan))
            rkn_list.append(np.full(2, np.nan))
        trunk_list.append(trunk_angle_deg(sh_xy, hip_xy))

    if len(hip_list) < 30:
        return None

    hip = np.array(hip_list, dtype=np.float64)
    sh = np.array(sh_list, dtype=np.float64)
    nose = np.array(nose_list, dtype=np.float64)
    wr = np.array(wr_list, dtype=np.float64)
    ank = np.array(ank_list, dtype=np.float64)
    trunk = np.array(trunk_list, dtype=np.float64)

    # --- Active frame filtering ---
    total_n = len(hip)
    if use_active_filter and len(hip) > 30:
        phase = detect_active_phase(hip, fps, idle_speed_thresh=1.0)
        si, ei = phase.start_idx, phase.end_idx
        # pad a small margin
        margin = max(5, int(0.5 * fps))
        si = max(0, si - margin)
        ei = min(len(hip), ei + margin)
        if ei - si >= 30:
            hip = hip[si:ei]
            sh = sh[si:ei]
            nose = nose[si:ei]
            wr = wr[si:ei]
            ank = ank[si:ei]
            trunk = trunk[si:ei]

    n = len(hip)
    active_ratio = n / max(total_n, 1)

    # --- Body reference ---
    body_h = np.median(np.abs(sh[:, 1] - ank[:, 1])) + 1e-8

    # --- Motion features ---
    vx = np.diff(hip[:, 0])
    vy = np.diff(hip[:, 1])
    speed = np.sqrt(vx ** 2 + vy ** 2)
    hip_disp = float(np.linalg.norm(hip[-1] - hip[0]))
    hip_disp_norm = hip_disp / body_h
    traj_len = float(np.sum(speed))
    linearity = hip_disp / max(traj_len, 1e-8)

    speed_mean = float(np.mean(speed))
    speed_std = float(np.std(speed))
    speed_max = float(np.max(speed))
    speed_norm = speed_mean / body_h

    # --- Displacement features ---
    hip_dx = float(np.std(hip[:, 0]))
    hip_dy = float(np.std(hip[:, 1]))
    nose_dx = float(np.std(nose[:, 0]))
    nose_dy = float(np.std(nose[:, 1]))

    hip_dx_norm = hip_dx / body_h
    hip_dy_norm = hip_dy / body_h

    # --- Trunk features ---
    trunk_mean = float(np.mean(trunk))
    trunk_std = float(np.std(trunk))
    trunk_rng = float(np.max(trunk) - np.min(trunk))
    trunk_max = float(np.max(trunk))
    trunk_min = float(np.min(trunk))
    trunk_upright_ratio = float(np.mean(trunk < 25.0))
    trunk_above_40 = float(np.mean(trunk > 40.0))
    trunk_above_60 = float(np.mean(trunk > 60.0))

    # Angular velocity of trunk
    d_trunk = np.abs(np.diff(trunk))
    ang_vel_mean = float(np.mean(d_trunk))
    ang_vel_max = float(np.max(d_trunk)) if len(d_trunk) > 0 else 0.0
    # smoothed max angular velocity
    if len(d_trunk) >= 5:
        kernel = np.ones(3) / 3
        smoothed_dtrunk = np.convolve(d_trunk, kernel, mode="valid")
        ang_vel_smooth_max = float(np.max(smoothed_dtrunk))
    else:
        ang_vel_smooth_max = ang_vel_max

    # --- Wrist / posture features ---
    wrist_below_hip = float(np.mean(wr[:, 1] > hip[:, 1]))
    wrist_below_sh_dist = wr[:, 1] - sh[:, 1]
    wrist_extended = float(np.mean(wrist_below_sh_dist > 0.3 * body_h))
    nose_above_sh = float(np.mean(nose[:, 1] < sh[:, 1]))
    wrist_above_head = float(np.mean(wr[:, 1] < nose[:, 1]))

    # Wrist near ground (below ankle level) — key for wheelbarrow/head_up
    wrist_near_ground = float(np.mean(wr[:, 1] > ank[:, 1] - 0.1 * body_h))

    # --- Periodicity features ---
    # Vertical periodicity (jumps)
    vert_peaks = _count_peaks(-hip[:, 1])
    vert_autocorr = _autocorr_peak(-hip[:, 1])

    # Horizontal periodicity (spin wobble)
    horiz_peaks = _count_peaks(nose[:, 0])
    horiz_autocorr = _autocorr_peak(nose[:, 0])

    # --- Duration-normalized features ---
    duration_s = n / max(fps, 1.0)
    vert_freq = vert_peaks / max(duration_s, 1.0)
    horiz_freq = horiz_peaks / max(duration_s, 1.0)

    # --- Body compression (curl) --- used to detect forward roll
    body_h_series = np.abs(sh[:, 1] - ank[:, 1])
    body_h_std = float(np.std(body_h_series)) / body_h
    body_h_min_ratio = float(np.min(body_h_series)) / body_h

    # --- Low-speed frames ---
    low_speed_ratio = float(np.mean(speed < 2.0))

    # --- Symmetry features ---
    lwr_arr = np.array(lwr_list, dtype=np.float64)
    rwr_arr = np.array(rwr_list, dtype=np.float64)
    lr_wrist_diff_y = np.abs(lwr_arr[:len(hip), 1] - rwr_arr[:len(hip), 1])
    wrist_symmetry = float(np.mean(lr_wrist_diff_y)) / body_h

    # --- Trunk transition features (crosses above/below threshold) ---
    trunk_cross_30 = float(np.sum(np.diff((trunk > 30.0).astype(int)) != 0))
    trunk_cross_45 = float(np.sum(np.diff((trunk > 45.0).astype(int)) != 0))

    # --- Wrist-ankle distance (body extension) ---
    wrist_ank_dist = np.linalg.norm(wr - ank, axis=1)
    wrist_ank_norm = float(np.mean(wrist_ank_dist)) / body_h

    # --- Motion direction dominance ---
    if len(vx) > 0:
        horiz_energy = float(np.sum(vx ** 2))
        vert_energy = float(np.sum(vy ** 2))
        total_energy = horiz_energy + vert_energy + 1e-8
        horiz_dom = horiz_energy / total_energy
        vert_dom = vert_energy / total_energy
    else:
        horiz_dom = 0.5
        vert_dom = 0.5

    # --- Hip vertical range normalized ---
    hip_y_range_norm = float(np.max(hip[:, 1]) - np.min(hip[:, 1])) / body_h

    # --- Body bounding box aspect ratio ---
    body_width = np.abs(np.array(lwr_list, dtype=np.float64)[:len(hip), 0] -
                        np.array(rwr_list, dtype=np.float64)[:len(hip), 0])
    body_width_mean = float(np.mean(body_width)) + 1e-8
    body_aspect = body_width_mean / body_h

    # --- Nose-hip vertical distance (body curl/extension indicator) ---
    nose_hip_dist_y = nose[:, 1] - hip[:, 1]
    nose_hip_norm = float(np.mean(np.abs(nose_hip_dist_y))) / body_h
    nose_below_hip = float(np.mean(nose_hip_dist_y > 0))

    # --- Speed variance ratio (bursty vs steady motion) ---
    speed_cv = speed_std / (speed_mean + 1e-8)

    feat = {
        "hip_disp": hip_disp,
        "hip_disp_norm": hip_disp_norm,
        "hip_dx": hip_dx,
        "hip_dy": hip_dy,
        "hip_dx_norm": hip_dx_norm,
        "hip_dy_norm": hip_dy_norm,
        "nose_dx": nose_dx,
        "nose_dy": nose_dy,
        "speed_mean": speed_mean,
        "speed_std": speed_std,
        "speed_max": speed_max,
        "speed_norm": speed_norm,
        "traj_len": traj_len,
        "linearity": linearity,
        "trunk_mean": trunk_mean,
        "trunk_std": trunk_std,
        "trunk_rng": trunk_rng,
        "trunk_max": trunk_max,
        "trunk_min": trunk_min,
        "trunk_upright_ratio": trunk_upright_ratio,
        "trunk_above_40": trunk_above_40,
        "trunk_above_60": trunk_above_60,
        "ang_vel_mean": ang_vel_mean,
        "ang_vel_max": ang_vel_max,
        "ang_vel_smooth_max": ang_vel_smooth_max,
        "wrist_below_hip": wrist_below_hip,
        "wrist_extended": wrist_extended,
        "wrist_near_ground": wrist_near_ground,
        "wrist_above_head": wrist_above_head,
        "nose_above_sh": nose_above_sh,
        "vert_peaks": float(vert_peaks),
        "vert_autocorr": vert_autocorr,
        "horiz_peaks": float(horiz_peaks),
        "horiz_autocorr": horiz_autocorr,
        "vert_freq": vert_freq,
        "horiz_freq": horiz_freq,
        "body_h_std": body_h_std,
        "body_h_min_ratio": body_h_min_ratio,
        "low_speed_ratio": low_speed_ratio,
        "active_ratio": active_ratio,
        "wrist_symmetry": wrist_symmetry,
        "trunk_cross_30": trunk_cross_30,
        "trunk_cross_45": trunk_cross_45,
        "wrist_ank_norm": wrist_ank_norm,
        "horiz_dom": horiz_dom,
        "vert_dom": vert_dom,
        "hip_y_range_norm": hip_y_range_norm,
        "body_aspect": body_aspect,
        "nose_hip_norm": nose_hip_norm,
        "nose_below_hip": nose_below_hip,
        "speed_cv": speed_cv,
        "duration_s": duration_s,
        "n_frames": float(n),
    }
    return feat


FEATURE_NAMES = [
    "hip_disp", "hip_disp_norm", "hip_dx", "hip_dy",
    "hip_dx_norm", "hip_dy_norm", "nose_dx", "nose_dy",
    "speed_mean", "speed_std", "speed_max", "speed_norm",
    "traj_len", "linearity",
    "trunk_mean", "trunk_std", "trunk_rng", "trunk_max", "trunk_min",
    "trunk_upright_ratio", "trunk_above_40", "trunk_above_60",
    "ang_vel_mean", "ang_vel_max", "ang_vel_smooth_max",
    "wrist_below_hip", "wrist_extended", "wrist_near_ground",
    "wrist_above_head", "nose_above_sh",
    "vert_peaks", "vert_autocorr", "horiz_peaks", "horiz_autocorr",
    "vert_freq", "horiz_freq",
    "body_h_std", "body_h_min_ratio",
    "low_speed_ratio", "active_ratio", "wrist_symmetry",
    "trunk_cross_30", "trunk_cross_45",
    "wrist_ank_norm", "horiz_dom", "vert_dom",
    "hip_y_range_norm", "body_aspect",
    "nose_hip_norm", "nose_below_hip", "speed_cv",
    "duration_s", "n_frames",
]


def features_to_vector(feat: Dict[str, float]) -> np.ndarray:
    """Convert feature dict → fixed-length numpy vector for classifier."""
    return np.array([feat.get(k, 0.0) for k in FEATURE_NAMES], dtype=np.float64)
