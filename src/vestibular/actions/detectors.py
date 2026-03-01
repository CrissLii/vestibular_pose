from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ..pose.keypoints import KeypointsFrame
from ..features.trunk_angle import trunk_angle_deg

# COCO-17 indices
NOSE = 0
LSH, RSH = 5, 6
LWR, RWR = 9, 10
LHP, RHP = 11, 12
LANK, RANK = 15, 16

@dataclass
class DetectionResult:
    action: str
    score: float
    details: Dict[str, float]

def _mid(xy: np.ndarray, a: int, b: int) -> np.ndarray:
    return (xy[a] + xy[b]) / 2.0

def _valid(cf: np.ndarray, ids: List[int], thr: float) -> bool:
    return all((i < len(cf) and cf[i] >= thr) for i in ids)

def _series(kpts: List[KeypointsFrame], conf_thr: float = 0.2) -> Dict[str, np.ndarray] | None:
    hip, sh, nose, wr, ank, trunk = [], [], [], [], [], []

    for f in kpts:
        xy, cf = f.xy, f.conf
        need = [NOSE, LSH, RSH, LHP, RHP, LWR, RWR, LANK, RANK]
        if not _valid(cf, need, conf_thr):
            continue

        hip_xy = _mid(xy, LHP, RHP)
        sh_xy = _mid(xy, LSH, RSH)
        wr_xy = _mid(xy, LWR, RWR)
        ank_xy = _mid(xy, LANK, RANK)

        hip.append(hip_xy)
        sh.append(sh_xy)
        nose.append(xy[NOSE])
        wr.append(wr_xy)
        ank.append(ank_xy)
        trunk.append(trunk_angle_deg(sh_xy, hip_xy))

    if len(hip) < 20:
        return None

    def arr(x): return np.asarray(x, dtype=np.float64)
    return {
        "hip": arr(hip), "sh": arr(sh), "nose": arr(nose),
        "wr": arr(wr), "ank": arr(ank),
        "trunk_deg": np.asarray(trunk, dtype=np.float64),
    }

def _std(x): return float(np.std(x))
def _mean(x): return float(np.mean(x))
def _range(x): return float(np.max(x) - np.min(x))


def _max_angular_velocity(trunk: np.ndarray) -> float:
    """Max angular velocity of trunk angle over a sliding window (°/frame)."""
    if len(trunk) < 5:
        return 0.0
    # Smooth to reduce noise
    kernel = np.ones(3) / 3
    smoothed = np.convolve(trunk, kernel, mode="valid")
    d_trunk = np.abs(np.diff(smoothed))
    # Max over a 3-frame window to capture peak rotation speed
    if len(d_trunk) < 3:
        return float(np.max(d_trunk))
    window_max = np.convolve(d_trunk, np.ones(3) / 3, mode="valid")
    return float(np.max(window_max))


def detect_action_mvp(kpts: List[KeypointsFrame]) -> Tuple[str, List[DetectionResult], Dict[str, float]]:
    s = _series(kpts)
    if s is None:
        cand = [DetectionResult("unknown", 0.0, {"n": 0})]
        return "unknown", cand, {}

    hip = s["hip"]; sh = s["sh"]; nose = s["nose"]
    wr = s["wr"]; ank = s["ank"]; trunk = s["trunk_deg"]

    # Basic stats
    nose_dx = _std(nose[:, 0])
    nose_dy = _std(nose[:, 1])
    hip_dx = _std(hip[:, 0])
    hip_dy = _std(hip[:, 1])
    hip_disp = float(np.linalg.norm(hip[-1] - hip[0]))
    trunk_mean = _mean(trunk)
    trunk_std = _std(trunk)
    trunk_rng = _range(trunk)
    trunk_max = float(np.max(trunk))

    vx = np.diff(hip[:, 0])
    speed = np.abs(vx)
    speed_mean = float(np.mean(speed))

    # Posture features
    wrist_below_hip_ratio = float(np.mean((wr[:, 1] > hip[:, 1]).astype(np.float64)))
    nose_above_sh_ratio = float(np.mean((nose[:, 1] < sh[:, 1]).astype(np.float64)))
    trunk_upright_ratio = float(np.mean((trunk < 25.0).astype(np.float64)))

    # Wrist extended below shoulder (arms supporting body on ground)
    wrist_to_sh_dist = wr[:, 1] - sh[:, 1]
    sh_to_ank = np.abs(sh[:, 1] - ank[:, 1])
    median_body_h = float(np.median(sh_to_ank)) + 1e-8
    wrist_extended_ratio = float(np.mean(wrist_to_sh_dist > 0.3 * median_body_h))

    # Max trunk angular velocity (key for forward roll detection)
    max_ang_vel = _max_angular_velocity(trunk)

    # How long trunk stays above 40° (sustained vs transient)
    trunk_above_40_ratio = float(np.mean((trunk > 40.0).astype(np.float64)))

    # --- SCORES ---

    # 1) run_straight: large horizontal + mostly upright + high speed
    run_horiz = min(1.0, hip_disp / 100.0)
    run_upright = min(1.0, trunk_upright_ratio / 0.5)
    run_speed = min(1.0, speed_mean / 3.0)
    run_score = run_horiz * run_upright * max(0.3, run_speed)

    # 2) jump_in_place: vertical oscillation + small horizontal + upright
    jump_vert = min(1.0, hip_dy / 25.0)
    jump_no_horiz = 1.0 - min(1.0, hip_disp / 100.0)
    jump_upright = min(1.0, trunk_upright_ratio / 0.4)
    jump_score = jump_vert * max(0.3, jump_no_horiz) * max(0.3, jump_upright)

    # 3) spin_in_place: small translation + stable trunk
    nose_x_range = _range(nose[:, 0])
    spin_no_move = 1.0 - min(1.0, hip_disp / 60.0)
    spin_stable = 1.0 - min(1.0, trunk_std / 5.0)
    spin_osc = min(1.0, nose_x_range / 40.0)
    spin_score = spin_no_move * spin_stable * max(0.2, spin_osc)

    # 4) forward_roll: large trunk range + high angular velocity + trunk goes > 60°
    #    Forward roll is a transient event: brief excursion, little forward travel
    roll_trunk = min(1.0, trunk_rng / 50.0)
    roll_ang_vel = min(1.0, max_ang_vel / 3.0)
    roll_peak = min(1.0, trunk_max / 60.0)
    roll_score = roll_trunk * max(0.3, roll_ang_vel) * roll_peak
    # Penalize if trunk stays above 40° for a long time (sustained = not a roll)
    if trunk_above_40_ratio > 0.3:
        roll_score *= max(0.1, 1.0 - trunk_above_40_ratio)
    # Penalize large forward displacement (rolls don't travel far)
    if hip_disp > 100:
        roll_score *= max(0.1, 1.0 - (hip_disp - 100) / 500.0)

    # Fraction of frames with low speed (stationary indicator)
    low_speed_frames = float(np.mean(speed < 2.0))

    # 5) wheelbarrow_walk: wrists extended + strong forward motion + not upright
    wheel_wrist = min(1.0, wrist_extended_ratio / 0.5)
    wheel_motion = min(1.0, hip_disp / 150.0)
    wheel_not_upright = max(0.0, 1.0 - trunk_upright_ratio)
    wheel_active = 1.0 - low_speed_frames
    wheel_score = wheel_wrist * wheel_motion * wheel_not_upright * max(0.2, wheel_active)
    # Require meaningful forward displacement for wheelbarrow
    if hip_disp < 250:
        wheel_score *= max(0.1, hip_disp / 250.0)

    # 6) head_up_prone: wrists extended + mostly stationary + head up
    headup_wrist = min(1.0, wrist_extended_ratio / 0.5)
    headup_stationary = min(1.0, low_speed_frames / 0.4)
    headup_nose_up = min(1.0, nose_above_sh_ratio / 0.6)
    headup_not_upright = max(0.0, 1.0 - trunk_upright_ratio)
    headup_score = headup_wrist * headup_stationary * headup_nose_up * headup_not_upright
    # Penalize large forward displacement
    if hip_disp > 400:
        headup_score *= 0.1

    candidates = [
        DetectionResult("run_straight", float(run_score),
                        {"hip_disp": hip_disp, "trunk_upright": trunk_upright_ratio, "speed_mean": speed_mean}),
        DetectionResult("jump_in_place", float(jump_score),
                        {"hip_dy": hip_dy, "hip_disp": hip_disp}),
        DetectionResult("spin_in_place", float(spin_score),
                        {"hip_disp": hip_disp, "trunk_std": trunk_std}),
        DetectionResult("forward_roll", float(roll_score),
                        {"trunk_range": trunk_rng, "max_ang_vel": max_ang_vel, "trunk_max": trunk_max}),
        DetectionResult("wheelbarrow_walk", float(wheel_score),
                        {"wrist_extended": wrist_extended_ratio, "hip_disp": hip_disp}),
        DetectionResult("head_up_prone", float(headup_score),
                        {"wrist_extended": wrist_extended_ratio, "hip_disp": hip_disp, "nose_above_sh": nose_above_sh_ratio}),
    ]
    candidates.sort(key=lambda x: x.score, reverse=True)

    best = candidates[0].action
    if candidates[0].score < 0.08:
        best = "spin_in_place"

    feat = {
        "nose_dx": nose_dx, "nose_dy": nose_dy,
        "hip_dx": hip_dx, "hip_dy": hip_dy,
        "hip_disp": hip_disp,
        "trunk_mean": trunk_mean, "trunk_std": trunk_std,
        "trunk_range": trunk_rng, "trunk_max": trunk_max,
        "trunk_upright_ratio": trunk_upright_ratio,
        "trunk_above_40_ratio": trunk_above_40_ratio,
        "max_ang_vel": max_ang_vel,
        "wrist_below_hip_ratio": wrist_below_hip_ratio,
        "wrist_extended_ratio": wrist_extended_ratio,
        "nose_above_sh_ratio": nose_above_sh_ratio,
        "speed_mean": speed_mean,
    }
    return best, candidates, feat
