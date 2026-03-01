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
    hip = []
    sh = []
    nose = []
    wr = []
    ank = []
    trunk = []

    used = 0
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

        trunk.append(trunk_angle_deg(sh_xy, hip_xy))  # degrees vs vertical
        used += 1

    if used < 20:
        return None

    def arr(x): return np.asarray(x, dtype=np.float64)
    return {
        "hip": arr(hip),
        "sh": arr(sh),
        "nose": arr(nose),
        "wr": arr(wr),
        "ank": arr(ank),
        "trunk_deg": np.asarray(trunk, dtype=np.float64),
    }

def _std(x): return float(np.std(x))
def _mean(x): return float(np.mean(x))
def _range(x): return float(np.max(x) - np.min(x))

def detect_action_mvp(kpts: List[KeypointsFrame]) -> Tuple[str, List[DetectionResult], Dict[str, float]]:
    """
    MVP 6-action detector (rule-based):
    - run_straight: strong horizontal translation + increasing speed (rough)
    - jump_in_place: strong vertical oscillation + small horizontal drift
    - spin_in_place: small translation + stable trunk angle
    - forward_roll: trunk angle range very large + nose vertical swing large
    - wheelbarrow_walk: wrist lower than hip (hands on ground) + horizontal motion
    - head_up_prone: trunk near horizontal + little translation + head lift (nose higher than shoulders)
    """
    s = _series(kpts)
    if s is None:
        cand = [DetectionResult("unknown", 0.0, {"n": 0})]
        return "unknown", cand, {}

    hip = s["hip"]; sh = s["sh"]; nose = s["nose"]; wr = s["wr"]; ank = s["ank"]; trunk = s["trunk_deg"]

    # basic motion stats
    nose_dx = _std(nose[:, 0])
    nose_dy = _std(nose[:, 1])
    hip_dx = _std(hip[:, 0])
    hip_dy = _std(hip[:, 1])
    hip_disp = float(np.linalg.norm(hip[-1] - hip[0]))
    trunk_std = _std(trunk)
    trunk_rng = _range(trunk)

    # approximate speed/accel along x using frame-to-frame diffs (fps unknown; relative is fine)
    vx = np.diff(hip[:, 0])
    speed = np.abs(vx)
    speed_mean = float(np.mean(speed))
    # "accel" proxy: later half speed > early half speed
    mid = len(speed) // 2
    accel_ratio = float((np.mean(speed[mid:]) + 1e-6) / (np.mean(speed[:mid]) + 1e-6))

    # posture relations (y larger means lower in image coords)
    # wrist lower than hip => hands likely on ground (wheelbarrow)
    wrist_below_hip_ratio = float(np.mean((wr[:, 1] > hip[:, 1]).astype(np.float64)))
    # head-up prone: trunk near horizontal (large trunk angle vs vertical) AND nose "above" shoulders (y smaller)
    nose_above_sh_ratio = float(np.mean((nose[:, 1] < sh[:, 1]).astype(np.float64)))

    # Scores (0~1-ish)
    # 1) run: big horizontal + accel
    run_score = min(1.0, hip_dx / 80.0) * min(1.0, max(0.0, (accel_ratio - 1.0)) / 0.8 + 0.2)

    # 2) jump: big vertical oscillation, small horizontal
    jump_score = min(1.0, nose_dy / 60.0) * (1.0 - min(1.0, nose_dx / 80.0))

    # 3) spin: small translation + stable trunk
    spin_score = (1.0 - min(1.0, hip_disp / 80.0)) * (1.0 - min(1.0, trunk_std / 3.0))

    # 4) forward roll: trunk angle range huge + nose y variation large
    roll_score = min(1.0, trunk_rng / 70.0) * min(1.0, nose_dy / 80.0)

    # 5) wheelbarrow: wrists on ground often + forward movement
    wheel_score = min(1.0, wrist_below_hip_ratio / 0.8) * min(1.0, hip_dx / 60.0)

    # 6) head-up prone: trunk close to horizontal (angle vs vertical large, e.g. >60) + low translation + nose above shoulders
    trunk_near_horizontal = float(np.mean((trunk > 60.0).astype(np.float64)))  # ratio
    low_motion = 1.0 - min(1.0, hip_disp / 60.0)
    headup_score = min(1.0, trunk_near_horizontal / 0.8) * min(1.0, nose_above_sh_ratio / 0.8) * max(0.0, low_motion)

    candidates = [
        DetectionResult("run_straight", float(run_score), {"hip_dx": hip_dx, "accel_ratio": accel_ratio}),
        DetectionResult("jump_in_place", float(jump_score), {"nose_dy": nose_dy, "nose_dx": nose_dx}),
        DetectionResult("spin_in_place", float(spin_score), {"hip_disp": hip_disp, "trunk_std": trunk_std}),
        DetectionResult("forward_roll", float(roll_score), {"trunk_range": trunk_rng, "nose_dy": nose_dy}),
        DetectionResult("wheelbarrow_walk", float(wheel_score), {"wrist_below_hip_ratio": wrist_below_hip_ratio, "hip_dx": hip_dx}),
        DetectionResult("head_up_prone", float(headup_score), {"trunk_horizontal_ratio": trunk_near_horizontal, "nose_above_sh_ratio": nose_above_sh_ratio}),
    ]
    candidates.sort(key=lambda x: x.score, reverse=True)

    best = candidates[0].action
    # fallback: if too uncertain, keep spin_in_place (you already have solid evaluator)
    if candidates[0].score < 0.18:
        best = "spin_in_place"

    feat = {
        "nose_dx": nose_dx, "nose_dy": nose_dy,
        "hip_dx": hip_dx, "hip_dy": hip_dy,
        "hip_disp": hip_disp,
        "trunk_std": trunk_std,
        "trunk_range": trunk_rng,
        "accel_ratio": accel_ratio,
        "wrist_below_hip_ratio": wrist_below_hip_ratio,
        "nose_above_sh_ratio": nose_above_sh_ratio,
        "trunk_horizontal_ratio": float(np.mean((trunk > 60.0).astype(np.float64))),
        "speed_mean": speed_mean,
    }
    return best, candidates, feat
