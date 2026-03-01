from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import numpy as np

from ..config import DEFAULT_CONF_KPT
from ..pose.keypoints import KeypointsFrame
from ..features.trunk_angle import trunk_angle_deg

# Default COCO-17 indices commonly used:
# 5/6 shoulders, 11/12 hips.
# If your YOLO pose keypoint order differs, adjust here.
LSH, RSH = 5, 6
LHP, RHP = 11, 12

@dataclass
class SpinMetrics:
    frames_total: int
    frames_used: int
    trunk_angle_std_deg: float
    trunk_angle_mean_deg: float

def compute_spin_metrics(
    kpt_frames: List[KeypointsFrame],
    conf_thresh: float = DEFAULT_CONF_KPT,
    min_frames: int = 10,
) -> tuple[SpinMetrics, Dict[str, Any]]:
    angles: list[float] = []
    used_indices: list[int] = []

    for f in kpt_frames:
        cf = f.conf
        xy = f.xy

        needed = [LSH, RSH, LHP, RHP]
        if any(i >= len(cf) for i in needed):
            continue
        if min(cf[LSH], cf[RSH], cf[LHP], cf[RHP]) < conf_thresh:
            continue

        shoulder_mid = (xy[LSH] + xy[RSH]) / 2.0
        hip_mid = (xy[LHP] + xy[RHP]) / 2.0
        angles.append(trunk_angle_deg(shoulder_mid, hip_mid))
        used_indices.append(f.frame_idx)

    if len(angles) < min_frames:
        metrics = SpinMetrics(
            frames_total=len(kpt_frames),
            frames_used=len(angles),
            trunk_angle_std_deg=float("nan"),
            trunk_angle_mean_deg=float("nan"),
        )
        debug = {"angles": angles, "used_frame_indices": used_indices, "note": "Not enough valid frames"}
        return metrics, debug

    a = np.asarray(angles, dtype=np.float64)
    metrics = SpinMetrics(
        frames_total=len(kpt_frames),
        frames_used=len(a),
        trunk_angle_std_deg=float(a.std()),
        trunk_angle_mean_deg=float(a.mean()),
    )
    debug = {"angles": angles, "used_frame_indices": used_indices}
    return metrics, debug
from enum import Enum

class Severity(str, Enum):
    NORMAL = "正常"
    MILD = "轻度偏差"
    MODERATE = "中度偏差"
    SEVERE = "重度偏差"

# def _severity_by_std(std_deg: float) -> Severity:
#     if np.isnan(std_deg):
#         return Severity.SEVERE
#     if std_deg <= 2.0:
#         return Severity.NORMAL
#     if std_deg <= 4.0:
#         return Severity.MILD
#     if std_deg <= 7.0:
#         return Severity.MODERATE
#     return Severity.SEVERE

def _severity_by_std(std_deg: float, t: dict) -> Severity:
    if np.isnan(std_deg):
        return Severity.SEVERE
    if std_deg <= t["mild_max"]:
        return Severity.NORMAL
    if std_deg <= t["moderate_max"]:
        return Severity.MILD
    if std_deg <= t["severe_max"]:
        return Severity.MODERATE
    return Severity.SEVERE

# def _severity_by_mean(mean_deg: float) -> Severity:
#     if np.isnan(mean_deg):
#         return Severity.SEVERE
#     if mean_deg <= 5.0:
#         return Severity.NORMAL
#     if mean_deg <= 10.0:
#         return Severity.MILD
#     if mean_deg <= 15.0:
#         return Severity.MODERATE
#     return Severity.SEVERE

def _severity_by_mean(mean_deg: float, t: dict) -> Severity:
    if np.isnan(mean_deg):
        return Severity.SEVERE
    if mean_deg <= t["mild_max"]:
        return Severity.NORMAL
    if mean_deg <= t["moderate_max"]:
        return Severity.MILD
    if mean_deg <= t["severe_max"]:
        return Severity.MODERATE
    return Severity.SEVERE

def _max_severity(a: Severity, b: Severity) -> Severity:
    order = {
        Severity.NORMAL: 0,
        Severity.MILD: 1,
        Severity.MODERATE: 2,
        Severity.SEVERE: 3,
    }
    return a if order[a] >= order[b] else b

def grade_spin(metrics: SpinMetrics, thresholds: dict | None = None) -> dict:
    """
    Return: {
      "pass": bool,
      "severity": "正常/轻度偏差/中度偏差/重度偏差",
      "reasons": {...},
      "suggestion": str
    }
    """
    # fallback：如果没给 thresholds，就用默认写死阈值（保证不崩）
    default_t_std = {"mild_max": 2.0, "moderate_max": 4.0, "severe_max": 7.0}
    default_t_mean = {"mild_max": 5.0, "moderate_max": 10.0, "severe_max": 15.0}

    t_std = default_t_std
    t_mean = default_t_mean
    if thresholds:
        t_std = thresholds["std_deg"]
        t_mean = thresholds["mean_deg"]

    sev_std = _severity_by_std(metrics.trunk_angle_std_deg, t_std)
    sev_mean = _severity_by_mean(metrics.trunk_angle_mean_deg, t_mean)
    sev = _max_severity(sev_std, sev_mean)

    passed = sev in (Severity.NORMAL, Severity.MILD)

    reasons = {
        "stability_std_deg": metrics.trunk_angle_std_deg,
        "stability_level": sev_std.value,
        "tilt_mean_deg": metrics.trunk_angle_mean_deg,
        "tilt_level": sev_mean.value,
    }

    # 简单的建议文案（可后续更细化）
    if sev == Severity.NORMAL:
        suggestion = "躯干很稳定且接近竖直，保持当前节奏与姿势即可。"
    elif sev == Severity.MILD:
        if sev_std.value != Severity.NORMAL.value and sev_mean.value == Severity.NORMAL.value:
            suggestion = "轻度晃动偏大，建议放慢旋转速度，关注躯干稳定。"
        elif sev_mean.value != Severity.NORMAL.value and sev_std.value == Severity.NORMAL.value:
            suggestion = "轻度倾斜，建议对齐身体中线，保持躯干竖直再旋转。"
        else:
            suggestion = "轻度晃动与倾斜同时存在，建议放慢速度并保持躯干竖直。"
    elif sev == Severity.MODERATE:
        suggestion = "中度偏差：稳定性或倾斜较明显，建议降低难度/减速，并在辅助下练习。"
    else:
        suggestion = "重度偏差：动作控制不足或姿态偏离明显，建议暂停高强度旋转，先做基础平衡训练。"

    return {
        "pass": passed,
        "severity": sev.value,
        "reasons": reasons,
        "suggestion": suggestion,
    }
