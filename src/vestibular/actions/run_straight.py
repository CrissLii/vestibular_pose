from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any

import numpy as np
from ..pose.keypoints import KeypointsFrame

LHP, RHP = 11, 12

class Severity(str, Enum):
    NORMAL = "正常"
    MILD = "轻度偏差"
    MODERATE = "中度偏差"
    SEVERE = "重度偏差"

@dataclass
class RunMetrics:
    frames_total: int
    frames_used: int
    hip_dx_px: float
    accel_ratio: float       # later speed / early speed
    path_wobble_px: float    # y std of hip (wobble)

def compute_run_metrics(kpt_frames: List[KeypointsFrame], conf_thresh: float = 0.2) -> tuple[RunMetrics, Dict[str, Any]]:
    hip = []
    used = 0
    for f in kpt_frames:
        xy, cf = f.xy, f.conf
        if max(LHP, RHP) >= len(cf):
            continue
        if min(cf[LHP], cf[RHP]) < conf_thresh:
            continue
        hip.append(((xy[LHP] + xy[RHP]) / 2.0).astype(np.float64))
        used += 1

    n = len(kpt_frames)
    if used < 20:
        m = RunMetrics(n, used, float("nan"), float("nan"), float("nan"))
        return m, {"frames_total": n, "frames_used": used}

    hip = np.asarray(hip, dtype=np.float64)
    hip_dx = float(np.std(hip[:, 0]))
    wobble = float(np.std(hip[:, 1]))

    vx = np.diff(hip[:, 0])
    speed = np.abs(vx)
    mid = len(speed) // 2
    accel_ratio = float((np.mean(speed[mid:]) + 1e-6) / (np.mean(speed[:mid]) + 1e-6))

    m = RunMetrics(n, used, hip_dx, accel_ratio, wobble)
    return m, {"frames_total": n, "frames_used": used}

def _sev_from_dx(dx: float) -> Severity:
    if np.isnan(dx): return Severity.SEVERE
    if dx >= 90: return Severity.NORMAL
    if dx >= 60: return Severity.MILD
    if dx >= 35: return Severity.MODERATE
    return Severity.SEVERE

def _sev_from_accel(r: float) -> Severity:
    if np.isnan(r): return Severity.SEVERE
    if r >= 1.25: return Severity.NORMAL
    if r >= 1.10: return Severity.MILD
    if r >= 1.03: return Severity.MODERATE
    return Severity.SEVERE

def _sev_from_wobble(w: float) -> Severity:
    if np.isnan(w): return Severity.SEVERE
    if w <= 12: return Severity.NORMAL
    if w <= 20: return Severity.MILD
    if w <= 35: return Severity.MODERATE
    return Severity.SEVERE

def _max_sev(a: Severity, b: Severity, c: Severity) -> Severity:
    order = {Severity.NORMAL: 0, Severity.MILD: 1, Severity.MODERATE: 2, Severity.SEVERE: 3}
    best = a
    for s in (b, c):
        if order[s] > order[best]:
            best = s
    return best

def grade_run(m: RunMetrics) -> Dict[str, Any]:
    s_dx = _sev_from_dx(m.hip_dx_px)
    s_acc = _sev_from_accel(m.accel_ratio)
    s_wob = _sev_from_wobble(m.path_wobble_px)
    sev = _max_sev(s_dx, s_acc, s_wob)
    passed = sev in (Severity.NORMAL, Severity.MILD)

    if sev == Severity.NORMAL:
        suggestion = "位移与加速特征明显，跑动路径稳定。"
    elif sev == Severity.MILD:
        suggestion = "轻度偏差：建议提高启动加速意图，并保持上身稳定减少晃动。"
    elif sev == Severity.MODERATE:
        suggestion = "中度偏差：加速不明显或路径波动偏大，建议分解练习启动与摆臂协调。"
    else:
        suggestion = "重度偏差：位移/加速特征不足或轨迹明显不稳，建议降低难度并纠正姿态。"

    return {
        "pass": passed,
        "severity": sev.value,
        "reasons": {
            "hip_dx_px": m.hip_dx_px,
            "dx_level": s_dx.value,
            "accel_ratio": m.accel_ratio,
            "accel_level": s_acc.value,
            "path_wobble_px": m.path_wobble_px,
            "wobble_level": s_wob.value,
        },
        "suggestion": suggestion,
    }
