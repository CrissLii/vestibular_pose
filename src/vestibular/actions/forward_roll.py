from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any

import numpy as np
from ..pose.keypoints import KeypointsFrame
from ..features.trunk_angle import trunk_angle_deg

NOSE = 0
LSH, RSH = 5, 6
LHP, RHP = 11, 12

class Severity(str, Enum):
    NORMAL = "正常"
    MILD = "轻度偏差"
    MODERATE = "中度偏差"
    SEVERE = "重度偏差"

@dataclass
class RollMetrics:
    frames_total: int
    frames_used: int
    trunk_angle_range_deg: float
    nose_vertical_range_px: float
    translation_px: float

def compute_roll_metrics(kpt_frames: List[KeypointsFrame], conf_thresh: float = 0.2) -> tuple[RollMetrics, Dict[str, Any]]:
    trunk = []
    nose_y = []
    hip = []
    used = 0

    for f in kpt_frames:
        xy, cf = f.xy, f.conf
        need = [NOSE, LSH, RSH, LHP, RHP]
        if max(need) >= len(cf):
            continue
        if min(cf[i] for i in need) < conf_thresh:
            continue

        sh = (xy[LSH] + xy[RSH]) / 2.0
        hp = (xy[LHP] + xy[RHP]) / 2.0
        trunk.append(trunk_angle_deg(sh, hp))
        nose_y.append(float(xy[NOSE][1]))
        hip.append(hp.astype(np.float64))
        used += 1

    n = len(kpt_frames)
    if used < 20:
        m = RollMetrics(n, used, float("nan"), float("nan"), float("nan"))
        return m, {"frames_total": n, "frames_used": used}

    trunk = np.asarray(trunk, dtype=np.float64)
    nose_y = np.asarray(nose_y, dtype=np.float64)
    hip = np.asarray(hip, dtype=np.float64)

    trunk_rng = float(np.max(trunk) - np.min(trunk))
    nose_rng = float(np.max(nose_y) - np.min(nose_y))
    trans = float(np.linalg.norm(hip[-1] - hip[0]))

    m = RollMetrics(n, used, trunk_rng, nose_rng, trans)
    return m, {"frames_total": n, "frames_used": used}

def _sev_rng(r: float) -> Severity:
    # forward roll should have big trunk angle range
    if np.isnan(r): return Severity.SEVERE
    if r >= 70: return Severity.NORMAL
    if r >= 55: return Severity.MILD
    if r >= 40: return Severity.MODERATE
    return Severity.SEVERE

def _sev_nose_rng(r: float) -> Severity:
    if np.isnan(r): return Severity.SEVERE
    if r >= 140: return Severity.NORMAL
    if r >= 100: return Severity.MILD
    if r >= 70: return Severity.MODERATE
    return Severity.SEVERE

def _sev_trans(t: float) -> Severity:
    if np.isnan(t): return Severity.SEVERE
    if t <= 120: return Severity.NORMAL
    if t <= 180: return Severity.MILD
    if t <= 260: return Severity.MODERATE
    return Severity.SEVERE

def _max_sev(*ss: Severity) -> Severity:
    order = {Severity.NORMAL:0, Severity.MILD:1, Severity.MODERATE:2, Severity.SEVERE:3}
    best = ss[0]
    for s in ss[1:]:
        if order[s] > order[best]:
            best = s
    return best

def grade_roll(m: RollMetrics) -> Dict[str, Any]:
    s_tr = _sev_rng(m.trunk_angle_range_deg)
    s_no = _sev_nose_rng(m.nose_vertical_range_px)
    s_tx = _sev_trans(m.translation_px)
    sev = _max_sev(s_tr, s_no, s_tx)
    passed = sev in (Severity.NORMAL, Severity.MILD)

    if sev == Severity.NORMAL:
        suggestion = "翻滚特征明显，躯干旋转充分且轨迹集中。"
    elif sev == Severity.MILD:
        suggestion = "轻度偏差：翻滚幅度稍弱或位移偏大，建议收紧身体、降低横向偏移。"
    elif sev == Severity.MODERATE:
        suggestion = "中度偏差：翻滚不充分或轨迹分散，建议分解练习（抱膝滚动）再合成。"
    else:
        suggestion = "重度偏差：动作特征不足或控制较差，建议在保护与辅助下练习基础滚动。"

    return {
        "pass": passed,
        "severity": sev.value,
        "reasons": {
            "trunk_angle_range_deg": m.trunk_angle_range_deg,
            "trunk_level": s_tr.value,
            "nose_vertical_range_px": m.nose_vertical_range_px,
            "nose_level": s_no.value,
            "translation_px": m.translation_px,
            "translation_level": s_tx.value,
        },
        "suggestion": suggestion,
    }
