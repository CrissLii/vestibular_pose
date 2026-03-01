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
class HeadUpMetrics:
    frames_total: int
    frames_used: int
    trunk_horizontal_ratio: float     # trunk angle > 60deg vs vertical
    head_lift_ratio: float            # nose above shoulders ratio
    translation_px: float             # hip displacement

def compute_headup_metrics(kpt_frames: List[KeypointsFrame], conf_thresh: float = 0.2) -> tuple[HeadUpMetrics, Dict[str, Any]]:
    trunk = []
    head_lift = []
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
        ang = trunk_angle_deg(sh, hp)
        trunk.append(ang)
        head_lift.append(float(xy[NOSE][1] < sh[1]))  # nose higher (smaller y)
        hip.append(hp.astype(np.float64))
        used += 1

    n = len(kpt_frames)
    if used < 20:
        m = HeadUpMetrics(n, used, float("nan"), float("nan"), float("nan"))
        return m, {"frames_total": n, "frames_used": used}

    trunk = np.asarray(trunk, dtype=np.float64)
    head_lift = np.asarray(head_lift, dtype=np.float64)
    hip = np.asarray(hip, dtype=np.float64)

    trunk_horizontal_ratio = float(np.mean((trunk > 60.0).astype(np.float64)))
    head_lift_ratio = float(np.mean(head_lift))
    trans = float(np.linalg.norm(hip[-1] - hip[0]))

    m = HeadUpMetrics(n, used, trunk_horizontal_ratio, head_lift_ratio, trans)
    return m, {"frames_total": n, "frames_used": used}

def _sev_ratio(x: float, good: float, ok: float, bad: float) -> Severity:
    if np.isnan(x): return Severity.SEVERE
    if x >= good: return Severity.NORMAL
    if x >= ok: return Severity.MILD
    if x >= bad: return Severity.MODERATE
    return Severity.SEVERE

def _sev_trans(t: float) -> Severity:
    if np.isnan(t): return Severity.SEVERE
    if t <= 40: return Severity.NORMAL
    if t <= 70: return Severity.MILD
    if t <= 110: return Severity.MODERATE
    return Severity.SEVERE

def _max_sev(*ss: Severity) -> Severity:
    order = {Severity.NORMAL:0, Severity.MILD:1, Severity.MODERATE:2, Severity.SEVERE:3}
    best = ss[0]
    for s in ss[1:]:
        if order[s] > order[best]:
            best = s
    return best

def grade_headup(m: HeadUpMetrics) -> Dict[str, Any]:
    s_tr = _sev_ratio(m.trunk_horizontal_ratio, good=0.70, ok=0.50, bad=0.30)
    s_hd = _sev_ratio(m.head_lift_ratio, good=0.70, ok=0.50, bad=0.30)
    s_tx = _sev_trans(m.translation_px)
    sev = _max_sev(s_tr, s_hd, s_tx)
    passed = sev in (Severity.NORMAL, Severity.MILD)

    if sev == Severity.NORMAL:
        suggestion = "俯卧姿势稳定，抬头明显且保持时间充足。"
    elif sev == Severity.MILD:
        suggestion = "轻度偏差：建议延长抬头保持时间，并减少身体滑动。"
    elif sev == Severity.MODERATE:
        suggestion = "中度偏差：抬头不充分或姿势不稳，建议缩短时长、分组练习。"
    else:
        suggestion = "重度偏差：动作特征不足或稳定性差，建议在辅助下练习颈背基础力量。"

    return {
        "pass": passed,
        "severity": sev.value,
        "reasons": {
            "trunk_horizontal_ratio": m.trunk_horizontal_ratio,
            "trunk_level": s_tr.value,
            "head_lift_ratio": m.head_lift_ratio,
            "head_level": s_hd.value,
            "translation_px": m.translation_px,
            "translation_level": s_tx.value,
        },
        "suggestion": suggestion,
    }
