from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any

import numpy as np
from ..pose.keypoints import KeypointsFrame

LWR, RWR = 9, 10
LHP, RHP = 11, 12
LANK, RANK = 15, 16

class Severity(str, Enum):
    NORMAL = "正常"
    MILD = "轻度偏差"
    MODERATE = "中度偏差"
    SEVERE = "重度偏差"

@dataclass
class WheelbarrowMetrics:
    frames_total: int
    frames_used: int
    wrist_below_hip_ratio: float
    hip_dx_px: float
    body_sag_ratio: float    # ankle much lower than hip (legs sagging) ratio

def compute_wheelbarrow_metrics(kpt_frames: List[KeypointsFrame], conf_thresh: float = 0.2) -> tuple[WheelbarrowMetrics, Dict[str, Any]]:
    hip = []
    wrist = []
    ankle = []
    used = 0

    for f in kpt_frames:
        xy, cf = f.xy, f.conf
        need = [LWR, RWR, LHP, RHP, LANK, RANK]
        if max(need) >= len(cf):
            continue
        if min(cf[i] for i in need) < conf_thresh:
            continue

        hip_xy = (xy[LHP] + xy[RHP]) / 2.0
        wr_xy = (xy[LWR] + xy[RWR]) / 2.0
        ank_xy = (xy[LANK] + xy[RANK]) / 2.0

        hip.append(hip_xy.astype(np.float64))
        wrist.append(wr_xy.astype(np.float64))
        ankle.append(ank_xy.astype(np.float64))
        used += 1

    n = len(kpt_frames)
    if used < 20:
        m = WheelbarrowMetrics(n, used, float("nan"), float("nan"), float("nan"))
        return m, {"frames_total": n, "frames_used": used}

    hip = np.asarray(hip)
    wrist = np.asarray(wrist)
    ankle = np.asarray(ankle)

    wrist_below_hip_ratio = float(np.mean((wrist[:, 1] > hip[:, 1]).astype(np.float64)))
    hip_dx = float(np.std(hip[:, 0]))

    # sag: ankles much lower (y bigger) than hips -> legs dropping
    sag = (ankle[:, 1] - hip[:, 1])  # positive if ankle lower
    body_sag_ratio = float(np.mean((sag > 80.0).astype(np.float64)))

    m = WheelbarrowMetrics(n, used, wrist_below_hip_ratio, hip_dx, body_sag_ratio)
    return m, {"frames_total": n, "frames_used": used}

def _sev_ratio(x: float, good: float, ok: float, bad: float) -> Severity:
    if np.isnan(x): return Severity.SEVERE
    if x >= good: return Severity.NORMAL
    if x >= ok: return Severity.MILD
    if x >= bad: return Severity.MODERATE
    return Severity.SEVERE

def _sev_dx(dx: float) -> Severity:
    if np.isnan(dx): return Severity.SEVERE
    if dx >= 60: return Severity.NORMAL
    if dx >= 40: return Severity.MILD
    if dx >= 20: return Severity.MODERATE
    return Severity.SEVERE

def _max_sev(*ss: Severity) -> Severity:
    order = {Severity.NORMAL:0, Severity.MILD:1, Severity.MODERATE:2, Severity.SEVERE:3}
    best = ss[0]
    for s in ss[1:]:
        if order[s] > order[best]:
            best = s
    return best

def grade_wheelbarrow(m: WheelbarrowMetrics) -> Dict[str, Any]:
    s_hand = _sev_ratio(m.wrist_below_hip_ratio, good=0.75, ok=0.55, bad=0.35)
    s_move = _sev_dx(m.hip_dx_px)
    # sag ratio: lower is better, so invert
    if np.isnan(m.body_sag_ratio):
        s_sag = Severity.SEVERE
    elif m.body_sag_ratio <= 0.20:
        s_sag = Severity.NORMAL
    elif m.body_sag_ratio <= 0.40:
        s_sag = Severity.MILD
    elif m.body_sag_ratio <= 0.60:
        s_sag = Severity.MODERATE
    else:
        s_sag = Severity.SEVERE

    sev = _max_sev(s_hand, s_move, s_sag)
    passed = sev in (Severity.NORMAL, Severity.MILD)

    if sev == Severity.NORMAL:
        suggestion = "手支撑比例高，移动连续，身体保持较好。"
    elif sev == Severity.MILD:
        suggestion = "轻度偏差：注意手支撑稳定性，避免腿部下垂。"
    elif sev == Severity.MODERATE:
        suggestion = "中度偏差：支撑不够或身体下垂明显，建议降低距离并加强核心控制。"
    else:
        suggestion = "重度偏差：姿势控制不足，建议在辅助下练习分解动作。"

    return {
        "pass": passed,
        "severity": sev.value,
        "reasons": {
            "wrist_below_hip_ratio": m.wrist_below_hip_ratio,
            "hand_support_level": s_hand.value,
            "hip_dx_px": m.hip_dx_px,
            "move_level": s_move.value,
            "body_sag_ratio": m.body_sag_ratio,
            "sag_level": s_sag.value,
        },
        "suggestion": suggestion,
    }
