from __future__ import annotations
from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Dict, Any

import numpy as np

from ..pose.keypoints import KeypointsFrame

NOSE = 0
LANK, RANK = 15, 16

class Severity(str, Enum):
    NORMAL = "正常"
    MILD = "轻度偏差"
    MODERATE = "中度偏差"
    SEVERE = "重度偏差"

@dataclass
class JumpMetrics:
    frames_total: int
    frames_used: int
    jump_amplitude_px: float         # nose vertical range proxy
    landing_instability_std_px: float # std of ankle y after landing proxy

def compute_jump_metrics(kpt_frames: List[KeypointsFrame], conf_thresh: float = 0.2) -> tuple[JumpMetrics, Dict[str, Any]]:
    nose_y = []
    ank_y = []
    used = 0
    for f in kpt_frames:
        xy, cf = f.xy, f.conf
        if max(NOSE, LANK, RANK) >= len(cf):
            continue
        if min(cf[NOSE], cf[LANK], cf[RANK]) < conf_thresh:
            continue
        nose_y.append(float(xy[NOSE][1]))
        ank_y.append(float((xy[LANK][1] + xy[RANK][1]) / 2.0))
        used += 1

    n = len(kpt_frames)
    if used < 20:
        m = JumpMetrics(n, used, float("nan"), float("nan"))
        return m, {"frames_total": n, "frames_used": used}

    nose_y = np.asarray(nose_y, dtype=np.float64)
    ank_y = np.asarray(ank_y, dtype=np.float64)

    amp = float(np.max(nose_y) - np.min(nose_y))  # y range; jump causes big swing

    # landing stability: take last 30% frames, measure ankle y std
    start = int(len(ank_y) * 0.70)
    landing_std = float(np.std(ank_y[start:]))

    m = JumpMetrics(n, used, amp, landing_std)
    return m, {"frames_total": n, "frames_used": used, "landing_start_idx": start}

def _sev_from_amp(amp: float) -> Severity:
    # smaller amplitude => weaker jump / incomplete
    if np.isnan(amp): return Severity.SEVERE
    if amp >= 90: return Severity.NORMAL
    if amp >= 60: return Severity.MILD
    if amp >= 35: return Severity.MODERATE
    return Severity.SEVERE

def _sev_from_landing(std: float) -> Severity:
    if np.isnan(std): return Severity.SEVERE
    if std <= 6: return Severity.NORMAL
    if std <= 10: return Severity.MILD
    if std <= 16: return Severity.MODERATE
    return Severity.SEVERE

def _max_sev(a: Severity, b: Severity) -> Severity:
    order = {Severity.NORMAL: 0, Severity.MILD: 1, Severity.MODERATE: 2, Severity.SEVERE: 3}
    return a if order[a] >= order[b] else b

def grade_jump(metrics: JumpMetrics) -> Dict[str, Any]:
    s1 = _sev_from_amp(metrics.jump_amplitude_px)
    s2 = _sev_from_landing(metrics.landing_instability_std_px)
    sev = _max_sev(s1, s2)
    passed = sev in (Severity.NORMAL, Severity.MILD)

    if sev == Severity.NORMAL:
        suggestion = "跳跃幅度与落地稳定性良好，保持节奏与落地缓冲。"
    elif sev == Severity.MILD:
        suggestion = "轻度偏差：建议关注落地缓冲（屈膝）或提高起跳一致性。"
    elif sev == Severity.MODERATE:
        suggestion = "中度偏差：建议降低频率，先练习稳定落地与起跳协调。"
    else:
        suggestion = "重度偏差：动作幅度不足或落地不稳，建议在辅助下练习基础下肢控制。"

    return {
        "pass": passed,
        "severity": sev.value,
        "reasons": {
            "jump_amplitude_px": metrics.jump_amplitude_px,
            "jump_level": s1.value,
            "landing_instability_std_px": metrics.landing_instability_std_px,
            "landing_level": s2.value,
        },
        "suggestion": suggestion,
    }
