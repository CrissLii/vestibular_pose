"""超人飞 / 抬头向上 (Prone Superman) evaluator.

Metrics from 评估指标.md:
  - delta_hip_y:      髋部 Y 下沉量 (normalised, linear drift)
  - sd_hip_x:         侧倾稳定性 — hip X std (normalised)
  - sd_head_y_static: 头部稳定性 — nose Y std (normalised)
  - si_load:          承重对称性 — 左右肩/髋高度差 (normalised)
  - t_hold:           姿势维持时间 (s)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np

from .context import EvalContext, Severity, max_severity
from ..features.trunk_angle import trunk_angle_deg

NOSE = 0
LSH, RSH = 5, 6
LHP, RHP = 11, 12


@dataclass
class HeadUpMetrics:
    frames_total: int
    frames_used: int
    delta_hip_y: float          # hip Y drift (normalised, positive = sinking)
    sd_hip_x: float             # lateral sway (normalised)
    sd_head_y_static: float     # head vertical stability (normalised)
    si_load: float              # load symmetry index (0 = perfect, 1 = fully asymmetric)
    t_hold: float               # posture hold duration (s)


def _extract(ctx: EvalContext):
    hip_l, hip_r, sh_l, sh_r, nose_xy, trunk = [], [], [], [], [], []
    hip_mid, sh_mid = [], []
    for f in ctx.kpt_frames:
        xy, cf = f.xy, f.conf
        needed = [NOSE, LSH, RSH, LHP, RHP]
        if max(needed) >= len(cf):
            continue
        if min(cf[i] for i in needed) < ctx.conf_thresh:
            continue
        hip_l.append(xy[LHP].astype(np.float64))
        hip_r.append(xy[RHP].astype(np.float64))
        sh_l.append(xy[LSH].astype(np.float64))
        sh_r.append(xy[RSH].astype(np.float64))
        hp = (xy[LHP] + xy[RHP]) / 2.0
        sp = (xy[LSH] + xy[RSH]) / 2.0
        hip_mid.append(hp.astype(np.float64))
        sh_mid.append(sp.astype(np.float64))
        nose_xy.append(xy[NOSE].astype(np.float64))
        trunk.append(trunk_angle_deg(sp, hp))

    return {
        "hip_l": np.asarray(hip_l), "hip_r": np.asarray(hip_r),
        "sh_l": np.asarray(sh_l), "sh_r": np.asarray(sh_r),
        "hip_mid": np.asarray(hip_mid), "sh_mid": np.asarray(sh_mid),
        "nose": np.asarray(nose_xy), "trunk": np.asarray(trunk),
    }


def compute_headup_metrics(ctx: EvalContext) -> tuple[HeadUpMetrics, Dict[str, Any]]:
    d = _extract(ctx)
    n_total = len(ctx.kpt_frames)
    n_used = len(d["hip_mid"])

    nan_m = HeadUpMetrics(n_total, n_used, *([float("nan")] * 5))
    if n_used < 20:
        return nan_m, {"note": "Not enough valid frames"}

    fps = ctx.fps
    bh = ctx.body_height_px
    hip_mid = d["hip_mid"]
    nose = d["nose"]
    trunk = d["trunk"]

    # ---- Posture hold detection ----
    # Superman/head-up pose: trunk tilted away from vertical.
    # In 2D, prone trunk angle depends on camera view and body geometry.
    # Use a lenient threshold to capture partial holds.
    hold_mask = trunk > 30.0
    if np.sum(hold_mask) < 10:
        hold_mask = trunk > 20.0

    # Find longest contiguous hold segment
    best_start, best_len, cur_start, cur_len = 0, 0, 0, 0
    for i, m in enumerate(hold_mask):
        if m:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_len = 0

    t_hold = best_len / fps if fps > 0 else 0.0

    # Restrict analysis to the hold segment
    hs, he = best_start, best_start + best_len
    if best_len < 10:
        hs, he = 0, n_used  # fallback: use all

    hip_hold = hip_mid[hs:he]
    nose_hold = nose[hs:he]

    # ---- Δhip_y: vertical drift (linear regression slope) ----
    hip_y = hip_hold[:, 1]
    if len(hip_y) > 5:
        t = np.arange(len(hip_y)) / fps
        slope = float(np.polyfit(t, hip_y, 1)[0])  # px/s (positive = sinking)
        delta_hip_y_px = slope * (len(hip_y) / fps)  # total drift in px over hold
        delta_hip_y = ctx.norm(abs(delta_hip_y_px))
    else:
        delta_hip_y = float("nan")

    # ---- SD_hip_x: lateral sway ----
    sd_hip_x_px = float(np.std(hip_hold[:, 0]))
    sd_hip_x = ctx.norm(sd_hip_x_px)

    # ---- SD_head_y: head vertical stability ----
    sd_head_y_px = float(np.std(nose_hold[:, 1]))
    sd_head_y_static = ctx.norm(sd_head_y_px)

    # ---- SI_load: load symmetry ----
    # Approximate by |left_hip_y - right_hip_y| + |left_sh_y - right_sh_y|
    hip_l_hold = d["hip_l"][hs:he]
    hip_r_hold = d["hip_r"][hs:he]
    sh_l_hold = d["sh_l"][hs:he]
    sh_r_hold = d["sh_r"][hs:he]

    hip_asym = np.abs(hip_l_hold[:, 1] - hip_r_hold[:, 1])
    sh_asym = np.abs(sh_l_hold[:, 1] - sh_r_hold[:, 1])
    si_load_px = float(np.mean(hip_asym + sh_asym))
    si_load = ctx.norm(si_load_px)

    metrics = HeadUpMetrics(
        frames_total=n_total,
        frames_used=n_used,
        delta_hip_y=delta_hip_y,
        sd_hip_x=sd_hip_x,
        sd_head_y_static=sd_head_y_static,
        si_load=si_load,
        t_hold=t_hold,
    )
    debug = {
        "hold_segment": (hs, he),
        "hold_frames": best_len,
        "body_height_px": bh,
    }
    return metrics, debug


# --------------- Grading ---------------

def _sev_drift(val: float) -> Severity:
    if np.isnan(val): return Severity.NORMAL
    if val <= 0.03: return Severity.NORMAL
    if val <= 0.06: return Severity.MILD
    if val <= 0.12: return Severity.MODERATE
    return Severity.SEVERE

def _sev_sway(val: float) -> Severity:
    if np.isnan(val): return Severity.NORMAL
    if val <= 0.03: return Severity.NORMAL
    if val <= 0.06: return Severity.MILD
    if val <= 0.10: return Severity.MODERATE
    return Severity.SEVERE

def _sev_head(val: float) -> Severity:
    if np.isnan(val): return Severity.NORMAL
    if val <= 0.04: return Severity.NORMAL
    if val <= 0.08: return Severity.MILD
    if val <= 0.15: return Severity.MODERATE
    return Severity.SEVERE

def _sev_si(val: float) -> Severity:
    if np.isnan(val): return Severity.NORMAL
    if val <= 0.02: return Severity.NORMAL
    if val <= 0.04: return Severity.MILD
    if val <= 0.08: return Severity.MODERATE
    return Severity.SEVERE

def _sev_hold(val: float) -> Severity:
    if np.isnan(val): return Severity.SEVERE
    if val >= 8.0: return Severity.NORMAL
    if val >= 5.0: return Severity.MILD
    if val >= 3.0: return Severity.MODERATE
    return Severity.SEVERE


def grade_headup(metrics: HeadUpMetrics, thresholds: dict | None = None) -> Dict[str, Any]:
    sevs = [
        ("delta_hip_y", _sev_drift(metrics.delta_hip_y)),
        ("sd_hip_x", _sev_sway(metrics.sd_hip_x)),
        ("sd_head_y_static", _sev_head(metrics.sd_head_y_static)),
        ("si_load", _sev_si(metrics.si_load)),
        ("t_hold", _sev_hold(metrics.t_hold)),
    ]

    sev = max_severity(*(s for _, s in sevs))
    passed = sev in (Severity.NORMAL, Severity.MILD)

    reasons = {}
    for name, s in sevs:
        reasons[name] = getattr(metrics, name)
        reasons[f"{name}_level"] = s.value

    suggestions = {
        Severity.NORMAL: "俯卧姿势稳定，抬头保持时间充足，左右承重对称。",
        Severity.MILD: "轻度偏差：建议延长抬头保持时间或减少身体侧向晃动。",
        Severity.MODERATE: "中度偏差：核心耐力或头部控制不足，建议缩短每组时长、分组练习。",
        Severity.SEVERE: "重度偏差：姿势维持困难，建议在辅助下练习颈背基础力量和抗重力能力。",
    }

    return {
        "pass": passed,
        "severity": sev.value,
        "reasons": reasons,
        "suggestion": suggestions[sev],
    }
