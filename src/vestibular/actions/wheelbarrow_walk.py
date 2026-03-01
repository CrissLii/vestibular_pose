"""小推车 (Wheelbarrow Walk) evaluator.

Metrics from 评估指标.md:
  - theta_torso_drop: 躯干下垂角 (° from horizontal)
  - sd_torso_lat:     侧向摆动 — trunk X std (normalised)
  - ai_hand:          左右手交替指数 (cross-correlation lag-1)
  - sl_sym:           步长对称性 (ratio, 1.0 = perfect)
  - cc_limb:          对侧协调性 (cross-correlation of R_wrist & L_ankle)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np

from .context import EvalContext, Severity, max_severity
from ..features.trunk_angle import trunk_angle_deg
from ..features.phase_detection import detect_active_phase

NOSE = 0
LSH, RSH = 5, 6
LWR, RWR = 9, 10
LHP, RHP = 11, 12
LANK, RANK = 15, 16


@dataclass
class WheelbarrowMetrics:
    frames_total: int
    frames_used: int
    theta_torso_drop: float     # mean trunk angle from horizontal (°)
    sd_torso_lat: float         # lateral sway (normalised)
    ai_hand: float              # hand alternation index (−1 to 1)
    sl_sym: float               # step-length symmetry (ratio)
    cc_limb: float              # contralateral coordination (−1 to 1)


_NEEDED = [LSH, RSH, LWR, RWR, LHP, RHP, LANK, RANK]


def _extract(ctx: EvalContext):
    sh_mid, hip_mid = [], []
    lwr_y, rwr_y = [], []
    lank_y, rank_y = [], []
    trunk_xy = []  # mid-trunk x for lateral sway

    for f in ctx.kpt_frames:
        xy, cf = f.xy, f.conf
        if max(_NEEDED) >= len(cf):
            continue
        if min(cf[i] for i in _NEEDED) < ctx.conf_thresh:
            continue

        sp = (xy[LSH] + xy[RSH]) / 2.0
        hp = (xy[LHP] + xy[RHP]) / 2.0
        sh_mid.append(sp.astype(np.float64))
        hip_mid.append(hp.astype(np.float64))
        trunk_xy.append(((sp + hp) / 2.0).astype(np.float64))
        lwr_y.append(float(xy[LWR][1]))
        rwr_y.append(float(xy[RWR][1]))
        lank_y.append(float(xy[LANK][1]))
        rank_y.append(float(xy[RANK][1]))

    return {
        "sh_mid": np.asarray(sh_mid), "hip_mid": np.asarray(hip_mid),
        "trunk_xy": np.asarray(trunk_xy),
        "lwr_y": np.asarray(lwr_y), "rwr_y": np.asarray(rwr_y),
        "lank_y": np.asarray(lank_y), "rank_y": np.asarray(rank_y),
    }


def _cross_corr_lag1(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation at lag ±1 (used for alternation index)."""
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = (np.std(a) * np.std(b) * len(a)) + 1e-8
    cc = float(np.correlate(a, b, mode="full")[len(a) - 1])  # lag-0
    return cc / denom


def _step_distances(wrist_y: np.ndarray) -> np.ndarray:
    """Estimate per-step forward distance from wrist Y (proxy for forward movement)."""
    # Detect peaks (hand forward swing) as local minima in Y (hand down = forward)
    from scipy.signal import find_peaks
    # invert for peaks (lower Y = forward in most camera setups)
    peaks, _ = find_peaks(-wrist_y, distance=5, prominence=5.0)
    if len(peaks) < 2:
        return np.array([])
    return np.abs(np.diff(peaks).astype(np.float64))


def compute_wheelbarrow_metrics(ctx: EvalContext) -> tuple[WheelbarrowMetrics, Dict[str, Any]]:
    d = _extract(ctx)
    n_total = len(ctx.kpt_frames)
    n_used = len(d["hip_mid"])

    nan_m = WheelbarrowMetrics(n_total, n_used, *([float("nan")] * 5))
    if n_used < 30:
        return nan_m, {"note": "Not enough valid frames"}

    fps = ctx.fps
    bh = ctx.body_height_px

    # ---- Active phase ----
    phase = detect_active_phase(d["hip_mid"], fps, idle_speed_thresh=6.0)
    s, e = phase.start_idx, phase.end_idx
    if e - s < 20:
        return nan_m, {"note": "Active phase too short"}

    sh = d["sh_mid"][s:e]
    hp = d["hip_mid"][s:e]
    trunk_xy = d["trunk_xy"][s:e]
    lwr = d["lwr_y"][s:e]
    rwr = d["rwr_y"][s:e]
    lank = d["lank_y"][s:e]
    rank = d["rank_y"][s:e]

    # ---- θ_torso_drop: trunk angle from horizontal ----
    # trunk_angle_deg gives angle from vertical; 90° - θ = angle from horizontal
    trunk_angles = np.array([trunk_angle_deg(sh[i], hp[i]) for i in range(len(sh))])
    # In wheelbarrow, trunk is roughly horizontal → angle from vertical ~ 60-90°
    # "drop" = how far below horizontal the trunk sags
    theta_drop = float(90.0 - np.mean(trunk_angles))
    theta_drop = max(0.0, theta_drop)  # clamp to non-negative

    # ---- SD_torso_lat: lateral sway ----
    sd_lat_px = float(np.std(trunk_xy[:, 0]))
    sd_torso_lat = ctx.norm(sd_lat_px)

    # ---- AI_hand: left-right hand alternation ----
    ai_hand = _cross_corr_lag1(lwr, rwr)

    # ---- SL_sym: step-length symmetry ----
    l_steps = _step_distances(lwr)
    r_steps = _step_distances(rwr)
    if len(l_steps) > 0 and len(r_steps) > 0:
        n_min = min(len(l_steps), len(r_steps))
        l_mean = float(np.mean(l_steps[:n_min]))
        r_mean = float(np.mean(r_steps[:n_min]))
        sl_sym = min(l_mean, r_mean) / (max(l_mean, r_mean) + 1e-8)
    else:
        sl_sym = float("nan")

    # ---- CC_limb: contralateral coordination (R_wrist vs L_ankle) ----
    cc_limb = _cross_corr_lag1(rwr, lank)

    metrics = WheelbarrowMetrics(
        frames_total=n_total,
        frames_used=n_used,
        theta_torso_drop=theta_drop,
        sd_torso_lat=sd_torso_lat,
        ai_hand=ai_hand,
        sl_sym=sl_sym,
        cc_limb=cc_limb,
    )
    debug = {
        "active_phase": (s, e),
        "body_height_px": bh,
        "trunk_angle_mean": float(np.mean(trunk_angles)),
    }
    return metrics, debug


# --------------- Grading ---------------

def _sev_drop(val: float) -> Severity:
    """Lower drop angle (closer to horizontal) is better.
    
    theta_torso_drop = 90 - mean_trunk_angle_from_vertical.
    A good wheelbarrow has trunk ~horizontal, so trunk_angle ~80-90° from vertical,
    meaning drop ~ 0-10°.  Larger drop = more sagging.
    """
    if np.isnan(val): return Severity.NORMAL
    if val <= 20: return Severity.NORMAL
    if val <= 35: return Severity.MILD
    if val <= 50: return Severity.MODERATE
    return Severity.SEVERE

def _sev_lat(val: float) -> Severity:
    if np.isnan(val): return Severity.NORMAL
    if val <= 0.04: return Severity.NORMAL
    if val <= 0.08: return Severity.MILD
    if val <= 0.15: return Severity.MODERATE
    return Severity.SEVERE

def _sev_ai(val: float) -> Severity:
    """Hand alternation: negative or low positive correlation indicates alternation.
    High positive (>0.8) means hands moving in sync (less alternation).
    Thresholds relaxed because in 2D both wrists naturally have similar Y."""
    if np.isnan(val): return Severity.NORMAL
    if val <= 0.3: return Severity.NORMAL
    if val <= 0.6: return Severity.MILD
    if val <= 0.85: return Severity.MODERATE
    return Severity.SEVERE

def _sev_sym(val: float) -> Severity:
    if np.isnan(val): return Severity.NORMAL
    if val >= 0.75: return Severity.NORMAL
    if val >= 0.55: return Severity.MILD
    if val >= 0.35: return Severity.MODERATE
    return Severity.SEVERE

def _sev_cc(val: float) -> Severity:
    """Contralateral coordination: negative = good alternation between
    opposite limbs.  Relaxed thresholds for 2D noise."""
    if np.isnan(val): return Severity.NORMAL
    if val <= 0.1: return Severity.NORMAL
    if val <= 0.4: return Severity.MILD
    if val <= 0.7: return Severity.MODERATE
    return Severity.SEVERE


def grade_wheelbarrow(metrics: WheelbarrowMetrics, thresholds: dict | None = None) -> Dict[str, Any]:
    sevs = [
        ("theta_torso_drop", _sev_drop(metrics.theta_torso_drop)),
        ("sd_torso_lat", _sev_lat(metrics.sd_torso_lat)),
        ("ai_hand", _sev_ai(metrics.ai_hand)),
        ("sl_sym", _sev_sym(metrics.sl_sym)),
        ("cc_limb", _sev_cc(metrics.cc_limb)),
    ]

    sev = max_severity(*(s for _, s in sevs))
    passed = sev in (Severity.NORMAL, Severity.MILD)

    reasons = {}
    for name, s in sevs:
        reasons[name] = getattr(metrics, name)
        reasons[f"{name}_level"] = s.value

    suggestions = {
        Severity.NORMAL: "手支撑稳定、躯干保持水平，左右交替节律良好。",
        Severity.MILD: "轻度偏差：注意保持躯干不下垂，左右手交替更有节奏。",
        Severity.MODERATE: "中度偏差：躯干下垂或双侧协调不足，建议缩短距离并加强核心力量。",
        Severity.SEVERE: "重度偏差：姿势控制明显不足，建议在辅助下练习分解动作。",
    }

    return {
        "pass": passed,
        "severity": sev.value,
        "reasons": reasons,
        "suggestion": suggestions[sev],
    }
