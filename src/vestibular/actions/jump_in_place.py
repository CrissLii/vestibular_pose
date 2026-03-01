"""原地纵跳 (Jump In Place) evaluator.

Metrics from 评估指标.md:
  - H_jump:       平均跳跃高度 (normalised)
  - CV_H:         跳跃高度变异系数
  - θ_knee_land:  落地膝屈角 (°)
  - V_knee:       落地膝屈角变化率 (°/s)
  - θ_torso_air:  空中躯干倾斜角 (°)
  - Asym_limb:    四肢对称性 (normalised)
  - CV_interval:  跳跃间隔变异系数
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np

from .context import EvalContext, Severity, max_severity
from ..features.joint_angles import (
    NOSE, LSH, RSH, LHP, RHP, LKNE, RKNE, LANK, RANK, LWR, RWR,
    left_knee_angle, right_knee_angle,
)
from ..features.trunk_angle import trunk_angle_deg
from ..features.phase_detection import detect_active_phase, detect_cycles, Cycle
from ..features.velocity import smooth_series


@dataclass
class JumpMetrics:
    frames_total: int
    frames_used: int
    n_jumps: int
    h_jump: float            # mean jump height (normalised)
    cv_h: float              # jump height CV
    theta_knee_land: float   # mean landing knee angle (°)
    v_knee: float            # mean knee-angle velocity at landing (°/s)
    theta_torso_air: float   # mean trunk tilt during airborne (°)
    asym_limb: float         # mean limb asymmetry (normalised)
    cv_interval: float       # jump interval CV


_ALL_KPT = [NOSE, LSH, RSH, LHP, RHP, LKNE, RKNE, LANK, RANK, LWR, RWR]


def _extract(ctx: EvalContext):
    """Extract per-frame data arrays from valid frames."""
    hip_y: list[float] = []
    hip_xy: list[np.ndarray] = []
    nose_y: list[float] = []
    trunk_ang: list[float] = []
    l_knee: list[float] = []
    r_knee: list[float] = []
    l_ank_y: list[float] = []
    r_ank_y: list[float] = []
    l_wr_y: list[float] = []
    r_wr_y: list[float] = []
    valid_indices: list[int] = []

    for f in ctx.kpt_frames:
        xy, cf = f.xy, f.conf
        if max(_ALL_KPT) >= len(cf):
            continue
        if min(cf[i] for i in _ALL_KPT) < ctx.conf_thresh:
            continue

        hp = (xy[LHP] + xy[RHP]) / 2.0
        sh = (xy[LSH] + xy[RSH]) / 2.0

        hip_y.append(float(hp[1]))
        hip_xy.append(hp.astype(np.float64))
        nose_y.append(float(xy[NOSE][1]))
        trunk_ang.append(trunk_angle_deg(sh, hp))
        l_knee.append(left_knee_angle(xy))
        r_knee.append(right_knee_angle(xy))
        l_ank_y.append(float(xy[LANK][1]))
        r_ank_y.append(float(xy[RANK][1]))
        l_wr_y.append(float(xy[LWR][1]))
        r_wr_y.append(float(xy[RWR][1]))
        valid_indices.append(f.frame_idx)

    return {
        "hip_y": np.asarray(hip_y),
        "hip_xy": np.asarray(hip_xy),
        "trunk_ang": np.asarray(trunk_ang),
        "l_knee": np.asarray(l_knee),
        "r_knee": np.asarray(r_knee),
        "l_ank_y": np.asarray(l_ank_y),
        "r_ank_y": np.asarray(r_ank_y),
        "l_wr_y": np.asarray(l_wr_y),
        "r_wr_y": np.asarray(r_wr_y),
        "indices": valid_indices,
    }


def compute_jump_metrics(ctx: EvalContext) -> tuple[JumpMetrics, Dict[str, Any]]:
    data = _extract(ctx)
    n_total = len(ctx.kpt_frames)
    n_used = len(data["hip_y"])

    nan_m = JumpMetrics(n_total, n_used, 0, *([float("nan")] * 7))
    if n_used < 30:
        return nan_m, {"note": "Not enough valid frames"}

    fps = ctx.fps
    bh = ctx.body_height_px
    hip_y = data["hip_y"]

    # ---- Trim idle ----
    phase = detect_active_phase(data["hip_xy"], fps, idle_speed_thresh=8.0)
    s, e = phase.start_idx, phase.end_idx

    # ---- Detect jump cycles (invert hip_y: higher jump = lower Y value) ----
    hip_y_act = hip_y[s:e]
    cycles = detect_cycles(hip_y_act, fps, invert=True, min_distance_sec=0.25)

    if len(cycles) < 1:
        return nan_m, {"note": "No jump cycles detected"}

    # ---- Per-jump metrics ----
    heights: list[float] = []
    landing_knee_angles: list[float] = []
    knee_velocities: list[float] = []
    air_trunk_angles: list[float] = []
    limb_asymmetries: list[float] = []
    intervals: list[float] = []

    l_knee_act = data["l_knee"][s:e]
    r_knee_act = data["r_knee"][s:e]
    trunk_act = data["trunk_ang"][s:e]
    l_ank_y_act = data["l_ank_y"][s:e]
    r_ank_y_act = data["r_ank_y"][s:e]
    l_wr_y_act = data["l_wr_y"][s:e]
    r_wr_y_act = data["r_wr_y"][s:e]

    prev_peak: int | None = None
    for cyc in cycles:
        pk = cyc.peak_idx

        # Jump height (in inverted signal, amplitude = height in pixels)
        h_px = cyc.amplitude
        heights.append(ctx.norm(h_px) if not np.isnan(bh) else h_px)

        # Landing knee angle (at cycle end)
        land_idx = min(cyc.end_idx, len(l_knee_act) - 1)
        avg_knee = (l_knee_act[land_idx] + r_knee_act[land_idx]) / 2.0
        landing_knee_angles.append(avg_knee)

        # Knee angle velocity at landing
        if land_idx > 0:
            dk = avg_knee - (l_knee_act[land_idx - 1] + r_knee_act[land_idx - 1]) / 2.0
            knee_velocities.append(abs(dk) * fps)

        # Airborne trunk tilt (around peak)
        air_start = max(0, pk - int(0.05 * fps))
        air_end = min(len(trunk_act), pk + int(0.05 * fps) + 1)
        if air_end > air_start:
            air_trunk_angles.append(float(np.mean(trunk_act[air_start:air_end])))

        # Limb asymmetry at peak
        if pk < len(l_ank_y_act):
            ank_diff = abs(l_ank_y_act[pk] - r_ank_y_act[pk])
            wr_diff = abs(l_wr_y_act[pk] - r_wr_y_act[pk])
            asym_px = (ank_diff + wr_diff) / 2.0
            limb_asymmetries.append(ctx.norm(asym_px) if not np.isnan(bh) else asym_px)

        # Interval between consecutive peaks
        if prev_peak is not None:
            dt = (pk - prev_peak) / fps
            intervals.append(dt)
        prev_peak = pk

    h_arr = np.asarray(heights)
    h_jump = float(np.mean(h_arr))
    cv_h = float(np.std(h_arr) / (np.mean(h_arr) + 1e-8))

    theta_knee = float(np.mean(landing_knee_angles)) if landing_knee_angles else float("nan")
    v_knee = float(np.mean(knee_velocities)) if knee_velocities else float("nan")
    theta_air = float(np.mean(air_trunk_angles)) if air_trunk_angles else float("nan")
    asym = float(np.mean(limb_asymmetries)) if limb_asymmetries else float("nan")

    cv_int = float("nan")
    if len(intervals) >= 2:
        iv = np.asarray(intervals)
        cv_int = float(np.std(iv) / (np.mean(iv) + 1e-8))

    metrics = JumpMetrics(
        frames_total=n_total,
        frames_used=n_used,
        n_jumps=len(cycles),
        h_jump=h_jump,
        cv_h=cv_h,
        theta_knee_land=theta_knee,
        v_knee=v_knee,
        theta_torso_air=theta_air,
        asym_limb=asym,
        cv_interval=cv_int,
    )
    debug = {
        "n_jumps": len(cycles),
        "active_phase": (s, e),
        "heights": heights,
        "intervals": intervals,
        "body_height_px": bh,
    }
    return metrics, debug


# --------------- Grading ---------------

def _sev_h(val: float) -> Severity:
    if np.isnan(val): return Severity.SEVERE
    if val >= 0.15: return Severity.NORMAL
    if val >= 0.10: return Severity.MILD
    if val >= 0.05: return Severity.MODERATE
    return Severity.SEVERE

def _sev_cv_h(val: float) -> Severity:
    if np.isnan(val): return Severity.MILD
    if val <= 0.15: return Severity.NORMAL
    if val <= 0.25: return Severity.MILD
    if val <= 0.40: return Severity.MODERATE
    return Severity.SEVERE

def _sev_knee(val: float) -> Severity:
    """Landing knee angle — moderate flexion (120-160°) is good cushioning."""
    if np.isnan(val): return Severity.MILD
    if 120 <= val <= 160: return Severity.NORMAL
    if 100 <= val <= 170: return Severity.MILD
    if 80 <= val <= 175: return Severity.MODERATE
    return Severity.SEVERE

def _sev_air_trunk(val: float) -> Severity:
    if np.isnan(val): return Severity.MILD
    if val <= 10: return Severity.NORMAL
    if val <= 18: return Severity.MILD
    if val <= 28: return Severity.MODERATE
    return Severity.SEVERE

def _sev_asym(val: float) -> Severity:
    if np.isnan(val): return Severity.MILD
    if val <= 0.03: return Severity.NORMAL
    if val <= 0.06: return Severity.MILD
    if val <= 0.10: return Severity.MODERATE
    return Severity.SEVERE

def _sev_cv_int(val: float) -> Severity:
    if np.isnan(val): return Severity.MILD
    if val <= 0.10: return Severity.NORMAL
    if val <= 0.20: return Severity.MILD
    if val <= 0.35: return Severity.MODERATE
    return Severity.SEVERE


def grade_jump(metrics: JumpMetrics, thresholds: dict | None = None) -> Dict[str, Any]:
    s_h = _sev_h(metrics.h_jump)
    s_cvh = _sev_cv_h(metrics.cv_h)
    s_knee = _sev_knee(metrics.theta_knee_land)
    s_air = _sev_air_trunk(metrics.theta_torso_air)
    s_asym = _sev_asym(metrics.asym_limb)
    s_cvi = _sev_cv_int(metrics.cv_interval)

    sev = max_severity(s_h, s_cvh, s_knee, s_air, s_asym, s_cvi)
    passed = sev in (Severity.NORMAL, Severity.MILD)

    suggestions = {
        Severity.NORMAL: "跳跃高度充足、节奏稳定，落地缓冲与空中姿态良好。",
        Severity.MILD: "轻度偏差：建议关注落地屈膝缓冲或保持跳跃节奏一致性。",
        Severity.MODERATE: "中度偏差：跳跃高度不足或节奏不稳，建议降低频率，先练习稳定的单次起跳-落地。",
        Severity.SEVERE: "重度偏差：动作幅度或控制明显不足，建议在辅助下练习基础弹跳与平衡。",
    }

    return {
        "pass": passed,
        "severity": sev.value,
        "reasons": {
            "h_jump": metrics.h_jump,
            "h_level": s_h.value,
            "cv_h": metrics.cv_h,
            "cv_h_level": s_cvh.value,
            "theta_knee_land": metrics.theta_knee_land,
            "knee_level": s_knee.value,
            "theta_torso_air": metrics.theta_torso_air,
            "air_level": s_air.value,
            "asym_limb": metrics.asym_limb,
            "asym_level": s_asym.value,
            "cv_interval": metrics.cv_interval,
            "cv_int_level": s_cvi.value,
            "n_jumps": metrics.n_jumps,
        },
        "suggestion": suggestions[sev],
    }
