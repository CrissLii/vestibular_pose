"""直线加速跑 (Run Straight) evaluator.

Metrics from 评估指标.md:
  - a_max:       最大加速度 (body-height / s²)
  - BI:          制动力指数  max_decel / max_accel
  - COP_stop:    急停后 3 秒重心轨迹长度 (normalised)
  - T_stabilize: 速度降到阈值以下所需时间 (s)
  - θ_prep:      停止前躯干倾角 (°)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, List

import numpy as np

from .context import EvalContext, Severity, max_severity
from ..pose.keypoints import KeypointsFrame
from ..features.trunk_angle import trunk_angle_deg
from ..features.velocity import speed_2d, acceleration_1d, smooth_series
from ..features.phase_detection import detect_active_phase, detect_stop_frame

LSH, RSH = 5, 6
LHP, RHP = 11, 12


@dataclass
class RunMetrics:
    frames_total: int
    frames_used: int
    a_max: float            # peak acceleration (body-height / s²)
    braking_index: float    # |max decel| / max accel
    cop_stop: float         # post-stop 3s trajectory length (normalised)
    t_stabilize: float      # seconds until speed < threshold after stop
    theta_prep: float       # trunk angle in the 0.5s before stop (°)


def _extract_series(ctx: EvalContext):
    """Extract hip & shoulder midpoint series from valid frames."""
    hip_xy: list[np.ndarray] = []
    sh_xy: list[np.ndarray] = []
    frame_indices: list[int] = []

    for f in ctx.kpt_frames:
        xy, cf = f.xy, f.conf
        needed = [LSH, RSH, LHP, RHP]
        if max(needed) >= len(cf):
            continue
        if min(cf[i] for i in needed) < ctx.conf_thresh:
            continue
        hip_xy.append(((xy[LHP] + xy[RHP]) / 2.0).astype(np.float64))
        sh_xy.append(((xy[LSH] + xy[RSH]) / 2.0).astype(np.float64))
        frame_indices.append(f.frame_idx)

    return np.asarray(hip_xy), np.asarray(sh_xy), frame_indices


def compute_run_metrics(ctx: EvalContext) -> tuple[RunMetrics, Dict[str, Any]]:
    hip, sh, indices = _extract_series(ctx)
    n_total = len(ctx.kpt_frames)
    n_used = len(hip)

    nan_metrics = RunMetrics(n_total, n_used, *([float("nan")] * 5))
    if n_used < 30:
        return nan_metrics, {"note": "Not enough valid frames"}

    fps = ctx.fps
    bh = ctx.body_height_px

    # ---- Active phase ----
    phase = detect_active_phase(hip, fps, idle_speed_thresh=10.0)
    hip_act = hip[phase.start_idx:phase.end_idx]
    sh_act = sh[phase.start_idx:phase.end_idx]

    if len(hip_act) < 20:
        return nan_metrics, {"note": "Active phase too short"}

    # ---- Horizontal speed (X direction, primary running axis) ----
    vx = np.diff(hip_act[:, 0]) * fps  # px/s
    vx_smooth = smooth_series(vx, window=5)

    # ---- Acceleration ----
    ax = np.diff(vx_smooth) * fps  # px/s²
    a_max_px = float(np.max(np.abs(ax))) if len(ax) > 0 else 0.0
    a_max = ctx.norm(a_max_px) if not np.isnan(bh) else a_max_px

    # ---- Braking index ----
    accel_vals = ax[ax > 0]
    decel_vals = ax[ax < 0]
    if len(accel_vals) > 0 and len(decel_vals) > 0:
        max_accel = float(np.max(accel_vals))
        max_decel = float(np.max(np.abs(decel_vals)))
        bi = max_decel / (max_accel + 1e-8)
    else:
        bi = float("nan")

    # ---- Stop detection & post-stop metrics ----
    stop_frame = detect_stop_frame(hip_act, fps, speed_thresh_px=8.0)
    cop_stop = float("nan")
    t_stabilize = float("nan")
    theta_prep = float("nan")

    if stop_frame is not None:
        # COP_stop: trajectory length in 3 seconds after stop
        post_frames = int(3.0 * fps)
        post_end = min(len(hip_act), stop_frame + post_frames)
        if post_end - stop_frame > 5:
            from ..features.velocity import trajectory_length
            traj_px = trajectory_length(hip_act[stop_frame:post_end])
            cop_stop = ctx.norm(traj_px) if not np.isnan(bh) else traj_px

        # T_stabilize: time from peak speed to speed < threshold
        spd = speed_2d(hip_act, fps)
        peak_spd_idx = int(np.argmax(spd))
        stable_thresh = 0.02 * bh * fps if not np.isnan(bh) else 15.0
        stable_indices = np.where(spd[peak_spd_idx:] < stable_thresh)[0]
        if len(stable_indices) > 0:
            t_stabilize = float(stable_indices[0]) / fps

        # θ_prep: trunk angle 0.5s before stop
        prep_frames = max(1, int(0.5 * fps))
        prep_start = max(0, stop_frame - prep_frames)
        prep_angles = []
        for i in range(prep_start, stop_frame):
            if i < len(sh_act) and i < len(hip_act):
                prep_angles.append(trunk_angle_deg(sh_act[i], hip_act[i]))
        theta_prep = float(np.mean(prep_angles)) if prep_angles else float("nan")

    metrics = RunMetrics(
        frames_total=n_total,
        frames_used=n_used,
        a_max=a_max,
        braking_index=bi,
        cop_stop=cop_stop,
        t_stabilize=t_stabilize,
        theta_prep=theta_prep,
    )
    debug = {
        "active_start": phase.start_idx,
        "active_end": phase.end_idx,
        "stop_frame": stop_frame,
        "body_height_px": bh,
    }
    return metrics, debug


# --------------- Grading ---------------

def _sev_a_max(val: float) -> Severity:
    if np.isnan(val):
        return Severity.SEVERE
    if val >= 3.0:
        return Severity.NORMAL
    if val >= 2.0:
        return Severity.MILD
    if val >= 1.0:
        return Severity.MODERATE
    return Severity.SEVERE


def _sev_bi(val: float) -> Severity:
    if np.isnan(val):
        return Severity.SEVERE
    if 0.6 <= val <= 1.5:
        return Severity.NORMAL
    if 0.4 <= val <= 2.0:
        return Severity.MILD
    if 0.2 <= val <= 3.0:
        return Severity.MODERATE
    return Severity.SEVERE


def _sev_cop_stop(val: float) -> Severity:
    if np.isnan(val):
        return Severity.MILD
    if val <= 0.3:
        return Severity.NORMAL
    if val <= 0.6:
        return Severity.MILD
    if val <= 1.0:
        return Severity.MODERATE
    return Severity.SEVERE


def _sev_t_stabilize(val: float) -> Severity:
    if np.isnan(val):
        return Severity.MILD
    if val <= 1.5:
        return Severity.NORMAL
    if val <= 3.0:
        return Severity.MILD
    if val <= 5.0:
        return Severity.MODERATE
    return Severity.SEVERE


def _sev_theta_prep(val: float) -> Severity:
    if np.isnan(val):
        return Severity.MILD
    if val <= 15.0:
        return Severity.NORMAL
    if val <= 25.0:
        return Severity.MILD
    if val <= 35.0:
        return Severity.MODERATE
    return Severity.SEVERE


def grade_run(metrics: RunMetrics, thresholds: dict | None = None) -> Dict[str, Any]:
    s_amax = _sev_a_max(metrics.a_max)
    s_bi = _sev_bi(metrics.braking_index)
    s_cop = _sev_cop_stop(metrics.cop_stop)
    s_stab = _sev_t_stabilize(metrics.t_stabilize)
    s_prep = _sev_theta_prep(metrics.theta_prep)

    sev = max_severity(s_amax, s_bi, s_cop, s_stab, s_prep)
    passed = sev in (Severity.NORMAL, Severity.MILD)

    suggestions = {
        Severity.NORMAL: "加速和制动特征良好，急停后重心稳定，姿态控制到位。",
        Severity.MILD: "轻度偏差：建议加强启动爆发力或急停后的身体控制。",
        Severity.MODERATE: "中度偏差：加速/制动不均衡或急停后摆动明显，建议分解练习加速与急停。",
        Severity.SEVERE: "重度偏差：运动控制不足，建议降低速度，在辅助下练习基础跑停协调。",
    }

    return {
        "pass": passed,
        "severity": sev.value,
        "reasons": {
            "a_max": metrics.a_max,
            "a_max_level": s_amax.value,
            "braking_index": metrics.braking_index,
            "bi_level": s_bi.value,
            "cop_stop": metrics.cop_stop,
            "cop_level": s_cop.value,
            "t_stabilize": metrics.t_stabilize,
            "t_stab_level": s_stab.value,
            "theta_prep": metrics.theta_prep,
            "prep_level": s_prep.value,
        },
        "suggestion": suggestions[sev],
    }
