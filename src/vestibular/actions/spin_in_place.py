"""原地旋转 (Spin In Place) evaluator.

Metrics from 评估指标.md:
  - omega_avg:     平均旋转角速度 (°/s, 2D trunk-vector proxy)
  - cv_omega:      角速度变异系数
  - d_head:        头部水平位移 — nose X 轨迹长度 (normalised)
  - sd_head_y:     头部垂直稳定性 — std(nose_y) (normalised)
  - theta_torso:   躯干倾斜角均值 (°)
  - sd_theta_torso: 躯干倾斜角标准差 (°)
  - t_recovery:    姿势恢复时间 (s)
  - cop_post:      停止后 5s 重心摆动 (normalised)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np

from .context import EvalContext, Severity, max_severity
from ..features.trunk_angle import trunk_angle_deg
from ..features.velocity import speed_2d, trajectory_length, smooth_series
from ..features.phase_detection import detect_active_phase, detect_stop_frame

NOSE = 0
LSH, RSH = 5, 6
LHP, RHP = 11, 12


@dataclass
class SpinMetrics:
    frames_total: int
    frames_used: int
    omega_avg: float          # mean angular velocity (°/s)
    cv_omega: float           # CV of angular velocity
    d_head: float             # head horizontal trajectory (normalised)
    sd_head_y: float          # head vertical stability (normalised)
    theta_torso: float        # mean trunk tilt (°)
    sd_theta_torso: float     # trunk tilt std (°)
    t_recovery: float         # recovery time (s)
    cop_post: float           # post-stop CoG sway (normalised)


def _extract(ctx: EvalContext):
    hip, sh, nose, trunk, indices = [], [], [], [], []
    for f in ctx.kpt_frames:
        xy, cf = f.xy, f.conf
        needed = [NOSE, LSH, RSH, LHP, RHP]
        if max(needed) >= len(cf):
            continue
        if min(cf[i] for i in needed) < ctx.conf_thresh:
            continue
        hp = (xy[LHP] + xy[RHP]) / 2.0
        sp = (xy[LSH] + xy[RSH]) / 2.0
        hip.append(hp.astype(np.float64))
        sh.append(sp.astype(np.float64))
        nose.append(xy[NOSE].astype(np.float64))
        trunk.append(trunk_angle_deg(sp, hp))
        indices.append(f.frame_idx)
    return (
        np.asarray(hip), np.asarray(sh), np.asarray(nose),
        np.asarray(trunk), indices,
    )


def compute_spin_metrics(ctx: EvalContext) -> tuple[SpinMetrics, Dict[str, Any]]:
    hip, sh, nose, trunk, indices = _extract(ctx)
    n_total = len(ctx.kpt_frames)
    n_used = len(hip)

    nan_m = SpinMetrics(n_total, n_used, *([float("nan")] * 8))
    if n_used < 30:
        return nan_m, {"note": "Not enough valid frames"}

    fps = ctx.fps
    bh = ctx.body_height_px

    # ---- Active phase ----
    phase = detect_active_phase(hip, fps, idle_speed_thresh=6.0)
    s, e = phase.start_idx, phase.end_idx
    hip_act = hip[s:e]
    sh_act = sh[s:e]
    nose_act = nose[s:e]
    trunk_act = trunk[s:e]

    if len(hip_act) < 20:
        return nan_m, {"note": "Active phase too short"}

    # ---- Angular velocity (2D proxy) ----
    # Use trunk vector angle change between frames as rotation proxy
    trunk_smooth = smooth_series(trunk_act, window=3)
    d_angle = np.abs(np.diff(trunk_smooth))
    omega = d_angle * fps  # °/s
    omega_avg = float(np.mean(omega))
    cv_omega = float(np.std(omega) / (omega_avg + 1e-8))

    # ---- Head stability ----
    nose_x = nose_act[:, 0]
    nose_y = nose_act[:, 1]
    d_head_px = trajectory_length(nose_act[:, :1].reshape(-1, 1).repeat(2, axis=1))
    # Simpler: total absolute X displacement
    d_head_px = float(np.sum(np.abs(np.diff(nose_x))))
    d_head = ctx.norm(d_head_px)
    sd_head_y_px = float(np.std(nose_y))
    sd_head_y = ctx.norm(sd_head_y_px)

    # ---- Trunk ----
    theta_torso = float(np.mean(trunk_act))
    sd_theta_torso = float(np.std(trunk_act))

    # ---- Recovery (post-spin) ----
    # Use frames AFTER the active phase for recovery analysis
    post_start = e
    post_hip = hip[post_start:] if post_start < len(hip) else np.empty((0, 2))

    t_recovery = float("nan")
    cop_post = float("nan")

    if len(post_hip) > 10:
        spd = speed_2d(post_hip, fps)
        spd_smooth = smooth_series(spd, window=5)
        # Recovery = time until speed < threshold
        thresh = 0.01 * bh * fps if not np.isnan(bh) else 5.0
        below = np.where(spd_smooth < thresh)[0]
        if len(below) > 0:
            t_recovery = float(below[0]) / fps

        # CoP sway: trajectory in first 5 seconds post-spin
        post_5s = min(len(post_hip), int(5.0 * fps))
        if post_5s > 5:
            cop_px = trajectory_length(post_hip[:post_5s])
            cop_post = ctx.norm(cop_px)
    else:
        # No post-spin data — try detect stop within active phase
        stop = detect_stop_frame(hip_act, fps, speed_thresh_px=6.0)
        if stop is not None and stop < len(hip_act) - 10:
            rest = hip_act[stop:]
            spd = speed_2d(rest, fps)
            thresh = 0.01 * bh * fps if not np.isnan(bh) else 5.0
            below = np.where(smooth_series(spd, 5) < thresh)[0]
            if len(below) > 0:
                t_recovery = float(below[0]) / fps
            post_5s = min(len(rest), int(5.0 * fps))
            if post_5s > 5:
                cop_post = ctx.norm(trajectory_length(rest[:post_5s]))

    metrics = SpinMetrics(
        frames_total=n_total,
        frames_used=n_used,
        omega_avg=omega_avg,
        cv_omega=cv_omega,
        d_head=d_head,
        sd_head_y=sd_head_y,
        theta_torso=theta_torso,
        sd_theta_torso=sd_theta_torso,
        t_recovery=t_recovery,
        cop_post=cop_post,
    )
    debug = {
        "active_phase": (s, e),
        "body_height_px": bh,
        "omega_series_len": len(omega),
    }
    return metrics, debug


# --------------- Grading ---------------

def _sev_omega(val: float) -> Severity:
    """Stable rotation speed is normal; too erratic or too slow is concerning."""
    if np.isnan(val): return Severity.SEVERE
    if val >= 20: return Severity.NORMAL
    if val >= 10: return Severity.MILD
    if val >= 5: return Severity.MODERATE
    return Severity.SEVERE

def _sev_cv_omega(val: float) -> Severity:
    if np.isnan(val): return Severity.MILD
    if val <= 0.30: return Severity.NORMAL
    if val <= 0.50: return Severity.MILD
    if val <= 0.80: return Severity.MODERATE
    return Severity.SEVERE

def _sev_d_head(val: float) -> Severity:
    if np.isnan(val): return Severity.MILD
    if val <= 0.5: return Severity.NORMAL
    if val <= 1.0: return Severity.MILD
    if val <= 2.0: return Severity.MODERATE
    return Severity.SEVERE

def _sev_sd_head_y(val: float) -> Severity:
    if np.isnan(val): return Severity.MILD
    if val <= 0.02: return Severity.NORMAL
    if val <= 0.04: return Severity.MILD
    if val <= 0.07: return Severity.MODERATE
    return Severity.SEVERE

def _sev_theta(val: float) -> Severity:
    if np.isnan(val): return Severity.SEVERE
    if val <= 8: return Severity.NORMAL
    if val <= 15: return Severity.MILD
    if val <= 25: return Severity.MODERATE
    return Severity.SEVERE

def _sev_sd_theta(val: float) -> Severity:
    if np.isnan(val): return Severity.SEVERE
    if val <= 3.0: return Severity.NORMAL
    if val <= 5.0: return Severity.MILD
    if val <= 8.0: return Severity.MODERATE
    return Severity.SEVERE

def _sev_recovery(val: float) -> Severity:
    if np.isnan(val): return Severity.MILD
    if val <= 2.0: return Severity.NORMAL
    if val <= 4.0: return Severity.MILD
    if val <= 7.0: return Severity.MODERATE
    return Severity.SEVERE

def _sev_cop_post(val: float) -> Severity:
    if np.isnan(val): return Severity.MILD
    if val <= 0.3: return Severity.NORMAL
    if val <= 0.6: return Severity.MILD
    if val <= 1.2: return Severity.MODERATE
    return Severity.SEVERE


def grade_spin(metrics: SpinMetrics, thresholds: dict | None = None) -> Dict[str, Any]:
    sevs = [
        ("omega_avg", _sev_omega(metrics.omega_avg)),
        ("cv_omega", _sev_cv_omega(metrics.cv_omega)),
        ("d_head", _sev_d_head(metrics.d_head)),
        ("sd_head_y", _sev_sd_head_y(metrics.sd_head_y)),
        ("theta_torso", _sev_theta(metrics.theta_torso)),
        ("sd_theta_torso", _sev_sd_theta(metrics.sd_theta_torso)),
        ("t_recovery", _sev_recovery(metrics.t_recovery)),
        ("cop_post", _sev_cop_post(metrics.cop_post)),
    ]

    sev = max_severity(*(s for _, s in sevs))
    passed = sev in (Severity.NORMAL, Severity.MILD)

    reasons = {}
    for name, s in sevs:
        reasons[name] = getattr(metrics, name)
        reasons[f"{name}_level"] = s.value

    suggestions = {
        Severity.NORMAL: "旋转节奏稳定，躯干与头部控制良好，停止后恢复快。",
        Severity.MILD: "轻度偏差：建议放慢旋转速度，注意头部稳定和躯干竖直。",
        Severity.MODERATE: "中度偏差：旋转控制不足或恢复偏慢，建议减少圈数并加强基础平衡训练。",
        Severity.SEVERE: "重度偏差：动作控制或恢复能力明显不足，建议暂停高强度旋转，先做基础前庭训练。",
    }

    return {
        "pass": passed,
        "severity": sev.value,
        "reasons": reasons,
        "suggestion": suggestions[sev],
    }
