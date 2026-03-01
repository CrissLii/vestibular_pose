"""前滚翻 (Forward Roll) evaluator.

Metrics from 评估指标.md:
  - t_roll:     翻滚完成时间 (s)
  - js_roll:    角速度平滑度 — jerk RMS (lower = smoother)
  - theta_yaw:  偏航角度 — 水平偏移量 (normalised, 2D proxy)
  - q_pose:     起止姿态质量 (0-1, higher = better)
  - hp_reflex:  头部保护反应 (bool — True = abnormal head lift)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np

from .context import EvalContext, Severity, max_severity
from ..features.trunk_angle import trunk_angle_deg
from ..features.velocity import smooth_series
from ..features.phase_detection import detect_active_phase

NOSE = 0
LSH, RSH = 5, 6
LHP, RHP = 11, 12


@dataclass
class RollMetrics:
    frames_total: int
    frames_used: int
    t_roll: float           # roll duration (s)
    js_roll: float          # jerk smoothness (normalised)
    theta_yaw: float        # lateral deviation (normalised)
    q_pose: float           # start/end posture quality (0-1)
    hp_reflex: bool         # abnormal head-protection reflex detected


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


def _detect_roll_phase(trunk: np.ndarray, fps: float) -> tuple[int, int]:
    """Detect the core roll phase where trunk angle changes dramatically.

    Forward roll involves trunk going from ~0° (upright) through large angles
    and back. Detect the segment where angle exceeds a threshold.
    """
    above = trunk > 30.0
    indices = np.where(above)[0]
    if len(indices) < 3:
        return 0, len(trunk)
    # Expand slightly for run-up and recovery
    margin = max(3, int(0.2 * fps))
    start = max(0, int(indices[0]) - margin)
    end = min(len(trunk), int(indices[-1]) + margin + 1)
    return start, end


def compute_roll_metrics(ctx: EvalContext) -> tuple[RollMetrics, Dict[str, Any]]:
    hip, sh, nose, trunk, indices = _extract(ctx)
    n_total = len(ctx.kpt_frames)
    n_used = len(hip)

    nan_m = RollMetrics(n_total, n_used, float("nan"), float("nan"),
                        float("nan"), float("nan"), False)
    if n_used < 20:
        return nan_m, {"note": "Not enough valid frames"}

    fps = ctx.fps
    bh = ctx.body_height_px

    # ---- Active phase (trim idle) ----
    phase = detect_active_phase(hip, fps, idle_speed_thresh=8.0)
    s0, e0 = phase.start_idx, phase.end_idx
    trunk_act = trunk[s0:e0]
    hip_act = hip[s0:e0]
    sh_act = sh[s0:e0]
    nose_act = nose[s0:e0]

    if len(trunk_act) < 15:
        return nan_m, {"note": "Active phase too short"}

    # ---- Roll phase detection ----
    rs, re = _detect_roll_phase(trunk_act, fps)
    roll_trunk = trunk_act[rs:re]
    roll_hip = hip_act[rs:re]
    roll_nose = nose_act[rs:re]
    roll_sh = sh_act[rs:re]

    if len(roll_trunk) < 5:
        return nan_m, {"note": "Roll phase not detected"}

    # ---- T_roll: duration ----
    t_roll = len(roll_trunk) / fps

    # ---- JS_roll: jerk smoothness ----
    trunk_smooth = smooth_series(roll_trunk, window=3)
    omega = np.diff(trunk_smooth) * fps  # angular velocity
    alpha = np.diff(omega) * fps          # angular acceleration
    jerk = np.diff(alpha) * fps           # angular jerk
    js_roll = float(np.sqrt(np.mean(jerk ** 2))) if len(jerk) > 0 else float("nan")
    # Normalise by body height (make it scale-independent in a loose sense)
    if not np.isnan(bh) and bh > 1.0:
        js_roll = js_roll / bh

    # ---- θ_yaw: lateral deviation (2D proxy) ----
    # X displacement between start and end of roll
    x_shift = abs(float(roll_hip[-1, 0] - roll_hip[0, 0]))
    theta_yaw = ctx.norm(x_shift)

    # ---- Q_pose: start/end posture quality ----
    # Good forward roll starts and ends near upright (trunk angle < 15°)
    n_check = max(1, int(0.3 * fps))
    start_angles = trunk_act[:n_check]
    end_angles = trunk_act[-n_check:]

    start_q = float(np.mean(start_angles < 20.0))  # fraction near upright
    end_q = float(np.mean(end_angles < 25.0))
    q_pose = (start_q + end_q) / 2.0

    # ---- HP_reflex: head protection ----
    # During mid-roll (trunk angle > 45°), nose should be tucked (below shoulders)
    mid_mask = roll_trunk > 45.0
    hp_reflex = False
    if np.sum(mid_mask) > 3:
        mid_indices = np.where(mid_mask)[0]
        for mi in mid_indices:
            if mi < len(roll_nose) and mi < len(roll_sh):
                if roll_nose[mi][1] < roll_sh[mi][1] - 20:
                    # Nose significantly above shoulders during mid-roll = head lift
                    hp_reflex = True
                    break

    metrics = RollMetrics(
        frames_total=n_total,
        frames_used=n_used,
        t_roll=t_roll,
        js_roll=js_roll,
        theta_yaw=theta_yaw,
        q_pose=q_pose,
        hp_reflex=hp_reflex,
    )
    debug = {
        "active_phase": (s0, e0),
        "roll_phase": (rs, re),
        "roll_frames": len(roll_trunk),
        "trunk_range": float(np.max(roll_trunk) - np.min(roll_trunk)),
        "body_height_px": bh,
    }
    return metrics, debug


# --------------- Grading ---------------

def _sev_t_roll(val: float) -> Severity:
    """Shorter roll time (but not too fast) indicates good control."""
    if np.isnan(val): return Severity.SEVERE
    if 0.5 <= val <= 2.0: return Severity.NORMAL
    if 0.3 <= val <= 3.0: return Severity.MILD
    if val <= 5.0: return Severity.MODERATE
    return Severity.SEVERE

def _sev_jerk(val: float) -> Severity:
    if np.isnan(val): return Severity.MILD
    if val <= 50: return Severity.NORMAL
    if val <= 120: return Severity.MILD
    if val <= 250: return Severity.MODERATE
    return Severity.SEVERE

def _sev_yaw(val: float) -> Severity:
    if np.isnan(val): return Severity.MILD
    if val <= 0.05: return Severity.NORMAL
    if val <= 0.10: return Severity.MILD
    if val <= 0.20: return Severity.MODERATE
    return Severity.SEVERE

def _sev_q_pose(val: float) -> Severity:
    if np.isnan(val): return Severity.MILD
    if val >= 0.7: return Severity.NORMAL
    if val >= 0.5: return Severity.MILD
    if val >= 0.3: return Severity.MODERATE
    return Severity.SEVERE

def _sev_hp(is_reflex: bool) -> Severity:
    return Severity.MILD if is_reflex else Severity.NORMAL


def grade_roll(metrics: RollMetrics, thresholds: dict | None = None) -> Dict[str, Any]:
    sevs = [
        ("t_roll", _sev_t_roll(metrics.t_roll)),
        ("js_roll", _sev_jerk(metrics.js_roll)),
        ("theta_yaw", _sev_yaw(metrics.theta_yaw)),
        ("q_pose", _sev_q_pose(metrics.q_pose)),
        ("hp_reflex", _sev_hp(metrics.hp_reflex)),
    ]

    sev = max_severity(*(s for _, s in sevs))
    passed = sev in (Severity.NORMAL, Severity.MILD)

    reasons = {}
    for name, s in sevs:
        val = getattr(metrics, name)
        reasons[name] = val
        reasons[f"{name}_level"] = s.value

    suggestions = {
        Severity.NORMAL: "翻滚流畅、轨迹集中，起止姿态稳定。",
        Severity.MILD: "轻度偏差：注意收紧身体减少横向偏移，保持翻滚节奏。",
        Severity.MODERATE: "中度偏差：翻滚不流畅或偏航明显，建议分解练习（抱膝滚动）。",
        Severity.SEVERE: "重度偏差：动作控制不足，建议在保护下练习基础滚动。",
    }

    return {
        "pass": passed,
        "severity": sev.value,
        "reasons": reasons,
        "suggestion": suggestions[sev],
    }
