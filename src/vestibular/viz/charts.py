"""Chart generation for Gradio UI: radar chart, COP trajectory, symmetry bars."""
from __future__ import annotations

import io
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.font_manager as fm

# Try to use a Chinese-capable font
_ZH_FONTS = ["PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS",
             "Songti SC", "Microsoft YaHei", "SimHei",
             "WenQuanYi Micro Hei", "Noto Sans CJK SC"]
_FONT_FAMILY = None
for _f in _ZH_FONTS:
    if any(_f.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        _FONT_FAMILY = _f
        break

if _FONT_FAMILY:
    plt.rcParams["font.family"] = _FONT_FAMILY
plt.rcParams["axes.unicode_minus"] = False


# ---- Severity to score mapping ----

_SEV_SCORE = {
    "正常": 95, "轻度偏差": 70, "中度偏差": 40, "重度偏差": 15,
    "NORMAL": 5, "MILD": 4, "MODERATE": 2, "SEVERE": 1,
}
_SEV_COLOR = {
    "正常": "#22c55e",       # green
    "轻度偏差": "#eab308",   # yellow
    "中度偏差": "#f97316",   # orange
    "重度偏差": "#ef4444",   # red
    "NORMAL": "#22c55e",
    "MILD": "#eab308",
    "MODERATE": "#f97316",
    "SEVERE": "#ef4444",
}


def severity_to_score(sev_str: str) -> float:
    return _SEV_SCORE.get(sev_str, 3)


def severity_color(sev_str: str) -> str:
    return _SEV_COLOR.get(sev_str, "#6b7280")


# ---- Metric name translations ----

_METRIC_ZH = {
    "omega_avg": "旋转速度", "cv_omega": "节奏稳定性",
    "d_head": "头部稳定", "sd_head_y": "头部垂直",
    "theta_torso": "躯干倾斜", "sd_theta_torso": "躯干稳定",
    "t_recovery": "恢复时间", "cop_post": "停后摆动",
    "h_jump": "跳跃高度", "cv_h": "高度稳定性",
    "theta_knee_land": "落地缓冲", "v_knee": "膝屈速度",
    "theta_torso_air": "空中躯干", "asym_limb": "肢体对称",
    "cv_interval": "节奏稳定",
    "a_max": "最大加速度", "braking_index": "制动力指数",
    "cop_stop": "急停摆动", "t_stabilize": "稳定时间",
    "theta_prep": "停前姿态",
    "theta_torso_drop": "躯干下垂", "sd_torso_lat": "侧向摆动",
    "ai_hand": "手交替性", "sl_sym": "步长对称",
    "cc_limb": "对侧协调",
    "t_roll": "翻滚时间", "js_roll": "动作流畅",
    "theta_yaw": "偏航角度", "q_pose": "起止姿态",
    "hp_reflex": "头部保护",
    "delta_hip_y": "髋部下沉", "sd_hip_x": "侧倾稳定",
    "sd_head_y_static": "头部稳定", "si_load": "承重对称",
    "t_hold": "维持时间",
}


def _metric_label(key: str) -> str:
    return _METRIC_ZH.get(key, key)


# ---- Radar chart ----

def generate_radar_chart(
    reasons: Dict[str, Any],
    action_name_zh: str = "",
) -> str:
    """Generate a radar chart from grading reasons. Returns path to PNG."""
    labels, scores, colors = [], [], []
    for k, v in reasons.items():
        if k.endswith("_level"):
            metric = k.replace("_level", "")
            label = _metric_label(metric)
            score = severity_to_score(str(v))
            labels.append(label)
            scores.append(score)
            colors.append(severity_color(str(v)))

    if not labels:
        return _empty_chart("无可用指标")

    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    scores_plot = scores + [scores[0]]
    angles_plot = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_rlabel_position(0)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=7, color="#888")

    ax.plot(angles_plot, scores_plot, "o-", linewidth=2, color="#3b82f6")
    ax.fill(angles_plot, scores_plot, alpha=0.15, color="#3b82f6")

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9)

    for i, (angle, score, color) in enumerate(zip(angles, scores, colors)):
        ax.plot(angle, score, "o", color=color, markersize=8, zorder=5)

    title = f"能力雷达图"
    if action_name_zh:
        title = f"{action_name_zh} — {title}"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)

    path = _save_fig(fig, "radar")
    return path


# ---- COP trajectory ----

def generate_cop_trajectory(
    kpt_frames,
    conf_thresh: float = 0.20,
) -> str:
    """Generate COP (hip midpoint) trajectory plot. Returns PNG path."""
    LHP, RHP = 11, 12
    xs, ys = [], []
    for f in kpt_frames:
        xy, cf = f.xy, f.conf
        if max(LHP, RHP) >= len(cf):
            continue
        if cf[LHP] < conf_thresh or cf[RHP] < conf_thresh:
            continue
        hip = (xy[LHP] + xy[RHP]) / 2.0
        xs.append(float(hip[0]))
        ys.append(float(hip[1]))

    if len(xs) < 10:
        return _empty_chart("COP数据不足")

    fig, ax = plt.subplots(figsize=(5, 5))
    t = np.linspace(0, 1, len(xs))

    scatter = ax.scatter(xs, ys, c=t, cmap="coolwarm", s=3, alpha=0.7)
    ax.plot(xs[0], ys[0], "go", markersize=10, label="起点", zorder=5)
    ax.plot(xs[-1], ys[-1], "rs", markersize=10, label="终点", zorder=5)

    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    ax.set_title("重心轨迹 (COP)", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=9)
    fig.colorbar(scatter, ax=ax, label="时间进度", shrink=0.7)

    path = _save_fig(fig, "cop")
    return path


# ---- Symmetry bar chart ----

def generate_symmetry_chart(
    reasons: Dict[str, Any],
    action_id: str,
) -> Optional[str]:
    """Generate symmetry comparison chart for applicable actions."""
    pairs = _get_symmetry_pairs(reasons, action_id)
    if not pairs:
        return None

    labels = [p["label"] for p in pairs]
    lefts = [p["left"] for p in pairs]
    rights = [p["right"] for p in pairs]

    fig, ax = plt.subplots(figsize=(5, 3))
    x = np.arange(len(labels))
    w = 0.35

    bars_l = ax.bar(x - w / 2, lefts, w, label="左侧", color="#3b82f6", alpha=0.8)
    bars_r = ax.bar(x + w / 2, rights, w, label="右侧", color="#f97316", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("相对值")
    ax.set_title("左右对称性", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)

    for bar in list(bars_l) + list(bars_r):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    path = _save_fig(fig, "symmetry")
    return path


def _get_symmetry_pairs(reasons: Dict[str, Any], action_id: str) -> List[Dict]:
    """Extract left/right pairs from metrics for symmetry visualization."""
    pairs = []

    if action_id == "jump_in_place":
        asym = reasons.get("asym_limb")
        if asym is not None and not (isinstance(asym, float) and np.isnan(asym)):
            pairs.append({
                "label": "肢体对称",
                "left": 1.0,
                "right": max(0, 1.0 - float(asym)),
            })

    elif action_id == "wheelbarrow_walk":
        sym = reasons.get("sl_sym")
        if sym is not None and not (isinstance(sym, float) and np.isnan(sym)):
            ratio = float(sym)
            pairs.append({
                "label": "步长对称",
                "left": 1.0,
                "right": ratio,
            })
        ai = reasons.get("ai_hand")
        if ai is not None and not (isinstance(ai, float) and np.isnan(ai)):
            pairs.append({
                "label": "手交替",
                "left": 1.0,
                "right": max(0, 1.0 - abs(float(ai))),
            })

    elif action_id == "head_up_prone":
        si = reasons.get("si_load")
        if si is not None and not (isinstance(si, float) and np.isnan(si)):
            pairs.append({
                "label": "承重对称",
                "left": 1.0,
                "right": max(0, 1.0 - float(si)),
            })

    return pairs


# ---- Structured HTML result ----

def generate_result_html(
    action_zh: str,
    grading: Dict[str, Any],
    metrics: Dict[str, Any],
) -> str:
    """Generate a structured HTML result card with severity badge and metric table."""
    severity = grading.get("severity", "?")
    passed = grading.get("pass", False)
    suggestion = grading.get("suggestion", "")
    reasons = grading.get("reasons", {})

    sev_color = severity_color(severity)
    pass_text = "达标" if passed else "未达标"
    pass_color = "#22c55e" if passed else "#ef4444"

    html = f"""
    <div style="font-family: system-ui, -apple-system, sans-serif; max-width: 600px; margin: 0 auto;">
      <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
        <span style="background: {sev_color}; color: white; padding: 6px 16px;
               border-radius: 20px; font-size: 16px; font-weight: bold;">
          {severity}
        </span>
        <span style="background: {pass_color}; color: white; padding: 6px 12px;
               border-radius: 20px; font-size: 14px;">
          {pass_text}
        </span>
        <span style="font-size: 16px; font-weight: 600;">{action_zh}</span>
      </div>

      <div style="background: #fffbeb; border-left: 4px solid {sev_color};
                  padding: 12px 16px; border-radius: 4px; margin-bottom: 16px;">
        <strong>建议：</strong>{suggestion}
      </div>

      <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
        <thead>
          <tr style="background: #f1f5f9;">
            <th style="padding: 8px; text-align: left; border-bottom: 2px solid #e2e8f0;">指标</th>
            <th style="padding: 8px; text-align: right; border-bottom: 2px solid #e2e8f0;">数值</th>
            <th style="padding: 8px; text-align: center; border-bottom: 2px solid #e2e8f0;">等级</th>
          </tr>
        </thead>
        <tbody>
    """

    for k, v in reasons.items():
        if k.endswith("_level"):
            metric_key = k.replace("_level", "")
            metric_val = reasons.get(metric_key, metrics.get(metric_key, "?"))
            level = str(v)
            color = severity_color(level)
            label = _metric_label(metric_key)

            if isinstance(metric_val, float):
                if np.isnan(metric_val):
                    val_str = "N/A"
                else:
                    val_str = f"{metric_val:.3f}"
            elif isinstance(metric_val, bool):
                val_str = "是" if metric_val else "否"
            else:
                val_str = str(metric_val)

            html += f"""
          <tr>
            <td style="padding: 6px 8px; border-bottom: 1px solid #e2e8f0;">{label}</td>
            <td style="padding: 6px 8px; text-align: right; border-bottom: 1px solid #e2e8f0;
                        font-family: monospace;">{val_str}</td>
            <td style="padding: 6px 8px; text-align: center; border-bottom: 1px solid #e2e8f0;">
              <span style="background: {color}; color: white; padding: 2px 8px;
                     border-radius: 10px; font-size: 12px;">{level}</span>
            </td>
          </tr>
            """

    html += """
        </tbody>
      </table>
    </div>
    """
    return html


# ---- Helpers ----

_chart_counter = 0

def _save_fig(fig, prefix: str) -> str:
    global _chart_counter
    _chart_counter += 1
    import tempfile, os
    path = os.path.join(tempfile.gettempdir(), f"vest_{prefix}_{_chart_counter}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _empty_chart(message: str) -> str:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.text(0.5, 0.5, message, ha="center", va="center",
            fontsize=14, color="#999")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return _save_fig(fig, "empty")
