"""Gradio web UI for vestibular pose evaluation — enhanced with charts and structured display."""
from __future__ import annotations

from pathlib import Path
import json
import time
import gradio as gr

from ..actions.labels import zh, ACTION_LABELS_ZH
from ..pipeline.auto_steps import (
    step_pose_infer,
    step_detect_action,
    step_load_thresholds,
    step_build_context,
    step_evaluate,
    step_render_video,
)
from ..config import get_paths
from ..viz.charts import (
    generate_radar_chart,
    generate_cop_trajectory,
    generate_symmetry_chart,
    generate_result_html,
)

ACTION_IDS = [
    "spin_in_place",
    "jump_in_place",
    "wheelbarrow_walk",
    "run_straight",
    "forward_roll",
    "head_up_prone",
]

ACTION_CHOICES = [(ACTION_LABELS_ZH.get(a, a), a) for a in ACTION_IDS]
VIEW_CHOICES = [("自动", "unknown"), ("正面", "front"), ("侧面", "side")]


def build_app():
    with gr.Blocks(title="儿童统感训练评估") as demo:
        gr.Markdown(
            "# 🧒 儿童统感训练动作评估系统\n"
            "**流程**：上传视频 → Pose 推理 → 自动识别动作 → 量化评分 → 可视化报告\n\n"
            "支持 6 种训练动作：原地旋转 · 原地纵跳 · 小推车 · 直线加速跑 · 前滚翻 · 抬头向上"
        )

        state = gr.State(value=None)

        # ---- Input section ----
        with gr.Group():
            gr.Markdown("### 📹 输入")
            with gr.Row():
                video_input = gr.Video(
                    label="上传视频", sources=["upload"], format="mp4",
                )
                with gr.Column():
                    video_path = gr.Textbox(
                        label="或输入本地路径", placeholder="data/raw/spin.mp4",
                    )
                    model_path = gr.Textbox(
                        label="YOLO Pose 权重", value="yolo11n-pose.pt",
                    )

            with gr.Row():
                thresholds_path = gr.Textbox(
                    label="阈值文件（可选）", value="data/thresholds.json",
                )
                kpt_conf = gr.Slider(
                    0.05, 0.8, value=0.20, step=0.05, label="关键点置信度",
                )
                view_select = gr.Dropdown(
                    choices=VIEW_CHOICES, value="unknown", label="拍摄视角",
                )

        with gr.Row():
            run_btn = gr.Button("🚀 开始识别与评估", variant="primary", size="lg")
            rerun_btn = gr.Button("🔄 用选择的动作重新评分", variant="secondary")

        status = gr.Markdown("**状态**：待开始")

        # ---- Detection info ----
        with gr.Row():
            detected_action_md = gr.Markdown("**自动识别动作：** —")
            candidates_md = gr.Markdown("**候选动作（Top-3）：** —")

        with gr.Row():
            manual_action = gr.Dropdown(
                choices=ACTION_CHOICES, value="spin_in_place",
                label="手动选择动作（识别错了就改这里）", interactive=True,
            )

        # ---- Results section ----
        gr.Markdown("### 📊 评估结果")

        with gr.Row():
            result_html = gr.HTML(label="结构化结果")

        with gr.Row(equal_height=True):
            radar_img = gr.Image(label="能力雷达图", type="filepath")
            cop_img = gr.Image(label="重心轨迹 (COP)", type="filepath")
            symmetry_img = gr.Image(label="对称性分析", type="filepath")

        # ---- Video section ----
        gr.Markdown("### 🎬 视频")
        with gr.Row():
            original_player = gr.Video(label="原始视频")
            annotated_player = gr.Video(label="骨骼标注视频")

        # ---- Raw data section ----
        with gr.Accordion("📋 原始 JSON 报告", open=False):
            report_json = gr.JSON(label="完整报告数据")
            report_file = gr.File(label="下载 report.json")

        # ---- Helpers ----

        def _resolve_video_path(uploaded, typed_path):
            if uploaded is not None:
                if isinstance(uploaded, dict) and "name" in uploaded:
                    return uploaded["name"]
                if isinstance(uploaded, str):
                    return uploaded
            if typed_path:
                return typed_path
            return None

        def _top3_md(candidates):
            if not candidates:
                return "**候选动作（Top-3）：** —"
            lines = []
            for c in candidates[:3]:
                lines.append(f"- {zh(c.action)}（{c.action}）: **{c.score:.3f}**")
            return "**候选动作（Top-3）：**\n" + "\n".join(lines)

        def _build_report(vp, ui_model, action_id, candidates, feat,
                          thresholds_meta, metrics_dict, grading, annotated_video,
                          fps=None, body_height_px=None, view="unknown"):
            return {
                "video": str(vp),
                "model": str(ui_model),
                "video_fps": fps,
                "body_height_px": body_height_px,
                "view": view,
                "action_detected": action_id,
                "action_detected_zh": zh(action_id),
                "action_candidates": [
                    {"action": c.action, "action_zh": zh(c.action),
                     "score": c.score, "details": c.details}
                    for c in (candidates or [])
                ],
                "detector_features": feat,
                "thresholds": thresholds_meta,
                "metrics": metrics_dict,
                "grading": grading,
                "artifacts": {"annotated_video": str(annotated_video)},
            }

        def _generate_charts(action_id, grading, metrics, kpt_frames, kpt_conf_val):
            reasons = grading.get("reasons", {})
            action_zh = zh(action_id)

            radar = generate_radar_chart(reasons, action_zh)
            cop = generate_cop_trajectory(kpt_frames, conf_thresh=kpt_conf_val)
            sym = generate_symmetry_chart(reasons, action_id)
            html = generate_result_html(action_zh, grading, metrics)

            return radar, cop, sym, html

        # ---- Full pipeline ----

        # 11 outputs: status, detected_md, cand_md, original, annotated,
        #             result_html, radar, cop, sym, report_json, report_file, state
        N_OUT = 12
        EMPTY = tuple([None] * N_OUT)

        def run_full(ui_video, ui_path, ui_model, ui_thresholds, ui_kpt_conf, ui_view):
            vp = _resolve_video_path(ui_video, ui_path)
            if not vp:
                yield ("**状态**：请先上传视频或输入路径", *EMPTY[1:])
                return

            yield ("**状态**：已获取视频 ✅", "—", "—", vp,
                   None, None, None, None, None, None, None, None)

            # Step 1: Pose
            yield ("**状态**：🔄 Pose 推理中（可能需要 1-2 分钟）…",
                   "—", "—", vp, None, None, None, None, None, None, None, None)
            kpt_frames, video_meta = step_pose_infer(video_path=vp, model_path=ui_model)

            # Step 2: Detect
            yield ("**状态**：🔍 动作识别中…",
                   "—", "—", vp, None, None, None, None, None, None, None, None)
            action_id, candidates, feat = step_detect_action(kpt_frames)

            detected_md = f"**自动识别动作：** **{zh(action_id)}**（{action_id}）"
            cand_md = _top3_md(candidates)

            # Step 3: Context + evaluate
            yield ("**状态**：📊 评分中…", detected_md, cand_md, vp,
                   None, None, None, None, None, None, None, None)
            thresholds, thresholds_meta = step_load_thresholds(
                ui_thresholds if ui_thresholds else None
            )
            ctx = step_build_context(
                kpt_frames=kpt_frames, fps=video_meta.fps,
                view=ui_view, kpt_conf_thresh=float(ui_kpt_conf),
            )
            action_id_eval, metrics_dict, grading, debug = step_evaluate(
                action_id=action_id, ctx=ctx, thresholds=thresholds,
            )

            # Step 4: Render video
            yield ("**状态**：🎬 生成标注视频中…", detected_md, cand_md, vp,
                   None, None, None, None, None, None, None, None)
            paths = get_paths()
            out_stem = Path(vp).stem
            annotated_video = paths.videos / f"{out_stem}_{action_id_eval}_annotated.mp4"
            step_render_video(
                video_path=vp, kpt_frames=kpt_frames, out_path=annotated_video,
                action_id=action_id_eval, grading=grading,
                conf_thresh=float(ui_kpt_conf),
            )

            # Step 5: Charts
            radar, cop, sym, res_html = _generate_charts(
                action_id_eval, grading, metrics_dict, kpt_frames, float(ui_kpt_conf),
            )

            # Build report
            report = _build_report(
                vp=vp, ui_model=ui_model, action_id=action_id_eval,
                candidates=candidates, feat=feat,
                thresholds_meta=thresholds_meta,
                metrics_dict=metrics_dict, grading=grading,
                annotated_video=annotated_video,
                fps=video_meta.fps, body_height_px=ctx.body_height_px,
                view=ui_view,
            )

            report_path = paths.reports / f"{out_stem}_{action_id_eval}_ui_report.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
            )

            sev = grading.get("severity", "?")
            pass_str = "是" if grading.get("pass") else "否"
            done_txt = (
                f"**状态**：完成 ✅ ｜ 动作：**{zh(action_id_eval)}** ｜"
                f" 偏差：**{sev}** ｜ 达标：**{pass_str}**\n\n"
                "若识别错误：在下拉框选择正确动作 → 点「用选择的动作重新评分」"
            )

            cache = {
                "video": str(vp),
                "model": str(ui_model),
                "kpt_conf": float(ui_kpt_conf),
                "view": ui_view,
                "kpt_frames": kpt_frames,
                "fps": video_meta.fps,
                "candidates": candidates,
                "feat": feat,
                "thresholds": thresholds,
                "thresholds_meta": thresholds_meta,
                "last_detected": action_id,
            }

            yield (done_txt, detected_md, cand_md, vp,
                   str(annotated_video), res_html, radar, cop, sym,
                   report, str(report_path), cache)

        # ---- Rerun with manual action ----

        def rerun_with_selected(selected_action_id, cache_state):
            if not cache_state:
                return (
                    "**状态**：请先点击「开始识别与评估」完成一次推理。",
                    *EMPTY[1:]
                )

            vp = cache_state["video"]
            ui_model = cache_state["model"]
            kpt_frames = cache_state["kpt_frames"]
            fps = cache_state["fps"]
            view = cache_state.get("view", "unknown")
            candidates = cache_state.get("candidates", [])
            feat = cache_state.get("feat", {})
            thresholds = cache_state.get("thresholds")
            thresholds_meta = cache_state.get("thresholds_meta")
            kpt_conf_local = float(cache_state.get("kpt_conf", 0.20))

            detected_md = (
                f"**自动识别动作：** **{zh(cache_state.get('last_detected', 'unknown'))}**"
                f"（{cache_state.get('last_detected', 'unknown')}）"
            )
            cand_md = _top3_md(candidates)

            ctx = step_build_context(
                kpt_frames=kpt_frames, fps=fps,
                view=view, kpt_conf_thresh=kpt_conf_local,
            )
            action_id_eval, metrics_dict, grading, debug = step_evaluate(
                action_id=selected_action_id, ctx=ctx, thresholds=thresholds,
            )

            paths = get_paths()
            out_stem = Path(vp).stem
            ts = int(time.time())
            annotated_video = paths.videos / f"{out_stem}_{action_id_eval}_manual_{ts}.mp4"
            step_render_video(
                video_path=vp, kpt_frames=kpt_frames, out_path=annotated_video,
                action_id=action_id_eval, grading=grading,
                conf_thresh=kpt_conf_local,
            )

            radar, cop, sym, res_html = _generate_charts(
                action_id_eval, grading, metrics_dict, kpt_frames, kpt_conf_local,
            )

            report = _build_report(
                vp=vp, ui_model=ui_model, action_id=action_id_eval,
                candidates=candidates, feat=feat,
                thresholds_meta=thresholds_meta,
                metrics_dict=metrics_dict, grading=grading,
                annotated_video=annotated_video,
                fps=fps, body_height_px=ctx.body_height_px, view=view,
            )
            report_path = paths.reports / f"{out_stem}_{action_id_eval}_manual_{ts}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
            )

            sev = grading.get("severity", "?")
            pass_str = "是" if grading.get("pass") else "否"
            done_txt = (
                f"**状态**：完成（手动重评分）✅ ｜ 动作：**{zh(action_id_eval)}** ｜"
                f" 偏差：**{sev}** ｜ 达标：**{pass_str}**"
            )

            cache_state["last_manual"] = action_id_eval

            return (done_txt, detected_md, cand_md, vp,
                    str(annotated_video), res_html, radar, cop, sym,
                    report, str(report_path), cache_state)

        all_outputs = [
            status, detected_action_md, candidates_md, original_player,
            annotated_player, result_html, radar_img, cop_img, symmetry_img,
            report_json, report_file, state,
        ]

        run_btn.click(
            fn=run_full,
            inputs=[video_input, video_path, model_path, thresholds_path,
                    kpt_conf, view_select],
            outputs=all_outputs,
        )

        rerun_btn.click(
            fn=rerun_with_selected,
            inputs=[manual_action, state],
            outputs=all_outputs,
        )

    return demo
