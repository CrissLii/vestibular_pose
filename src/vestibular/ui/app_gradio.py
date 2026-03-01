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
    step_evaluate,
    step_render_video,
)
from ..config import get_paths


# 6个动作（内部ID）——用于下拉框 value
ACTION_IDS = [
    "spin_in_place",
    "jump_in_place",
    "wheelbarrow_walk",
    "run_straight",
    "forward_roll",
    "head_up_prone",
]

# 下拉框 choices：显示中文(label) + 内部ID(value)
ACTION_CHOICES = [(ACTION_LABELS_ZH.get(a, a), a) for a in ACTION_IDS]


def build_app():
    with gr.Blocks(title="Vestibular Pose Evaluator (UX upgrade)") as demo:
        gr.Markdown(
            "# 儿童统感训练评估\n"
            "流程：上传/输入视频 → **Pose推理(一次)** → 自动识别动作 → 评分 → 生成骨骼标注视频\n\n"
            "如果动作识别错了：选择正确动作 → **不重复跑Pose** → 重新评分 + 重新生成标注视频"
        )

        # 用来缓存关键点与中间结果（不重复跑 YOLO）
        state = gr.State(value=None)

        with gr.Row():
            video_input = gr.Video(label="输入视频（上传）", sources=["upload"], format="mp4")
            video_path = gr.Textbox(label="或输入本地视频路径（可选）", placeholder="data/raw/spin.mp4")

        with gr.Row():
            model_path = gr.Textbox(label="YOLO Pose 权重路径", value="yolo11n-pose.pt")
            thresholds_path = gr.Textbox(label="thresholds.json（可选）", value="data/thresholds.json")
            kpt_conf = gr.Slider(0.05, 0.8, value=0.20, step=0.05, label="关键点置信度阈值")

        with gr.Row():
            run_btn = gr.Button("开始识别与评估", variant="primary")
            rerun_btn = gr.Button("用选择的动作重新评分", variant="secondary")

        status = gr.Markdown("状态：待开始")

        with gr.Row():
            detected_action_md = gr.Markdown("**自动识别动作：** -")
            candidates_md = gr.Markdown("**候选动作（Top-3）：** -")

        with gr.Row():
            manual_action = gr.Dropdown(
                choices=ACTION_CHOICES,
                value="spin_in_place",
                label="手动选择动作（识别错了就改这里）",
                interactive=True,
            )

        with gr.Row():
            original_player = gr.Video(label="原始视频（回放）")
            annotated_player = gr.Video(label="标注视频（回放）")

        with gr.Row():
            report_json = gr.JSON(label="Report（结构化结果）")
            report_file = gr.File(label="下载 report.json")

        def _resolve_video_path(uploaded, typed_path):
            if uploaded is not None:
                if isinstance(uploaded, dict) and "name" in uploaded:
                    return uploaded["name"]
                if isinstance(uploaded, str):
                    return uploaded
            if typed_path:
                return typed_path
            return None

        def _top3_candidates_md(candidates):
            if not candidates:
                return "**候选动作（Top-3）：** -"
            top = candidates[:3]
            lines = []
            for c in top:
                a = c.action
                lines.append(f"- {zh(a)}（{a}）: **{c.score:.3f}**  细节: {c.details}")
            return "**候选动作（Top-3）：**\n" + "\n".join(lines)

        def _build_report(vp, ui_model, action_id, candidates, feat, thresholds_meta, metrics_dict, grading, annotated_video, debug):
            return {
                "video": str(vp),
                "model": str(ui_model),
                "action_detected": action_id,
                "action_detected_zh": zh(action_id),
                "action_candidates": [
                    {"action": c.action, "action_zh": zh(c.action), "score": c.score, "details": c.details}
                    for c in (candidates or [])
                ],
                "detector_features": feat,
                "thresholds": thresholds_meta,
                "metrics": metrics_dict,
                "grading": grading,
                "artifacts": {"annotated_video": str(annotated_video)},
                "debug": {
                    "frames_total": debug.get("frames_total"),
                    "frames_used": debug.get("frames_used"),
                },
            }

        def run_full(ui_video, ui_path, ui_model, ui_thresholds, ui_kpt_conf):
            """
            全流程：Pose(一次) -> 自动识别 -> 评分 -> 渲染 -> report
            并把关键点/中间结果缓存到 state，供“重新评分”复用
            """
            vp = _resolve_video_path(ui_video, ui_path)
            if not vp:
                yield (
                    "状态：请先上传视频或输入路径",
                    "**自动识别动作：** -",
                    "**候选动作（Top-3）：** -",
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                return

            # 先显示原视频
            yield (
                "状态：已获取视频 ✅",
                "**自动识别动作：** -",
                "**候选动作（Top-3）：** -",
                vp,
                None,
                None,
                None,
                None,
            )

            # Step 1: Pose
            yield (
                "状态：Pose 推理中…",
                "**自动识别动作：** -",
                "**候选动作（Top-3）：** -",
                vp,
                None,
                None,
                None,
                None,
            )
            kpt_frames = step_pose_infer(video_path=vp, model_path=ui_model)

            # Step 2: Detect
            yield (
                "状态：动作识别中…",
                "**自动识别动作：** -",
                "**候选动作（Top-3）：** -",
                vp,
                None,
                None,
                None,
                None,
            )
            action_id, candidates, feat = step_detect_action(kpt_frames)

            detected_md = f"**自动识别动作：** **{zh(action_id)}**（{action_id}）"
            cand_md = _top3_candidates_md(candidates)

            # Step 3: Score
            yield (
                "状态：评分中…",
                detected_md,
                cand_md,
                vp,
                None,
                None,
                None,
                None,
            )
            thresholds_spin, thresholds_meta = step_load_thresholds(ui_thresholds if ui_thresholds else None)
            action_id_eval, metrics_dict, grading, debug = step_evaluate(
                action_id=action_id,
                kpt_frames=kpt_frames,
                thresholds_spin=thresholds_spin,
                kpt_conf_thresh=float(ui_kpt_conf),
            )

            # Step 4: Render
            yield (
                "状态：生成可视化中…",
                detected_md,
                cand_md,
                vp,
                None,
                None,
                None,
                None,
            )
            paths = get_paths()
            out_stem = Path(vp).stem
            annotated_video = paths.videos / f"{out_stem}_{action_id_eval}_annotated.mp4"
            step_render_video(
                video_path=vp,
                kpt_frames=kpt_frames,
                out_path=annotated_video,
                action_id=action_id_eval,
                grading=grading,
                conf_thresh=float(ui_kpt_conf),
            )

            # Report
            report = _build_report(
                vp=vp,
                ui_model=ui_model,
                action_id=action_id_eval,
                candidates=candidates,
                feat=feat,
                thresholds_meta=thresholds_meta,
                metrics_dict=metrics_dict,
                grading=grading,
                annotated_video=annotated_video,
                debug=debug,
            )

            report_path = paths.reports / f"{out_stem}_{action_id_eval}_ui_report.json"
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

            done_txt = (
                f"状态：完成 ✅ ｜动作：**{zh(action_id_eval)}** ｜偏差：**{grading.get('severity')}** ｜达标：**{'是' if grading.get('pass') else '否'}**"
                "\n\n若识别错了：在下拉框选择正确动作 → 点「用选择的动作重新评分」。"
            )

            # 缓存 state（关键点+阈值+候选+特征）
            cache = {
                "video": str(vp),
                "model": str(ui_model),
                "kpt_conf": float(ui_kpt_conf),
                "kpt_frames": kpt_frames,
                "candidates": candidates,
                "feat": feat,
                "thresholds_spin": thresholds_spin,
                "thresholds_meta": thresholds_meta,
                "last_detected": action_id,
            }

            yield (
                done_txt,
                detected_md,
                cand_md,
                vp,
                str(annotated_video),
                report,
                str(report_path),
                cache,
            )

        def rerun_with_selected(selected_action_id, cache_state):
            """
            不重复跑 pose：
            - 直接用缓存 keypoints 做 evaluate
            - 重新生成标注视频（角标更新：动作/达标/偏差）
            - 更新 report
            """
            if not cache_state:
                return (
                    "状态：请先点击「开始识别与评估」完成一次推理后，才能重新评分。",
                    "**自动识别动作：** -",
                    "**候选动作（Top-3）：** -",
                    None,
                    None,
                    None,
                    None,
                    None,
                )

            vp = cache_state["video"]
            ui_model = cache_state["model"]
            kpt_frames = cache_state["kpt_frames"]
            candidates = cache_state.get("candidates", [])
            feat = cache_state.get("feat", {})
            thresholds_spin = cache_state.get("thresholds_spin", None)
            thresholds_meta = cache_state.get("thresholds_meta", None)
            kpt_conf_local = float(cache_state.get("kpt_conf", 0.20))

            detected_md = f"**自动识别动作：** **{zh(cache_state.get('last_detected','unknown'))}**（{cache_state.get('last_detected','unknown')}）"
            cand_md = _top3_candidates_md(candidates)

            # 评分（强制动作）
            action_id_eval, metrics_dict, grading, debug = step_evaluate(
                action_id=selected_action_id,
                kpt_frames=kpt_frames,
                thresholds_spin=thresholds_spin,
                kpt_conf_thresh=kpt_conf_local,
            )

            # 重新渲染标注视频（避免覆盖旧的，带时间戳）
            paths = get_paths()
            out_stem = Path(vp).stem
            ts = int(time.time())
            annotated_video = paths.videos / f"{out_stem}_{action_id_eval}_manual_{ts}.mp4"
            step_render_video(
                video_path=vp,
                kpt_frames=kpt_frames,
                out_path=annotated_video,
                action_id=action_id_eval,
                grading=grading,
                conf_thresh=kpt_conf_local,
            )

            report = _build_report(
                vp=vp,
                ui_model=ui_model,
                action_id=action_id_eval,
                candidates=candidates,
                feat=feat,
                thresholds_meta=thresholds_meta,
                metrics_dict=metrics_dict,
                grading=grading,
                annotated_video=annotated_video,
                debug=debug,
            )
            report_path = paths.reports / f"{out_stem}_{action_id_eval}_manual_{ts}.json"
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

            done_txt = (
                f"状态：完成（手动重评分）✅ ｜动作：**{zh(action_id_eval)}** ｜偏差：**{grading.get('severity')}** ｜达标：**{'是' if grading.get('pass') else '否'}**"
            )

            # 更新缓存：最后选择的动作
            cache_state["last_manual"] = action_id_eval

            return (
                done_txt,
                detected_md,
                cand_md,
                vp,
                str(annotated_video),
                report,
                str(report_path),
                cache_state,
            )

        # 绑定：全流程（generator -> 实时进度）
        run_btn.click(
            fn=run_full,
            inputs=[video_input, video_path, model_path, thresholds_path, kpt_conf],
            outputs=[status, detected_action_md, candidates_md, original_player, annotated_player, report_json, report_file, state],
        )

        # 绑定：手动重评分（不跑pose）
        rerun_btn.click(
            fn=rerun_with_selected,
            inputs=[manual_action, state],
            outputs=[status, detected_action_md, candidates_md, original_player, annotated_player, report_json, report_file, state],
        )

    return demo
