"""Pipeline steps for the auto evaluation flow."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import get_paths
from ..io.thresholds import load_thresholds
from ..io.video_reader import get_video_meta
from ..pose.yolo_pose import YoloPoseConfig, YoloPoseEstimator
from ..actions.context import EvalContext, ViewAngle
from ..actions.detectors import detect_action_mvp
from ..actions.registry import ACTION_REGISTRY
from ..actions.labels import zh
from ..viz.overlay_pose import render_annotated_video


def step_pose_infer(
    video_path: str | Path,
    model_path: str,
    conf: float = 0.25,
    imgsz: int = 640,
    device: Optional[str] = None,
    vid_stride: int = 1,
):
    """Step 1: pose inference -> keypoints frames + video metadata."""
    estimator = YoloPoseEstimator(
        YoloPoseConfig(model_path=model_path, conf=conf, imgsz=imgsz, device=device)
    )
    results_stream = estimator.predict_video(video_path, vid_stride=vid_stride)
    kpt_frames = estimator.results_to_keypoints(results_stream)
    meta = get_video_meta(video_path)
    return kpt_frames, meta


def step_detect_action(kpt_frames, fps: float = 30.0):
    """Step 2: action detection (ML classifier or rule-based fallback)."""
    action, candidates, feat = detect_action_mvp(kpt_frames, fps=fps)
    return action, candidates, feat


def step_load_thresholds(thresholds_path: Optional[str | Path]):
    """Load thresholds.json (optional). Returns (None, None) if missing."""
    if not thresholds_path:
        return None, None

    p = Path(thresholds_path)
    if not p.exists():
        return None, None

    all_t = load_thresholds(thresholds_path)
    spin_t = all_t.get("spin_in_place")
    meta = {
        "thresholds_path": str(thresholds_path),
        "spin_n_videos": (spin_t or {}).get("n_videos"),
    }
    return spin_t, meta


def step_build_context(
    kpt_frames,
    fps: float,
    view: str = "unknown",
    kpt_conf_thresh: float = 0.20,
) -> EvalContext:
    """Step 3: build EvalContext with FPS, view, and body-height normalisation."""
    view_enum = ViewAngle(view) if view in ("front", "side") else ViewAngle.UNKNOWN
    return EvalContext(
        kpt_frames=kpt_frames,
        fps=fps,
        view=view_enum,
        conf_thresh=kpt_conf_thresh,
    )


def step_evaluate(
    action_id: str,
    ctx: EvalContext,
    thresholds=None,
):
    """Step 4: evaluate by action handler; fallback to spin."""
    if action_id not in ACTION_REGISTRY:
        action_id = "spin_in_place"

    handler = ACTION_REGISTRY[action_id]
    eval_res = handler.evaluator(ctx, thresholds=thresholds)
    metrics_dict = asdict(eval_res["metrics"])
    return action_id, metrics_dict, eval_res["grading"], eval_res.get("debug", {})


def step_render_video(
    video_path: str | Path,
    kpt_frames,
    out_path: str | Path,
    action_id: str,
    grading: Dict[str, Any],
    conf_thresh: float = 0.20,
):
    """Step 5: render annotated video with Chinese overlay."""
    label = (
        f"动作：{zh(action_id)}｜"
        f"偏差：{grading.get('severity')}｜"
        f"达标：{'是' if grading.get('pass') else '否'}"
    )
    return render_annotated_video(
        video_path=video_path,
        kpt_frames=kpt_frames,
        out_path=out_path,
        label=label,
        conf_thresh=conf_thresh,
        grading=grading,
    )
