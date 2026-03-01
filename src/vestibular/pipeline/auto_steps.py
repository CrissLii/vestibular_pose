from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..config import get_paths
from ..io.thresholds import load_thresholds
from ..pose.yolo_pose import YoloPoseConfig, YoloPoseEstimator
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
):
    """Step 1: pose inference -> keypoints frames"""
    estimator = YoloPoseEstimator(
        YoloPoseConfig(model_path=model_path, conf=conf, imgsz=imgsz, device=device)
    )
    results_stream = estimator.predict_video(video_path)
    kpt_frames = estimator.results_to_keypoints(results_stream)
    return kpt_frames


def step_detect_action(kpt_frames):
    """Step 2: MVP action detection by rules"""
    action, candidates, feat = detect_action_mvp(kpt_frames)
    return action, candidates, feat


def step_load_thresholds(thresholds_path: Optional[str | Path]):
    """Load thresholds.json (optional)"""
    if not thresholds_path:
        return None, None

    all_t = load_thresholds(thresholds_path)
    spin_t = all_t.get("spin_in_place")
    meta = {
        "thresholds_path": str(thresholds_path),
        "spin_n_videos": (spin_t or {}).get("n_videos"),
        "spin_method_std": ((spin_t or {}).get("std_deg") or {}).get("method"),
        "spin_method_mean": ((spin_t or {}).get("mean_deg") or {}).get("method"),
    }
    return spin_t, meta


def step_evaluate(action_id: str, kpt_frames, thresholds_spin=None, kpt_conf_thresh: float = 0.20):
    """Step 3: evaluate by action handler; fallback to spin"""
    if action_id not in ACTION_REGISTRY:
        action_id = "spin_in_place"

    handler = ACTION_REGISTRY[action_id]
    eval_res = handler.evaluator(
        kpt_frames,
        thresholds_spin=thresholds_spin,
        kpt_conf_thresh=kpt_conf_thresh,
    )
    # normalize metrics to dict for report
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
    """Step 4: render annotated video with Chinese overlay"""
    label = f"动作：{zh(action_id)}｜偏差：{grading.get('severity')}｜达标：{'是' if grading.get('pass') else '否'}"
    return render_annotated_video(
        video_path=video_path,
        kpt_frames=kpt_frames,
        out_path=out_path,
        label=label,
        conf_thresh=conf_thresh,
    )
