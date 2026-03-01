"""Full auto pipeline: pose → detect → evaluate → render → report."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import get_paths
from ..actions.labels import zh
from .auto_steps import (
    step_pose_infer,
    step_detect_action,
    step_load_thresholds,
    step_build_context,
    step_evaluate,
    step_render_video,
)


def run_auto_on_video(
    video_path: str | Path,
    model_path: str,
    thresholds_path: Optional[str | Path] = None,
    out_stem: Optional[str] = None,
    conf: float = 0.25,
    imgsz: int = 640,
    device: Optional[str] = None,
    kpt_conf_thresh: float = 0.20,
    view: str = "unknown",
    project_root: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Full run (non-stream): pose -> detect -> evaluate -> render -> report."""
    paths = get_paths(project_root)
    video_path = Path(video_path)
    out_stem = out_stem or video_path.stem

    # 1) Pose inference + video metadata
    kpt_frames, video_meta = step_pose_infer(
        video_path=video_path, model_path=model_path,
        conf=conf, imgsz=imgsz, device=device,
    )

    # 2) Action detection
    action_id, candidates, feat = step_detect_action(kpt_frames)

    # 3) Thresholds
    thresholds, thresholds_meta = step_load_thresholds(thresholds_path)

    # 4) Build evaluation context
    ctx = step_build_context(
        kpt_frames=kpt_frames,
        fps=video_meta.fps,
        view=view,
        kpt_conf_thresh=kpt_conf_thresh,
    )

    # 5) Evaluate
    action_id, metrics_dict, grading, debug = step_evaluate(
        action_id=action_id, ctx=ctx, thresholds=thresholds,
    )

    # 6) Render annotated video
    annotated_video = paths.videos / f"{out_stem}_{action_id}_annotated.mp4"
    step_render_video(
        video_path=video_path,
        kpt_frames=kpt_frames,
        out_path=annotated_video,
        action_id=action_id,
        grading=grading,
        conf_thresh=kpt_conf_thresh,
    )

    # 7) Report
    report = {
        "video": str(video_path),
        "model": str(model_path),
        "video_fps": video_meta.fps,
        "video_resolution": f"{video_meta.width}x{video_meta.height}",
        "body_height_px": ctx.body_height_px,
        "view": view,
        "action_detected": action_id,
        "action_detected_zh": zh(action_id),
        "action_candidates": [
            {"action": c.action, "action_zh": zh(c.action),
             "score": c.score, "details": c.details}
            for c in candidates
        ],
        "detector_features": feat,
        "thresholds": thresholds_meta,
        "metrics": metrics_dict,
        "grading": grading,
        "artifacts": {"annotated_video": str(annotated_video)},
    }

    report_path = paths.reports / f"{out_stem}_{action_id}_auto_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {"report_path": str(report_path), **report}
