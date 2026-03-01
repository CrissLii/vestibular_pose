"""Single-action spin evaluation pipeline (legacy entry point)."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..io.thresholds import load_thresholds
from ..io.video_reader import get_video_meta
from ..actions.context import EvalContext
from ..actions.spin_in_place import compute_spin_metrics, grade_spin
from ..config import get_paths
from ..pose.yolo_pose import YoloPoseConfig, YoloPoseEstimator
from ..viz.plot_series import plot_series


def save_keypoints_npz(out_path: Path, kpt_frames) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xy = [f.xy for f in kpt_frames]
    conf = [f.conf for f in kpt_frames]
    frame_idx = [f.frame_idx for f in kpt_frames]
    np.savez_compressed(out_path, xy=xy, conf=conf, frame_idx=frame_idx)
    return out_path


def run_spin_on_video(
    video_path: str | Path,
    model_path: str,
    out_stem: Optional[str] = None,
    project_root: Optional[str | Path] = None,
    conf: float = 0.25,
    imgsz: int = 640,
    device: Optional[str] = None,
    kpt_conf_thresh: float = 0.20,
    thresholds_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    paths = get_paths(project_root)
    video_path = Path(video_path)
    out_stem = out_stem or video_path.stem

    # 1) Pose
    estimator = YoloPoseEstimator(
        YoloPoseConfig(model_path=model_path, conf=conf, imgsz=imgsz, device=device)
    )
    results_stream = estimator.predict_video(video_path)
    kpt_frames = estimator.results_to_keypoints(results_stream)

    # 2) Save keypoints
    kpt_out = paths.keypoints / f"{out_stem}.npz"
    save_keypoints_npz(kpt_out, kpt_frames)

    # 3) Build context & compute metrics
    video_meta = get_video_meta(video_path)
    ctx = EvalContext(
        kpt_frames=kpt_frames,
        fps=video_meta.fps,
        conf_thresh=kpt_conf_thresh,
    )
    metrics, debug = compute_spin_metrics(ctx)

    thresholds = None
    thresholds_meta = None
    if thresholds_path:
        all_t = load_thresholds(thresholds_path)
        thresholds = all_t.get("spin_in_place")
        thresholds_meta = {
            "thresholds_path": str(thresholds_path),
            "n_videos": thresholds.get("n_videos") if thresholds else None,
        }

    grading = grade_spin(metrics, thresholds=thresholds)

    # 4) Report JSON
    report = {
        "video": str(video_path),
        "model": str(model_path),
        "video_fps": video_meta.fps,
        "body_height_px": ctx.body_height_px,
        "action": "spin_in_place",
        "keypoints_npz": str(kpt_out),
        "metrics": asdict(metrics),
        "thresholds": thresholds_meta,
        "grading": grading,
    }
    report_path = paths.reports / f"{out_stem}_spin_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {"report_path": str(report_path), **report}
