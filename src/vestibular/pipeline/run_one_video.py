from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from ..io.thresholds import load_thresholds
from ..actions.spin_in_place import compute_spin_metrics, grade_spin
from ..config import get_paths
from ..pose.yolo_pose import YoloPoseConfig, YoloPoseEstimator
from ..actions.spin_in_place import compute_spin_metrics, SpinMetrics
from ..viz.plot_series import plot_series

def save_keypoints_npz(out_path: Path, kpt_frames) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # store as ragged arrays (lists) for safety
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
    estimator = YoloPoseEstimator(YoloPoseConfig(model_path=model_path, conf=conf, imgsz=imgsz, device=device))
    results_stream = estimator.predict_video(video_path)
    kpt_frames = estimator.results_to_keypoints(results_stream)

    # 2) Save keypoints
    kpt_out = paths.keypoints / f"{out_stem}.npz"
    save_keypoints_npz(kpt_out, kpt_frames)

    # 3) Metrics
    metrics, debug = compute_spin_metrics(kpt_frames, conf_thresh=kpt_conf_thresh)
    thresholds = None
    thresholds_meta = None
    if thresholds_path:
        all_t = load_thresholds(thresholds_path)
        thresholds = all_t.get("spin_in_place")
        thresholds_meta = {"thresholds_path": str(thresholds_path),
                           "n_videos": thresholds.get("n_videos") if thresholds else None}

    grading = grade_spin(metrics, thresholds=thresholds)
    # 4) Plot
    plot_path = None
    if "angles" in debug and len(debug["angles"]) > 0:
        plot_path = paths.reports / "plots" / f"{out_stem}_trunk_angle.png"
        plot_series(
            debug["angles"],
            title="Spin in place: trunk angle (deg)",
            out_path=plot_path,
            y_label="deg",
        )

    # 5) Report JSON
    report = {
        "video": str(video_path),
        "model": str(model_path),
        "action": "spin_in_place",
        "keypoints_npz": str(kpt_out),
        "plot": str(plot_path) if plot_path else None,
        "metrics": asdict(metrics),
        "thresholds": thresholds_meta,
        "grading": grading,
        "debug": {"frames_used": metrics.frames_used, "frames_total": metrics.frames_total},
    }
    report_path = paths.reports / f"{out_stem}_spin_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"report_path": str(report_path), **report}
