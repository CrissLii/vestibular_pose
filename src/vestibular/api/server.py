"""FastAPI backend for vestibular pose evaluation."""
from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

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

_API_SEV_SCORE = {
    "正常": 5, "轻度偏差": 4, "中度偏差": 2, "重度偏差": 1,
    "NORMAL": 5, "MILD": 4, "MODERATE": 2, "SEVERE": 1,
}


def _sev_to_5(sev: str) -> int:
    return _API_SEV_SCORE.get(sev, 3)

app = FastAPI(title="儿童统感训练评估 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session cache: session_id -> { kpt_frames, fps, candidates, ... }
_sessions: dict = {}

paths = get_paths()
paths.videos.mkdir(parents=True, exist_ok=True)
paths.reports.mkdir(parents=True, exist_ok=True)

app.mount("/static/videos", StaticFiles(directory=str(paths.videos)), name="videos")
app.mount("/static/reports", StaticFiles(directory=str(paths.reports)), name="reports")


@app.get("/api/actions")
def list_actions():
    """List all supported actions with Chinese labels."""
    return [
        {"id": k, "label": v}
        for k, v in ACTION_LABELS_ZH.items()
        if k != "unknown"
    ]


@app.post("/api/evaluate")
async def evaluate_video(
    video: UploadFile = File(...),
    model_path: str = Form("yolo11n-pose.pt"),
    kpt_conf: float = Form(0.20),
    view: str = Form("unknown"),
):
    """Full evaluation pipeline: upload video → pose → detect → evaluate → render."""
    # Save uploaded video to temp file
    suffix = Path(video.filename or "video.mp4").suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    shutil.copyfileobj(video.file, tmp)
    tmp.close()
    tmp_path = tmp.name

    try:
        # Step 1: Pose inference
        kpt_frames, video_meta = step_pose_infer(
            video_path=tmp_path, model_path=model_path,
        )

        # Step 2: Detect action
        action_id, candidates, feat = step_detect_action(kpt_frames)

        # Step 3: Load thresholds (optional)
        thresholds, thresholds_meta = step_load_thresholds(
            paths.data / "thresholds.json"
        )

        # Step 4: Build context & evaluate
        ctx = step_build_context(
            kpt_frames=kpt_frames,
            fps=video_meta.fps,
            view=view,
            kpt_conf_thresh=kpt_conf,
        )
        action_id_eval, metrics_dict, grading, debug = step_evaluate(
            action_id=action_id, ctx=ctx, thresholds=thresholds,
        )

        # Step 5: Render annotated video
        out_stem = Path(video.filename or "video").stem
        annotated_path = paths.videos / f"{out_stem}_{action_id_eval}_annotated.mp4"
        step_render_video(
            video_path=tmp_path,
            kpt_frames=kpt_frames,
            out_path=annotated_path,
            action_id=action_id_eval,
            grading=grading,
            conf_thresh=kpt_conf,
        )

        # Step 6: Generate charts
        reasons = grading.get("reasons", {})
        radar_path = generate_radar_chart(reasons, zh(action_id_eval))
        cop_path = generate_cop_trajectory(kpt_frames, conf_thresh=kpt_conf)
        sym_path = generate_symmetry_chart(reasons, action_id_eval)

        # Build metric scores for frontend radar chart (1-5 scale)
        radar_data = []
        for k, v in reasons.items():
            if k.endswith("_level"):
                metric_key = k.replace("_level", "")
                radar_data.append({
                    "metric": metric_key,
                    "score": _sev_to_5(str(v)),
                    "level": str(v),
                    "value": reasons.get(metric_key),
                })

        # Build COP trajectory data
        cop_data = _extract_cop_data(kpt_frames, kpt_conf)

        # Build symmetry data
        sym_data = _extract_symmetry_data(reasons, action_id_eval)

        # Cache session for re-evaluation
        session_id = str(uuid.uuid4())[:8]
        _sessions[session_id] = {
            "kpt_frames": kpt_frames,
            "fps": video_meta.fps,
            "view": view,
            "kpt_conf": kpt_conf,
            "candidates": candidates,
            "feat": feat,
            "thresholds": thresholds,
            "video_filename": video.filename,
            "tmp_path": tmp_path,
        }

        # Save report
        report = {
            "video": video.filename,
            "video_fps": video_meta.fps,
            "body_height_px": ctx.body_height_px,
            "view": view,
            "action_detected": action_id_eval,
            "action_detected_zh": zh(action_id_eval),
            "action_candidates": [
                {"action": c.action, "label": zh(c.action),
                 "score": round(c.score, 4), "details": c.details}
                for c in candidates
            ],
            "metrics": metrics_dict,
            "grading": grading,
        }
        report_path = paths.reports / f"{out_stem}_{action_id_eval}_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "session_id": session_id,
            "action_detected": action_id_eval,
            "action_detected_zh": zh(action_id_eval),
            "candidates": [
                {"action": c.action, "label": zh(c.action),
                 "score": round(c.score, 4)}
                for c in candidates[:5]
            ],
            "metrics": metrics_dict,
            "grading": grading,
            "radar_data": radar_data,
            "cop_data": cop_data,
            "symmetry_data": sym_data,
            "annotated_video": f"/static/videos/{annotated_path.name}",
            "report_url": f"/static/reports/{report_path.name}",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/re-evaluate")
async def re_evaluate(
    session_id: str = Form(...),
    action_id: str = Form(...),
):
    """Re-evaluate with a manually selected action (no re-running pose inference)."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found. Please run evaluation first.")

    kpt_frames = session["kpt_frames"]
    fps = session["fps"]
    view = session["view"]
    kpt_conf = session["kpt_conf"]
    thresholds = session["thresholds"]
    tmp_path = session["tmp_path"]

    ctx = step_build_context(
        kpt_frames=kpt_frames, fps=fps,
        view=view, kpt_conf_thresh=kpt_conf,
    )
    action_id_eval, metrics_dict, grading, debug = step_evaluate(
        action_id=action_id, ctx=ctx, thresholds=thresholds,
    )

    out_stem = Path(session["video_filename"] or "video").stem
    annotated_path = paths.videos / f"{out_stem}_{action_id_eval}_re_annotated.mp4"
    step_render_video(
        video_path=tmp_path,
        kpt_frames=kpt_frames,
        out_path=annotated_path,
        action_id=action_id_eval,
        grading=grading,
        conf_thresh=kpt_conf,
    )

    reasons = grading.get("reasons", {})
    radar_data = []
    for k, v in reasons.items():
        if k.endswith("_level"):
            metric_key = k.replace("_level", "")
            radar_data.append({
                "metric": metric_key,
                "score": _sev_to_5(str(v)),
                "level": str(v),
                "value": reasons.get(metric_key),
            })

    cop_data = _extract_cop_data(kpt_frames, kpt_conf)
    sym_data = _extract_symmetry_data(reasons, action_id_eval)

    return {
        "session_id": session_id,
        "action_detected": action_id_eval,
        "action_detected_zh": zh(action_id_eval),
        "candidates": [
            {"action": c.action, "label": zh(c.action),
             "score": round(c.score, 4)}
            for c in session["candidates"][:5]
        ],
        "metrics": metrics_dict,
        "grading": grading,
        "radar_data": radar_data,
        "cop_data": cop_data,
        "symmetry_data": sym_data,
        "annotated_video": f"/static/videos/{annotated_path.name}",
    }


@app.get("/api/chart/radar/{session_id}")
def get_radar_chart(session_id: str):
    """Return pre-generated radar chart image."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    # Generate on-the-fly from cached data
    return JSONResponse({"message": "Use radar_data from evaluation response"})


def _extract_cop_data(kpt_frames, conf_thresh: float = 0.20) -> list:
    """Extract COP (hip midpoint) trajectory as JSON-serializable list."""
    import numpy as np
    LHP, RHP = 11, 12
    points = []
    for i, f in enumerate(kpt_frames):
        xy, cf = f.xy, f.conf
        if max(LHP, RHP) >= len(cf):
            continue
        if cf[LHP] < conf_thresh or cf[RHP] < conf_thresh:
            continue
        hip = (xy[LHP] + xy[RHP]) / 2.0
        points.append({
            "x": round(float(hip[0]), 1),
            "y": round(float(hip[1]), 1),
            "t": round(i / len(kpt_frames), 3),
        })
    # Downsample if too many points
    if len(points) > 300:
        step = len(points) // 300
        points = points[::step]
    return points


def _extract_symmetry_data(reasons: dict, action_id: str) -> list:
    """Extract symmetry comparison data for frontend bar chart."""
    import numpy as np
    pairs = []

    if action_id == "jump_in_place":
        asym = reasons.get("asym_limb")
        if asym is not None and not (isinstance(asym, float) and np.isnan(asym)):
            pairs.append({
                "label": "肢体对称",
                "left": 1.0,
                "right": round(max(0, 1.0 - float(asym)), 3),
            })

    elif action_id == "wheelbarrow_walk":
        sym = reasons.get("sl_sym")
        if sym is not None and not (isinstance(sym, float) and np.isnan(sym)):
            pairs.append({"label": "步长对称", "left": 1.0, "right": round(float(sym), 3)})
        ai = reasons.get("ai_hand")
        if ai is not None and not (isinstance(ai, float) and np.isnan(ai)):
            pairs.append({"label": "手交替", "left": 1.0,
                          "right": round(max(0, 1.0 - abs(float(ai))), 3)})

    elif action_id == "head_up_prone":
        si = reasons.get("si_load")
        if si is not None and not (isinstance(si, float) and np.isnan(si)):
            pairs.append({"label": "承重对称", "left": 1.0,
                          "right": round(max(0, 1.0 - float(si)), 3)})

    return pairs


def create_app() -> FastAPI:
    """Factory for external usage."""
    paths.videos.mkdir(parents=True, exist_ok=True)
    paths.reports.mkdir(parents=True, exist_ok=True)
    return app
