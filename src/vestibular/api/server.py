"""FastAPI backend for vestibular pose evaluation."""
from __future__ import annotations

import json
import math
import shutil
import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles


def _sanitize(obj: Any) -> Any:
    """Replace NaN / Infinity with None so JSON serialization succeeds."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj

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
    """Streaming evaluation: yields NDJSON progress events, final line is the result."""
    import time as _time

    suffix = Path(video.filename or "video.mp4").suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    shutil.copyfileobj(video.file, tmp)
    tmp.close()
    tmp_path = tmp.name
    filename = video.filename

    def _event(evt: str, **kwargs) -> str:
        return json.dumps({"event": evt, **kwargs}, ensure_ascii=False) + "\n"

    def generate():
        timings: dict[str, float] = {}

        try:
            # Step 1: Pose inference
            t0 = _time.perf_counter()
            kpt_frames, video_meta = step_pose_infer(
                video_path=tmp_path, model_path=model_path,
            )
            elapsed = round(_time.perf_counter() - t0, 2)
            timings["pose_inference"] = elapsed
            yield _event("step_done", step="pose", elapsed=elapsed)

            # Step 2: Detect action
            t0 = _time.perf_counter()
            action_id, candidates, feat = step_detect_action(kpt_frames)
            elapsed = round(_time.perf_counter() - t0, 2)
            timings["action_detection"] = elapsed
            yield _event("step_done", step="detect", elapsed=elapsed,
                         action=action_id, action_zh=zh(action_id))

            # Step 3: Evaluate
            t0 = _time.perf_counter()
            thresholds, _ = step_load_thresholds(paths.data / "thresholds.json")
            ctx = step_build_context(
                kpt_frames=kpt_frames, fps=video_meta.fps,
                view=view, kpt_conf_thresh=kpt_conf,
            )
            action_id_eval, metrics_dict, grading, debug = step_evaluate(
                action_id=action_id, ctx=ctx, thresholds=thresholds,
            )
            elapsed = round(_time.perf_counter() - t0, 2)
            timings["evaluation"] = elapsed
            yield _event("step_done", step="evaluate", elapsed=elapsed)

            # Step 4: Render video
            t0 = _time.perf_counter()
            out_stem = Path(filename or "video").stem
            annotated_path = paths.videos / f"{out_stem}_{action_id_eval}_annotated.mp4"
            step_render_video(
                video_path=tmp_path, kpt_frames=kpt_frames,
                out_path=annotated_path, action_id=action_id_eval,
                grading=grading, conf_thresh=kpt_conf,
            )
            elapsed = round(_time.perf_counter() - t0, 2)
            timings["video_render"] = elapsed
            yield _event("step_done", step="render", elapsed=elapsed)

            # Step 5: Charts & report
            t0 = _time.perf_counter()
            reasons = grading.get("reasons", {})
            generate_radar_chart(reasons, zh(action_id_eval))
            generate_cop_trajectory(kpt_frames, conf_thresh=kpt_conf)
            generate_symmetry_chart(reasons, action_id_eval)

            radar_data = []
            for k, v in reasons.items():
                if k.endswith("_level"):
                    mk = k.replace("_level", "")
                    radar_data.append({
                        "metric": mk, "score": _sev_to_5(str(v)),
                        "level": str(v), "value": reasons.get(mk),
                    })
            cop_data = _extract_cop_data(kpt_frames, kpt_conf)
            sym_data = _extract_symmetry_data(reasons, action_id_eval)

            session_id = str(uuid.uuid4())[:8]
            _sessions[session_id] = {
                "kpt_frames": kpt_frames, "fps": video_meta.fps,
                "view": view, "kpt_conf": kpt_conf,
                "candidates": candidates, "feat": feat,
                "thresholds": thresholds,
                "video_filename": filename, "tmp_path": tmp_path,
            }

            report = {
                "video": filename, "video_fps": video_meta.fps,
                "body_height_px": ctx.body_height_px, "view": view,
                "action_detected": action_id_eval,
                "action_detected_zh": zh(action_id_eval),
                "action_candidates": [
                    {"action": c.action, "label": zh(c.action),
                     "score": round(c.score, 4), "details": c.details}
                    for c in candidates
                ],
                "metrics": metrics_dict, "grading": grading,
            }
            report_path = paths.reports / f"{out_stem}_{action_id_eval}_report.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(_sanitize(report), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            elapsed = round(_time.perf_counter() - t0, 2)
            timings["chart_generation"] = elapsed
            timings["total"] = round(sum(timings.values()), 2)
            yield _event("step_done", step="done", elapsed=elapsed)

            # Final result
            result = _sanitize({
                "session_id": session_id,
                "action_detected": action_id_eval,
                "action_detected_zh": zh(action_id_eval),
                "candidates": [
                    {"action": c.action, "label": zh(c.action),
                     "score": round(c.score, 4)}
                    for c in candidates[:5]
                ],
                "metrics": metrics_dict, "grading": grading,
                "radar_data": radar_data, "cop_data": cop_data,
                "symmetry_data": sym_data,
                "annotated_video": f"/static/videos/{annotated_path.name}",
                "report_url": f"/static/reports/{report_path.name}",
                "timings": timings,
            })
            yield json.dumps({"event": "complete", "result": result},
                             ensure_ascii=False) + "\n"

        except Exception as e:
            yield json.dumps({"event": "error", "detail": str(e)},
                             ensure_ascii=False) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


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

    return _sanitize({
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
    })


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
