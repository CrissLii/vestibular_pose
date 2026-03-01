from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Any

from .spin_in_place import compute_spin_metrics, grade_spin
from .jump_in_place import compute_jump_metrics, grade_jump
from .run_straight import compute_run_metrics, grade_run
from .wheelbarrow_walk import compute_wheelbarrow_metrics, grade_wheelbarrow
from .forward_roll import compute_roll_metrics, grade_roll
from .head_up_prone import compute_headup_metrics, grade_headup

@dataclass
class ActionHandler:
    name: str
    evaluator: Callable[..., Dict[str, Any]]

def evaluate_spin(kpt_frames, thresholds_spin=None, kpt_conf_thresh=0.2):
    metrics, debug = compute_spin_metrics(kpt_frames, conf_thresh=kpt_conf_thresh)
    grading = grade_spin(metrics, thresholds=thresholds_spin)
    return {"metrics": metrics, "grading": grading, "debug": debug}

def evaluate_jump(kpt_frames, thresholds_spin=None, kpt_conf_thresh=0.2):
    metrics, debug = compute_jump_metrics(kpt_frames, conf_thresh=kpt_conf_thresh)
    grading = grade_jump(metrics)
    return {"metrics": metrics, "grading": grading, "debug": debug}

def evaluate_run(kpt_frames, thresholds_spin=None, kpt_conf_thresh=0.2):
    metrics, debug = compute_run_metrics(kpt_frames, conf_thresh=kpt_conf_thresh)
    grading = grade_run(metrics)
    return {"metrics": metrics, "grading": grading, "debug": debug}

def evaluate_wheelbarrow(kpt_frames, thresholds_spin=None, kpt_conf_thresh=0.2):
    metrics, debug = compute_wheelbarrow_metrics(kpt_frames, conf_thresh=kpt_conf_thresh)
    grading = grade_wheelbarrow(metrics)
    return {"metrics": metrics, "grading": grading, "debug": debug}

def evaluate_roll(kpt_frames, thresholds_spin=None, kpt_conf_thresh=0.2):
    metrics, debug = compute_roll_metrics(kpt_frames, conf_thresh=kpt_conf_thresh)
    grading = grade_roll(metrics)
    return {"metrics": metrics, "grading": grading, "debug": debug}

def evaluate_headup(kpt_frames, thresholds_spin=None, kpt_conf_thresh=0.2):
    metrics, debug = compute_headup_metrics(kpt_frames, conf_thresh=kpt_conf_thresh)
    grading = grade_headup(metrics)
    return {"metrics": metrics, "grading": grading, "debug": debug}

ACTION_REGISTRY: Dict[str, ActionHandler] = {
    "spin_in_place": ActionHandler("spin_in_place", evaluate_spin),
    "jump_in_place": ActionHandler("jump_in_place", evaluate_jump),
    "run_straight": ActionHandler("run_straight", evaluate_run),
    "wheelbarrow_walk": ActionHandler("wheelbarrow_walk", evaluate_wheelbarrow),
    "forward_roll": ActionHandler("forward_roll", evaluate_roll),
    "head_up_prone": ActionHandler("head_up_prone", evaluate_headup),
}
