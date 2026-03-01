"""Action → evaluator registry, using the new EvalContext interface."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Dict, Any

from .context import EvalContext
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


def _evaluate(compute_fn, grade_fn, ctx: EvalContext, thresholds=None):
    metrics, debug = compute_fn(ctx)
    grading = grade_fn(metrics, thresholds=thresholds)
    return {"metrics": metrics, "grading": grading, "debug": debug}


def evaluate_spin(ctx: EvalContext, thresholds=None, **_kw):
    return _evaluate(compute_spin_metrics, grade_spin, ctx, thresholds)


def evaluate_jump(ctx: EvalContext, thresholds=None, **_kw):
    return _evaluate(compute_jump_metrics, grade_jump, ctx, thresholds)


def evaluate_run(ctx: EvalContext, thresholds=None, **_kw):
    return _evaluate(compute_run_metrics, grade_run, ctx, thresholds)


def evaluate_wheelbarrow(ctx: EvalContext, thresholds=None, **_kw):
    return _evaluate(compute_wheelbarrow_metrics, grade_wheelbarrow, ctx, thresholds)


def evaluate_roll(ctx: EvalContext, thresholds=None, **_kw):
    return _evaluate(compute_roll_metrics, grade_roll, ctx, thresholds)


def evaluate_headup(ctx: EvalContext, thresholds=None, **_kw):
    return _evaluate(compute_headup_metrics, grade_headup, ctx, thresholds)


ACTION_REGISTRY: Dict[str, ActionHandler] = {
    "spin_in_place": ActionHandler("spin_in_place", evaluate_spin),
    "jump_in_place": ActionHandler("jump_in_place", evaluate_jump),
    "run_straight": ActionHandler("run_straight", evaluate_run),
    "wheelbarrow_walk": ActionHandler("wheelbarrow_walk", evaluate_wheelbarrow),
    "forward_roll": ActionHandler("forward_roll", evaluate_roll),
    "head_up_prone": ActionHandler("head_up_prone", evaluate_headup),
}
