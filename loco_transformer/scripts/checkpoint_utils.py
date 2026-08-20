"""Checkpoint argument and loading helpers for the training entry point.

This module intentionally has no Isaac Sim dependencies so its behavior can be
unit-tested without launching the simulator.
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Literal


CheckpointMode = Literal["resume", "finetune"]


def add_finetune_args(parser: argparse.ArgumentParser) -> None:
    """Add the explicit cross-task finetuning checkpoint argument."""
    parser.add_argument(
        "--finetune-path",
        "--finetune_path",
        dest="finetune_path",
        type=str,
        default=None,
        help=(
            "Checkpoint whose model weights initialize a new task. The optimizer is not loaded "
            "and the learning iteration starts from zero."
        ),
    )


def validate_finetune_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str | None:
    """Validate finetune CLI arguments and normalize the checkpoint path."""
    if args.finetune_path is None:
        return None

    conflicts = []
    if getattr(args, "resume", False):
        conflicts.append("--resume")
    if getattr(args, "load_run", None) is not None:
        conflicts.append("--load_run")
    if getattr(args, "checkpoint", None) is not None:
        conflicts.append("--checkpoint")
    if conflicts:
        parser.error(f"--finetune-path cannot be combined with {', '.join(conflicts)}")

    checkpoint_path = os.path.abspath(os.path.expanduser(args.finetune_path))
    if not os.path.isfile(checkpoint_path):
        parser.error(f"finetune checkpoint does not exist: {checkpoint_path}")
    args.finetune_path = checkpoint_path
    return checkpoint_path


def validate_finetune_runner(runner_class_name: str) -> None:
    """Restrict finetuning to the runner whose checkpoint semantics are defined here."""
    if runner_class_name != "OnPolicyRunner":
        raise ValueError(
            "--finetune-path currently supports only OnPolicyRunner, "
            f"but the selected agent uses {runner_class_name}."
        )


def get_resume_checkpoint_path(log_path: str, run_dir: str, checkpoint: str) -> str:
    """Find the latest checkpoint, skipping empty run directories."""
    runs = [
        os.path.join(log_path, run.name)
        for run in os.scandir(log_path)
        if run.is_dir() and re.match(run_dir, run.name)
    ]
    if not runs:
        raise ValueError(f"No runs present in the directory: '{log_path}' match: '{run_dir}'.")
    runs = sorted(runs, key=os.path.getmtime)
    for run_path in reversed(runs):
        model_checkpoints = [file for file in os.listdir(run_path) if re.match(checkpoint, file)]
        if model_checkpoints:
            model_checkpoints.sort(key=lambda name: f"{name:0>15}")
            return os.path.join(run_path, model_checkpoints[-1])
    raise ValueError(f"No checkpoints in the directory: '{log_path}' match '{checkpoint}'.")


def load_checkpoint_for_training(
    runner,
    checkpoint_path: str,
    *,
    mode: CheckpointMode,
    map_location: str,
    snapshot_path: str | None = None,
) -> int | None:
    """Load a resume or finetune checkpoint with deliberately distinct semantics.

    Resume restores every state supported by the runner and preserves the saved
    iteration. Finetune restores weights only, resets the iteration, and may
    save a rank-zero pre-update snapshot containing the fresh optimizer.
    """
    if mode == "resume":
        runner.load(checkpoint_path, map_location=map_location)
        return None
    if mode != "finetune":
        raise ValueError(f"Unsupported checkpoint loading mode: {mode}")

    runner.load(checkpoint_path, load_optimizer=False, map_location=map_location)
    source_iteration = int(runner.current_learning_iteration)
    runner.current_learning_iteration = 0

    if snapshot_path is not None:
        runner.save(
            snapshot_path,
            infos={
                "finetune_source": {
                    "checkpoint": os.path.abspath(checkpoint_path),
                    "iteration": source_iteration,
                }
            },
        )
    return source_iteration


__all__ = [
    "CheckpointMode",
    "add_finetune_args",
    "get_resume_checkpoint_path",
    "load_checkpoint_for_training",
    "validate_finetune_args",
    "validate_finetune_runner",
]
