"""Temporary launcher for History/Rough VRAM comparison under the sandbox."""

from __future__ import annotations

import os
import sys


WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(WORKSPACE, "loco_transformer", "scripts"))

import train  # noqa: E402


_load_cfg_from_registry = train.load_cfg_from_registry


def _load_measurement_cfg(task_name: str, entry_point_key: str):
    cfg = _load_cfg_from_registry(task_name, entry_point_key)
    if entry_point_key == "env_cfg_entry_point":
        cfg.scene.robot.spawn.usd_dir = os.path.join(WORKSPACE, ".codex_tmp", "rpo_usd")
        cfg.commands.base_velocity.debug_vis = False
    elif entry_point_key == "rsl_rl_cfg_entry_point" and os.getenv("MEASURE_DISABLE_SYMMETRY") == "1":
        cfg.algorithm.symmetry_cfg = None
    return cfg


train.load_cfg_from_registry = _load_measurement_cfg

try:
    train.main()
finally:
    train.simulation_app.close()
