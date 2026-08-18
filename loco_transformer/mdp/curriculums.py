# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum functions for the loco_transformer task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Update terrain levels from achieved distance and commanded velocity.

    Environments move up after travelling more than half an 8 m tile. They
    move down when travelling less than half of the commanded theoretical
    episode distance. ``TerrainImporter.update_env_origins`` supplies clamping
    at level zero and random resampling after the highest level.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    distance = torch.norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
        dim=1,
    )
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    move_down = distance < (
        torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    )
    move_down &= ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
