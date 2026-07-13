# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom termination functions for the loco_transformer task.

All functions follow the Isaac Lab manager-based MDP convention:
    func(env: ManagerBasedRLEnv, ...) -> torch.Tensor  with shape (num_envs,) of bool
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate episodes that exceed the maximum episode length."""
    return env.episode_length_buf >= env.max_episode_length


def bad_orientation(
    env: ManagerBasedRLEnv,
    limit_angle: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the base orientation exceeds the limit angle from upright.

    The projected gravity z-component gives cos(pitch_angle_from_vertical).
    We terminate when |angle| > limit_angle.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.acos(-asset.data.projected_gravity_b[:, 2]).abs() > limit_angle


def root_height_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the robot root height falls below the minimum."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < minimum_height


def illegal_contact(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate when undesired body parts contact the ground with force above threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(
        torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1),
        dim=1,
    )[0] > threshold
    return torch.any(is_contact, dim=1)
