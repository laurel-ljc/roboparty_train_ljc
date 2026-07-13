# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom reward functions for the loco_transformer task.

Following the direct/base pattern with projected-gravity masking:
    reward *= clamp(-projected_gravity_b[:, 2], 0, 0.7) / 0.7

All functions follow the Isaac Lab manager-based MDP convention:
    func(env: ManagerBasedRLEnv, ...) -> torch.Tensor  with shape (num_envs,)
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Track linear velocity command (xy) with exponential kernel.

    Compares body-frame linear velocity against the command.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    # mask by uprightness: only reward when the robot is reasonably upright
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], min=0.0, max=0.7) / 0.7
    return reward


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Track angular velocity command (z/yaw) with exponential kernel.

    Compares body-frame z angular velocity against the command.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2]
    )
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], min=0.0, max=0.7) / 0.7
    return reward


def lin_vel_z_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize z-axis base linear velocity (body frame)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize xy-axis base angular velocity (body frame)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def flat_orientation_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize non-flat base orientation via xy components of projected gravity."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions (first-order smoothness)."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def joint_torques_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint torques."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint accelerations."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)
