# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation functions for the loco_transformer task.

All functions follow the Isaac Lab manager-based MDP convention:
    func(env: ManagerBasedRLEnv, ...) -> torch.Tensor  with shape (num_envs, dim)
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def base_ang_vel(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Base angular velocity in body frame. (3 dims)"""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b


def projected_gravity(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Projected gravity vector in body frame. (3 dims)"""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


def joint_pos_rel(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Joint positions relative to default. (num_joints dims)"""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos - asset.data.default_joint_pos


def joint_vel_rel(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Joint velocities relative to default. (num_joints dims)"""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel - asset.data.default_joint_vel


def last_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Last action applied to the robot. (num_actions dims)"""
    return env.action_manager.action


def generated_commands(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    """Current velocity command. (3 dims: lin_vel_x, lin_vel_y, ang_vel_z)"""
    return env.command_manager.get_command(command_name)
