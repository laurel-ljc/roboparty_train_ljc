# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for the loco_transformer task.

Port of the 29-term RPO-Flat reward set, adapted to ManagerBasedRLEnv API.

Key API changes from RPO-Flat's DirectRLEnv:
- ``env.command_generator.command`` → ``env.command_manager.get_command(command_name)``
- ``env.action_buffer.buffer[:, -1, :]`` → ``env.action_manager.action``
- ``env.action_buffer.buffer[:, -2, :]`` → ``env.action_manager.prev_action``
- ``env.action_buffer.buffer[:, -3, :]`` → ``env.action_buffer.buffer[:, -3, :]``
  (requires custom LocoTransformerEnv with action_buffer)
- ``env.reset_terminated`` → ``env.reset_terminated`` (same name)

All functions follow the Isaac Lab manager-based MDP convention:
    func(env: ManagerBasedRLEnv, ...) -> torch.Tensor  with shape (num_envs,)
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

from .reward_math import joint_pos_limit_violation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# -- configurable constant, consistent with RPO-Flat --
UPRIGHT_MASK_MAX = 0.7  # clamp(-projected_gravity_z, 0, max) divisor for upright mask


def _upright_mask(projected_gravity_z: torch.Tensor) -> torch.Tensor:
    """Mask reward by uprightness: clamp(-g_z, 0, 0.7) / 0.7."""
    return torch.clamp(-projected_gravity_z, min=0.0, max=UPRIGHT_MASK_MAX) / UPRIGHT_MASK_MAX


def _command_norm(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """L1 norm of horizontal velocity command — used to gate stand-still rewards."""
    cmd = env.command_manager.get_command(command_name)
    return torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])


# ======================================================================
# Task rewards — velocity tracking
# ======================================================================


def track_lin_vel_xy_yaw_frame_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track linear velocity (xy) in yaw-aligned world frame with exponential kernel.

    Computes velocity in the robot's heading (yaw) frame so that tracking is
    invariant to pitch/roll of the torso.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= _upright_mask(asset.data.projected_gravity_b[:, 2])
    return reward


def track_ang_vel_z_world_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track angular velocity (z/yaw) in world frame with exponential kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2]
    )
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= _upright_mask(asset.data.projected_gravity_b[:, 2])
    return reward


# ======================================================================
# Regularization penalties
# ======================================================================


def lin_vel_z_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize z-axis base linear velocity in body frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    reward *= _upright_mask(asset.data.projected_gravity_b[:, 2])
    return reward


def ang_vel_xy_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize xy-axis base angular velocity in body frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    reward *= _upright_mask(asset.data.projected_gravity_b[:, 2])
    return reward


def energy(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize energy consumption = sum(|torque_i * vel_i|) across all joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=-1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize first-order action rate (smoothness between consecutive actions)."""
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action),
        dim=1,
    )


def action_smoothness_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize second-order action smoothness.

    Requires ``env.action_buffer`` (a 3-step CircularBuffer) provided by
    the ``LocoTransformerEnv`` subclass.
    """
    # action_buffer stores: [..., a_{t-2}, a_{t-1}, a_t]
    # We want (a_{t} - 2*a_{t-1} + a_{t-2})^2
    return torch.sum(
        torch.square(
            env.action_buffer.buffer[:, -3, :]
            - 2 * env.action_buffer.buffer[:, -2, :]
            + env.action_buffer.buffer[:, -1, :]
        ),
        dim=1,
    )


def joint_torques_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint torques (L2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def joint_vel_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint velocities (L2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint accelerations (L2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def flat_orientation_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize non-flat base orientation via xy components of projected gravity."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def undesired_contacts(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize contacts on non-foot body parts.

    Counts the number of bodies (filtered by sensor_cfg.body_names) that are
    in contact with the ground (force > 1N).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(
        torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1
    )[0] > 1.0
    return torch.sum(is_contact, dim=1)


def termination_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize early termination (non-timeout episode endings)."""
    return env.reset_terminated.float()


# ======================================================================
# Gait / foot behavior rewards
# ======================================================================


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward longer foot air time during single-stance phases (bipedal gait).

    Returns the shorter air time among the two feet when only one foot is
    in contact with the ground.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    is_contact = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
        > 1.0
    )
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_mode_time = torch.where(is_contact, contact_time, air_time)
    single_stance = torch.sum(is_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, min=0.0, max=threshold)
    # no reward for zero command
    reward *= _command_norm(env, "base_velocity") > 0.01
    reward *= _upright_mask(env.scene["robot"].data.projected_gravity_b[:, 2])
    return reward


def feet_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize foot sliding: body xy velocity weighted by ground contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(body_vel.norm(dim=-1) * contacts.float(), dim=1)


def feet_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 500,
    max_reward: float = 400,
) -> torch.Tensor:
    """Penalize excessive foot contact forces above a threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = torch.sum(
        torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=2),
        dim=1,
    )
    return (force - threshold).clamp(min=0.0, max=max_reward)


def feet_stumble(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize stumbling: horizontal force > 3x vertical force on feet."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 3 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    ).float()


def feet_orientation_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize non-flat foot orientation (xy components of projected gravity per body)."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = torch.stack(
        [
            math_utils.quat_apply_inverse(
                asset.data.body_quat_w[:, body_id, :], asset.data.GRAVITY_VEC_W
            )
            for body_id in asset_cfg.body_ids
            if body_id is not None
        ],
        dim=-1,
    )
    return torch.sum(torch.sum(torch.square(body_orientation[:, :2, :]), dim=1), dim=-1)


def body_distance_y(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min: float = 0.2,
    max: float = 0.5,
) -> torch.Tensor:
    """Reward appropriate y-direction spacing between two bodies (e.g. feet, knees).

    Uses an exponential barrier: reward ≈ 1 when min < distance < max, decays outside.
    """
    assert len(asset_cfg.body_ids) == 2, "body_distance_y expects exactly 2 body_ids"
    asset: Articulation = env.scene[asset_cfg.name]
    root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, 2, -1)
    root_pos_w = asset.data.root_pos_w.unsqueeze(1).expand(-1, 2, -1)
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    body_pos_b = math_utils.quat_apply_inverse(root_quat_w, body_pos_w - root_pos_w)
    distance = torch.abs(body_pos_b[:, 0, 1] - body_pos_b[:, 1, 1])
    d_min = torch.clamp(distance - min, min=-0.5, max=0.0)
    d_max = torch.clamp(distance - max, min=0.0, max=0.5)
    return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Reward both feet in contact with ground when the velocity command is zero."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    reward = (torch.sum(contacts, dim=-1) == 2).float()
    reward *= _command_norm(env, command_name) < 0.01
    reward *= _upright_mask(env.scene["robot"].data.projected_gravity_b[:, 2])
    return reward


def feet_height(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg1: SceneEntityCfg | None = None,
    sensor_cfg2: SceneEntityCfg | None = None,
    ankle_height: float = 0.04,
    threshold: float = 0.02,
) -> torch.Tensor:
    """Reward swing foot clearance from ground during single-stance phases.

    Uses per-foot ray-caster scanners (``sensor_cfg1`` / ``sensor_cfg2``) to
    measure the height of each foot above the terrain.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    # Measure foot height via per-foot scanners
    feet_height_tensor = torch.stack(
        [
            env.scene[sensor.name].data.pos_w[:, 2]
            - env.scene[sensor.name].data.ray_hits_w[..., 2].mean(dim=-1)
            for sensor in [sensor_cfg1, sensor_cfg2]
            if sensor is not None
        ],
        dim=-1,
    )
    feet_height_tensor = torch.clamp(feet_height_tensor - ankle_height, min=0.0, max=1.0)
    feet_height_tensor = torch.nan_to_num(feet_height_tensor, nan=1.0, posinf=1.0, neginf=0.0)
    single_stance = contacts.sum(dim=1) == 1
    rew_pos = feet_height_tensor > threshold
    reward = torch.where(
        torch.logical_and(~contacts, single_stance.unsqueeze(-1)),
        rew_pos.float(),
        0.0,
    ).sum(dim=1)
    reward *= _command_norm(env, "base_velocity") > 0.01
    reward *= _upright_mask(env.scene["robot"].data.projected_gravity_b[:, 2])
    return reward


# ======================================================================
# Joint posture / deviation rewards
# ======================================================================


def joint_pos_limits(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint positions that exceed soft limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    out_of_limits = joint_pos_limit_violation(
        asset.data.joint_pos[:, asset_cfg.joint_ids],
        asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids],
    )
    return torch.sum(out_of_limits, dim=1)


def joint_deviation_l1(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize L1 deviation from default joint positions."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(
        torch.abs(
            asset.data.joint_pos[:, asset_cfg.joint_ids]
            - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        ),
        dim=1,
    )


def upward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward upward base orientation: -projected_gravity_z (1 when perfectly upright)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return -asset.data.projected_gravity_b[:, 2]


def stand_still(
    env: ManagerBasedRLEnv,
    pos_cfg: SceneEntityCfg,
    vel_cfg: SceneEntityCfg,
    pos_weight: float = 1.0,
    vel_weight: float = 0.04,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Penalize joint movement when the robot should be standing still.

    Activated only when velocity command is near zero AND the robot is
    nearly stationary.
    """
    asset: Articulation = env.scene["robot"]
    cmd_norm = _command_norm(env, command_name)
    body_lin_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    body_ang_vel = torch.abs(asset.data.root_ang_vel_b[:, 2])
    body_vel = body_ang_vel + body_lin_vel
    pos_reward = pos_weight * torch.sum(
        torch.abs(
            asset.data.joint_pos[:, pos_cfg.joint_ids]
            - asset.data.default_joint_pos[:, pos_cfg.joint_ids]
        ),
        dim=1,
    )
    vel_reward = vel_weight * torch.sum(
        torch.abs(asset.data.joint_vel[:, vel_cfg.joint_ids]),
        dim=1,
    )
    reward = torch.where(
        torch.logical_or(cmd_norm > 0.01, body_vel > 0.5),
        torch.tensor(0.0, device=body_vel.device),
        pos_reward + vel_reward,
    )
    reward *= _upright_mask(asset.data.projected_gravity_b[:, 2])
    return reward
