# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP module for the loco_transformer task.

All observation, reward, and termination functions are self-written.
Event functions and action/command config classes are imported from Isaac Lab core.
"""

# -- Custom observation functions --
from .observations import (
    base_ang_vel,
    generated_commands,
    joint_pos_rel,
    joint_vel_rel,
    last_action,
    projected_gravity,
)

# -- Isaac Lab built-in observation functions --
from isaaclab.envs.mdp.observations import height_scan  # noqa: F401

# -- Curriculum functions --
from .curriculums import terrain_levels_vel

# -- History-Rough symmetry callback --
from .symmetry import compute_symmetric_states

# -- Custom reward functions (29 terms, 1:1 from RPO-Flat) --
from .rewards import (
    # task / tracking
    track_lin_vel_xy_yaw_frame_exp,
    track_ang_vel_z_world_exp,
    # regularization
    lin_vel_z_l2,
    ang_vel_xy_l2,
    energy,
    action_rate_l2,
    action_smoothness_l2,
    joint_torques_l2,
    joint_vel_l2,
    joint_acc_l2,
    flat_orientation_l2,
    undesired_contacts,
    termination_penalty,
    # gait / foot behavior
    feet_air_time_positive_biped,
    feet_slide,
    feet_force,
    feet_stumble,
    feet_orientation_l2,
    body_distance_y,
    feet_contact_without_cmd,
    feet_height,
    # joint posture / deviation
    joint_pos_limits,
    joint_deviation_l1,
    upward,
    stand_still,
)

# -- Custom termination functions --
from .terminations import (
    bad_orientation,
    illegal_contact,
    root_height_below_minimum,
    time_out,
)

# -- Command and action configs (Isaac Lab core) --
from isaaclab.envs.mdp import JointPositionActionCfg, UniformVelocityCommandCfg  # noqa: F401

# -- Event functions (Isaac Lab core, physics-level randomizations) --
from isaaclab.envs.mdp import (  # noqa: F401
    push_by_setting_velocity,
    randomize_actuator_gains,
    randomize_joint_parameters,
    randomize_rigid_body_com,
    randomize_rigid_body_mass,
    randomize_rigid_body_material,
    reset_joints_by_scale,
    reset_root_state_uniform,
)
