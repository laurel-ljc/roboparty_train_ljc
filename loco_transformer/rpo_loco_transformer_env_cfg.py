# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RPO-specific environment configuration for the loco_transformer task."""

from isaaclab.utils import configclass

from robolab.assets.robots.roboparty import RPO_CFG
from .loco_transformer_env_cfg import LocoTransformerEnvCfg


_HISTORICAL_PROPRIOCEPTIVE_TERMS = (
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
)


def _configure_proprioceptive_history(cfg: LocoTransformerEnvCfg) -> None:
    """Stack ten control-rate samples for the 52-D sensed robot state."""
    for group in (cfg.observations.policy, cfg.observations.critic):
        for term_name in _HISTORICAL_PROPRIOCEPTIVE_TERMS:
            term_cfg = getattr(group, term_name)
            term_cfg.history_length = 10
            term_cfg.flatten_history_dim = True


def _configure_play_cfg(cfg: LocoTransformerEnvCfg) -> None:
    """Apply deterministic playback settings shared by Transformer and MLP tasks."""
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 2.5
    cfg.episode_length_s = 40.0

    cfg.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

    cfg.observations.policy.enable_corruption = False

    cfg.events.push_robot = None
    cfg.events.physics_material = None
    cfg.events.add_base_mass = None
    cfg.events.scale_link_mass = None
    cfg.events.randomize_rigid_body_com = None
    cfg.events.scale_actuator_gains = None
    cfg.events.scale_joint_parameters = None
    cfg.events.reset_base = None
    cfg.events.reset_robot_joints = None


@configclass
class RPOLocoTransformerEnvCfg(LocoTransformerEnvCfg):
    """Concrete config that plugs the RPO robot into the loco_transformer environment."""

    def __post_init__(self):
        # post init of parent (sets sim.dt, decimation, etc.)
        super().__post_init__()

        # ------------------------------------------------------
        # Robot — assign RPO_CFG from the project's robot definitions
        # ------------------------------------------------------
        self.scene.robot = RPO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class RPOLocoTransformerEnvCfg_PLAY(RPOLocoTransformerEnvCfg):
    """Play config: single environment, no noise, no pushing, no domain rand."""

    def __post_init__(self):
        super().__post_init__()
        _configure_play_cfg(self)


@configclass
class RPOLocoTransformerHistoryEnvCfg(RPOLocoTransformerEnvCfg):
    """Transformer task with ten frames of the 52-D sensed robot state.

    Commands, previous actions, and the height scan remain single-frame terms,
    giving the term-major observation layout::

        [ang_vel(3x10), gravity(3x10), command(3), joint_pos(23x10),
         joint_vel(23x10), previous_action(23), height_scan(231)]

    The policy and critic observations are both 777-dimensional.  Actor
    histories retain the policy group's observation corruption while critic
    histories remain noise-free.
    """

    def __post_init__(self):
        super().__post_init__()
        _configure_proprioceptive_history(self)


@configclass
class RPOLocoTransformerHistoryEnvCfg_PLAY(RPOLocoTransformerHistoryEnvCfg):
    """Deterministic playback configuration for the history task."""

    def __post_init__(self):
        super().__post_init__()
        _configure_play_cfg(self)


@configclass
class RPOLocoMLPEnvCfg(RPOLocoTransformerEnvCfg):
    """RPO locomotion baseline with 78-dimensional proprioceptive observations."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.scene.height_scanner = None


@configclass
class RPOLocoMLPEnvCfg_PLAY(RPOLocoMLPEnvCfg):
    """Deterministic playback configuration for the 78-dimensional MLP baseline."""

    def __post_init__(self):
        super().__post_init__()
        _configure_play_cfg(self)
