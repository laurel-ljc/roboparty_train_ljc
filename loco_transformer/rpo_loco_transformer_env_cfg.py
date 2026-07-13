# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RPO-specific environment configuration for the loco_transformer task."""

from isaaclab.utils import configclass

from robolab.assets.robots.roboparty import RPO_CFG
from .loco_transformer_env_cfg import LocoTransformerEnvCfg


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
    """Play config: single environment, no noise, no pushing."""

    def __post_init__(self):
        super().__post_init__()

        # single environment for visualization
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        # fixed velocity command
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # disable observation noise
        self.observations.policy.enable_corruption = False

        # disable random pushing
        self.events.push_robot = None
