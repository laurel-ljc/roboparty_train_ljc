# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Loco-Transformer task — a minimal ManagerBasedRLEnv for locomotion."""

import gymnasium as gym

from . import agents


##
# Gym environment registration
##

gym.register(
    id="RPO-Loco-Transformer",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rpo_loco_transformer_env_cfg:RPOLocoTransformerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.loco_transformer_agent_cfg:LocoTransformerAgentCfg",
    },
)

gym.register(
    id="RPO-Loco-Transformer-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rpo_loco_transformer_env_cfg:RPOLocoTransformerEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.loco_transformer_agent_cfg:LocoTransformerAgentCfg",
    },
)
