# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Loco-Transformer task — a ManagerBasedRLEnv for locomotion with 29-term reward.

Uses a custom ``LocoTransformerEnv`` subclass that adds:
- A 3-step action buffer for the ``action_smoothness_l2`` reward.
- Per-foot ray-caster scanners for the ``feet_height`` reward.
"""

import gymnasium as gym

from . import agents


##
# Gym environment registration
##

gym.register(
    id="RPO-Loco-Transformer",
    entry_point=f"{__name__}.loco_transformer_env:LocoTransformerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rpo_loco_transformer_env_cfg:RPOLocoTransformerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.loco_transformer_agent_cfg:LocoTransformerAgentCfg",
    },
)

gym.register(
    id="RPO-Loco-MLP",
    entry_point=f"{__name__}.loco_transformer_env:LocoTransformerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rpo_loco_transformer_env_cfg:RPOLocoMLPEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.loco_transformer_agent_cfg:LocoTransformerMLPAgentCfg",
    },
)

gym.register(
    id="RPO-Loco-MLP-Play",
    entry_point=f"{__name__}.loco_transformer_env:LocoTransformerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rpo_loco_transformer_env_cfg:RPOLocoMLPEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.loco_transformer_agent_cfg:LocoTransformerMLPAgentCfg",
    },
)

gym.register(
    id="RPO-Loco-Transformer-Play",
    entry_point=f"{__name__}.loco_transformer_env:LocoTransformerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rpo_loco_transformer_env_cfg:RPOLocoTransformerEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.loco_transformer_agent_cfg:LocoTransformerAgentCfg",
    },
)

gym.register(
    id="RPO-Loco-Transformer-History",
    entry_point=f"{__name__}.loco_transformer_env:LocoTransformerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rpo_loco_transformer_env_cfg:RPOLocoTransformerHistoryEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.loco_transformer_agent_cfg:LocoTransformerHistoryAgentCfg"
        ),
    },
)

gym.register(
    id="RPO-Loco-Transformer-History-Play",
    entry_point=f"{__name__}.loco_transformer_env:LocoTransformerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rpo_loco_transformer_env_cfg:RPOLocoTransformerHistoryEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.loco_transformer_agent_cfg:LocoTransformerHistoryAgentCfg"
        ),
    },
)

gym.register(
    id="RPO-Loco-Transformer-History-Rough",
    entry_point=f"{__name__}.loco_transformer_env:LocoTransformerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rpo_loco_transformer_env_cfg:RPOLocoTransformerHistoryRoughEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.loco_transformer_agent_cfg:LocoTransformerHistoryRoughAgentCfg"
        ),
    },
)

gym.register(
    id="RPO-Loco-Transformer-History-Rough-Play",
    entry_point=f"{__name__}.loco_transformer_env:LocoTransformerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rpo_loco_transformer_env_cfg:RPOLocoTransformerHistoryRoughEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.loco_transformer_agent_cfg:LocoTransformerHistoryRoughAgentCfg"
        ),
    },
)
