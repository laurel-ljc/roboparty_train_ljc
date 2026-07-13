# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Agent configuration for the loco_transformer task.

A simple PPO agent with a 3-layer MLP policy (256 → 128 → 64).
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class LocoTransformerAgentCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for the loco_transformer PPO agent with 3-layer MLP."""

    class_name = "OnPolicyRunner"
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 9001
    save_interval = 500
    experiment_name = "loco_transformer"
    run_name = ""
    logger = "wandb"
    neptune_project = "robolab"
    wandb_project = "loco_transformer"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
    clip_actions = 100.0
    empirical_normalization = False
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}

    # 3-layer MLP policy: 256 → 128 → 64
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,
        rnd_cfg=None,
    )
