# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Agent configuration for the loco_transformer task.

Three agent variants are provided:

- ``LocoTransformerAgentCfg`` — uses the cross-attention actor-critic,
  suitable for the height-scanner-equipped environment.
- ``LocoTransformerHistoryAgentCfg`` — uses the same cross-attention model
  with ten frames of the 52-D sensed robot state.
- ``LocoTransformerHistoryRoughAgentCfg`` — keeps that model and separates
  mixed-terrain curriculum runs into their own experiment directory.
- ``LocoTransformerMLPAgentCfg`` — legacy pure-MLP actor-critic,
  for the baseline without height-scan input (78-dim obs, 23 actions).
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from ..mdp.symmetry import compute_symmetric_states


@configclass
class CrossAttentionActorCriticCfg(RslRlPpoActorCriticCfg):
    """Configuration for the cross-attention actor-critic module.

    Inherits the standard PPO actor-critic fields (actor_hidden_dims, activation, etc.)
    from :class:`RslRlPpoActorCriticCfg` and adds cross-attention-specific parameters.

    The ``class_name`` is overridden to ``"CrossAttentionActorCritic"``; the runner
    resolves it via ``rsl_rl.utils.resolve_callable``.

    Observation layout (flat, concatenated — matching the current env):
        [base_ang_vel(3) | proj_gravity(3) | commands(3) |
         joint_pos(23) | joint_vel(23) | last_action(23) |
         height_scan(231 = 21×11)]

    ``actor_perception_range`` / ``critic_perception_range`` tell the module which
    slice of the flat tensor is the height map.  The remainder is treated as
    proprioceptive input.
    """

    class_name: str = "rsl_rl.modules.actor_critic_cross_attn:CrossAttentionActorCritic"
    """Module class name with explicit module path for resolve_callable."""

    init_noise_std: float = 1.0
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False
    actor_hidden_dims: list[int] = [256, 128, 64]
    critic_hidden_dims: list[int] = [256, 128, 64]
    activation: str = "elu"

    # ------------------------------------------------------------------
    # Observation split
    # ------------------------------------------------------------------
    actor_perception_range: tuple[int, int] = (78, 309)
    """(start, end) of the height-scan slice inside the actor's flat observation."""

    critic_perception_range: tuple[int, int] = (78, 309)
    """(start, end) of the height-scan slice inside the critic's flat observation."""

    height_map_shape: tuple[int, int] = (11, 21)
    """(H, W) to reshape the flat height-scan vector into a 2D grid.
    11 rows × 21 cols = 231 rays from GridPatternCfg(resolution=0.1, size=(2.0, 1.0)).
    """

    # ------------------------------------------------------------------
    # Cross-attention encoder
    # ------------------------------------------------------------------
    embed_dim: int = 64
    """Common embedding dimension for cross-attention."""

    num_heads: int = 8
    """Number of multi-head attention heads."""

    grid_size: tuple[int, int] = (4, 3)
    """(m, n) spatial grid to split the height map into patches. 4×3 = 12 patches."""

    proprio_hidden_dims: list[int] = [128]
    """Hidden dimensions of the MLP that encodes proprioceptive input before cross-attn."""

    cnn_channels: list[int] = [16, 32, 64]
    """Output channel list for the CNN that encodes the height map before cross-attn."""

    cnn_kernel_size: int = 3
    """Kernel size for all CNN layers."""

    cnn_activation: str = "elu"
    """Activation function for CNN layers."""


@configclass
class LocoTransformerAgentCfg(RslRlOnPolicyRunnerCfg):
    """Cross-attention PPO agent for the loco_transformer task.

    Uses the ``CrossAttentionActorCritic`` module which fuses
    proprioceptive observations with height-scan perception via
    a CNN + cross-attention encoder.
    """

    class_name = "OnPolicyRunner"
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 9001
    save_interval = 500
    experiment_name = "loco_transformer"
    run_name = ""
    logger = "tensorboard"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
    clip_actions = 100.0
    empirical_normalization = False

    # Observation group mapping — matches the env's obs groups.
    # "policy" and "critic" are the two 1D concatenated groups from the env.
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}

    # Cross-attention actor-critic module
    policy = CrossAttentionActorCriticCfg(
        class_name="rsl_rl.modules.actor_critic_cross_attn:CrossAttentionActorCritic",
        actor_perception_range=(78, 309),
        critic_perception_range=(78, 309),
        height_map_shape=(11, 21),
        embed_dim=64,
        num_heads=8,
        grid_size=(4, 3),
        proprio_hidden_dims=[128],
        cnn_channels=[16, 32, 64],
        cnn_kernel_size=3,
        cnn_activation="elu",
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type="scalar",
        state_dependent_std=False,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
    )

    # PPO algorithm
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="rsl_rl.algorithms.ppo:PPO",
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


@configclass
class LocoTransformerHistoryAgentCfg(LocoTransformerAgentCfg):
    """Cross-attention PPO agent for the 777-D ten-frame history task."""

    experiment_name = "loco_transformer_history"
    resume = False
    # The 546-D non-height portion contains 52 sensed dimensions over ten
    # frames plus the current 3-D command and 23-D previous action.
    policy = CrossAttentionActorCriticCfg(
        class_name="rsl_rl.modules.actor_critic_cross_attn:CrossAttentionActorCritic",
        actor_perception_range=(546, 777),
        critic_perception_range=(546, 777),
        height_map_shape=(11, 21),
        embed_dim=64,
        num_heads=8,
        grid_size=(4, 3),
        proprio_hidden_dims=[128],
        cnn_channels=[16, 32, 64],
        cnn_kernel_size=3,
        cnn_activation="elu",
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type="scalar",
        state_dependent_std=False,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
    )


@configclass
class LocoTransformerHistoryRoughAgentCfg(LocoTransformerHistoryAgentCfg):
    """PPO runner for the 777-D mixed-terrain curriculum task."""

    experiment_name = "loco_transformer_history_rough"
    resume = False
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="rsl_rl.algorithms.ppo:PPO",
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
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            mirror_loss_coeff=0.2,
            data_augmentation_func=compute_symmetric_states,
        ),
        rnd_cfg=None,
    )


@configclass
class LocoTransformerHistoryTeacherAgentCfg(LocoTransformerHistoryAgentCfg):
    """Clean-observation teacher runner for the ten-frame history task."""

    experiment_name = "loco_transformer_history_teacher"
    resume = False


@configclass
class LocoTransformerHistoryRoughTeacherAgentCfg(LocoTransformerHistoryRoughAgentCfg):
    """Clean-observation teacher runner for the rough-terrain history task."""

    experiment_name = "loco_transformer_history_rough_teacher"
    resume = False


@configclass
class LocoTransformerMLPAgentCfg(RslRlOnPolicyRunnerCfg):
    """Legacy pure-MLP PPO agent for the loco_transformer task (78-dim obs).

    This is a fallback for baseline experiments without height-scan input.
    Uses a 3-layer MLP policy: 256 → 128 → 64.
    """

    class_name = "OnPolicyRunner"
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 9001
    save_interval = 500
    experiment_name = "loco_transformer_mlp"
    run_name = ""
    logger = "tensorboard"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
    clip_actions = 100.0
    empirical_normalization = False
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}

    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="rsl_rl.algorithms.ppo:PPO",
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


@configclass
class LocoTransformerMLPTeacherAgentCfg(LocoTransformerMLPAgentCfg):
    """Clean-observation teacher runner for the proprioceptive MLP task."""

    experiment_name = "loco_transformer_mlp_teacher"
    resume = False
