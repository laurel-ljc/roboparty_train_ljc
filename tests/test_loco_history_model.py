"""Network construction tests for the 777-D history observation."""

import pytest

torch = pytest.importorskip("torch")
TensorDict = pytest.importorskip("tensordict").TensorDict
CrossAttentionActorCritic = pytest.importorskip(
    "rsl_rl.modules.actor_critic_cross_attn"
).CrossAttentionActorCritic


def test_history_cross_attention_actor_and_critic_shapes():
    batch_size = 3
    observations = TensorDict(
        {
            "policy": torch.randn(batch_size, 777),
            "critic": torch.randn(batch_size, 777),
        },
        batch_size=[batch_size],
    )
    model = CrossAttentionActorCritic(
        obs=observations,
        obs_groups={"policy": ["policy"], "critic": ["critic"]},
        num_actions=23,
        actor_perception_range=(546, 777),
        critic_perception_range=(546, 777),
        height_map_shape=(11, 21),
        embed_dim=64,
        num_heads=8,
        grid_size=(4, 3),
        proprio_hidden_dims=[128],
        cnn_channels=[16, 32, 64],
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
    )

    actions = model.act_inference(observations)
    values = model.evaluate(observations)

    assert model.num_actor_obs == 777
    assert model.num_critic_obs == 777
    assert model.num_actor_proprio == 546
    assert model.num_critic_proprio == 546
    assert actions.shape == (batch_size, 23)
    assert values.shape == (batch_size, 1)
