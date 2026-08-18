"""Network construction tests for the 777-D history observation."""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
TensorDict = pytest.importorskip("tensordict").TensorDict
CrossAttentionActorCritic = pytest.importorskip(
    "rsl_rl.modules.actor_critic_cross_attn"
).CrossAttentionActorCritic

_symmetry_spec = importlib.util.spec_from_file_location(
    "loco_history_rough_symmetry",
    Path(__file__).parents[1] / "loco_transformer" / "mdp" / "symmetry.py",
)
symmetry = importlib.util.module_from_spec(_symmetry_spec)
_symmetry_spec.loader.exec_module(symmetry)


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


def test_history_rough_mirror_is_an_involution():
    observations = torch.randn(4, 777)
    actions = torch.randn(4, 23)

    assert torch.equal(
        symmetry.mirror_history_observation(symmetry.mirror_history_observation(observations)),
        observations,
    )
    assert torch.equal(symmetry.mirror_actions(symmetry.mirror_actions(actions)), actions)


def test_history_rough_mirror_preserves_time_and_maps_every_term():
    observations = torch.arange(777, dtype=torch.float32).unsqueeze(0)
    mirrored = symmetry.mirror_history_observation(observations)
    joint_indices = torch.tensor(symmetry.JOINT_MIRROR_INDICES)
    joint_signs = torch.tensor(symmetry.JOINT_MIRROR_SIGNS)

    assert torch.equal(
        mirrored[..., 0:30].reshape(1, 10, 3),
        observations[..., 0:30].reshape(1, 10, 3) * torch.tensor([-1, 1, -1]),
    )
    assert torch.equal(
        mirrored[..., 30:60].reshape(1, 10, 3),
        observations[..., 30:60].reshape(1, 10, 3) * torch.tensor([1, -1, 1]),
    )
    assert torch.equal(mirrored[..., 60:63], observations[..., 60:63] * torch.tensor([1, -1, -1]))
    assert torch.equal(
        mirrored[..., 63:293].reshape(1, 10, 23),
        observations[..., 63:293].reshape(1, 10, 23)[..., joint_indices] * joint_signs,
    )
    assert torch.equal(
        mirrored[..., 293:523].reshape(1, 10, 23),
        observations[..., 293:523].reshape(1, 10, 23)[..., joint_indices] * joint_signs,
    )
    assert torch.equal(
        mirrored[..., 523:546],
        observations[..., 523:546][..., joint_indices] * joint_signs,
    )
    assert torch.equal(
        mirrored[..., 546:777].reshape(1, 11, 21),
        observations[..., 546:777].reshape(1, 11, 21).flip(dims=(-2,)),
    )


def test_history_rough_tensordict_augmentation_order_and_input_integrity():
    policy = torch.randn(3, 777)
    critic = torch.randn(3, 777)
    actions = torch.randn(3, 23)
    observations = TensorDict(
        {"policy": policy.clone(), "critic": critic.clone()},
        batch_size=[3],
    )
    observations_before = observations.clone()
    actions_before = actions.clone()

    observations_augmented, actions_augmented = symmetry.compute_symmetric_states(
        env=None,
        obs=observations,
        actions=actions,
    )

    assert observations_augmented.batch_size == torch.Size([6])
    assert torch.equal(observations_augmented["policy"][:3], policy)
    assert torch.equal(
        observations_augmented["policy"][3:], symmetry.mirror_history_observation(policy)
    )
    assert torch.equal(observations_augmented["critic"][:3], critic)
    assert torch.equal(
        observations_augmented["critic"][3:], symmetry.mirror_history_observation(critic)
    )
    assert torch.equal(actions_augmented[:3], actions)
    assert torch.equal(actions_augmented[3:], symmetry.mirror_actions(actions))
    assert torch.equal(observations["policy"], observations_before["policy"])
    assert torch.equal(observations["critic"], observations_before["critic"])
    assert torch.equal(actions, actions_before)


def test_history_rough_symmetry_rejects_stale_layout_dimensions():
    with pytest.raises(ValueError, match="last dimension 777"):
        symmetry.mirror_history_observation(torch.zeros(2, 776))
    with pytest.raises(ValueError, match="last dimension 23"):
        symmetry.mirror_actions(torch.zeros(2, 22))


def test_history_cross_attention_mirror_loss_has_finite_gradients():
    batch_size = 2
    policy = torch.randn(batch_size, 777)
    mirrored_policy = symmetry.mirror_history_observation(policy)
    observations = TensorDict(
        {
            "policy": torch.cat((policy, mirrored_policy), dim=0),
            "critic": torch.cat((policy, mirrored_policy), dim=0),
        },
        batch_size=[2 * batch_size],
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

    predicted_actions = model.act_inference(observations)
    expected_mirrored_actions = symmetry.mirror_actions(predicted_actions[:batch_size]).detach()
    mirror_loss = torch.nn.functional.mse_loss(
        predicted_actions[batch_size:],
        expected_mirrored_actions,
    )
    mirror_loss.backward()

    assert torch.isfinite(mirror_loss)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.actor.parameters()
    )
