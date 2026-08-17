"""Configuration regression tests for Transformer, MLP, and playback variants."""

import pytest

pytest.importorskip("pxr", reason="Isaac Sim Python runtime is required for config tests")
pytest.importorskip("isaaclab")

from loco_transformer.agents.loco_transformer_agent_cfg import (
    LocoTransformerAgentCfg,
    LocoTransformerMLPAgentCfg,
)
from loco_transformer.rpo_loco_transformer_env_cfg import (
    RPOLocoMLPEnvCfg,
    RPOLocoMLPEnvCfg_PLAY,
    RPOLocoTransformerEnvCfg,
    RPOLocoTransformerEnvCfg_PLAY,
)
from robolab.tasks.direct.base.rpo_env_cfg import RPORewardCfg
from robolab.tasks.direct.base.agents.rpo_agent_cfg import RPOFlatAgentCfg


def _reward_weights(rewards_cfg):
    return {
        name: value.weight
        for name in dir(rewards_cfg)
        if not name.startswith("_")
        and (value := getattr(rewards_cfg, name)) is not None
        and hasattr(value, "weight")
        and hasattr(value, "func")
    }


def test_transformer_observation_layout_and_sensor_periods():
    env_cfg = RPOLocoTransformerEnvCfg()
    agent_cfg = LocoTransformerAgentCfg()

    assert agent_cfg.policy.actor_perception_range == (78, 309)
    assert agent_cfg.policy.critic_perception_range == (78, 309)
    assert agent_cfg.policy.height_map_shape == (11, 21)
    assert env_cfg.observations.policy.height_scan is not None
    assert env_cfg.observations.critic.height_scan is not None
    assert env_cfg.scene.height_scanner is not None
    assert env_cfg.scene.contact_forces.update_period == pytest.approx(0.005)
    assert env_cfg.scene.height_scanner.update_period == pytest.approx(0.02)
    assert env_cfg.scene.left_feet_scanner.update_period == pytest.approx(0.02)
    assert env_cfg.scene.right_feet_scanner.update_period == pytest.approx(0.02)


def test_mlp_variant_has_exactly_the_78_dimensional_terms():
    env_cfg = RPOLocoMLPEnvCfg()
    agent_cfg = LocoTransformerMLPAgentCfg()

    assert env_cfg.observations.policy.height_scan is None
    assert env_cfg.observations.critic.height_scan is None
    assert env_cfg.scene.height_scanner is None
    assert agent_cfg.experiment_name == "loco_transformer_mlp"


@pytest.mark.parametrize(
    "cfg_type", [RPOLocoTransformerEnvCfg_PLAY, RPOLocoMLPEnvCfg_PLAY]
)
def test_play_variants_disable_all_randomization(cfg_type):
    cfg = cfg_type()
    disabled_events = (
        "physics_material",
        "add_base_mass",
        "scale_link_mass",
        "randomize_rigid_body_com",
        "scale_actuator_gains",
        "scale_joint_parameters",
        "reset_base",
        "reset_robot_joints",
        "push_robot",
    )

    assert all(getattr(cfg.events, name) is None for name in disabled_events)
    assert cfg.observations.policy.enable_corruption is False


def test_reward_names_weights_contacts_and_terminations_match_rpo_flat():
    loco_cfg = RPOLocoTransformerEnvCfg()
    direct_rewards = RPORewardCfg()

    assert _reward_weights(loco_cfg.rewards) == _reward_weights(direct_rewards)
    assert len(_reward_weights(loco_cfg.rewards)) == 29
    assert (
        loco_cfg.rewards.undesired_contacts.params["sensor_cfg"].body_names
        == "(?!.*ankle_roll.*).*"
    )
    assert getattr(loco_cfg.terminations, "bad_orientation", None) is None
    assert getattr(loco_cfg.terminations, "root_height", None) is None


def test_direct_rpo_normalization_behavior_is_explicitly_legacy_compatible():
    policy_cfg = RPOFlatAgentCfg().policy

    assert policy_cfg.actor_obs_normalization is False
    assert policy_cfg.critic_obs_normalization is False
