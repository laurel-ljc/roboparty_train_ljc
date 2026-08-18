"""Configuration regression tests for Transformer, MLP, and playback variants."""

import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("pxr", reason="Isaac Sim Python runtime is required for config tests")
pytest.importorskip("isaaclab")

from loco_transformer.agents.loco_transformer_agent_cfg import (
    LocoTransformerAgentCfg,
    LocoTransformerHistoryAgentCfg,
    LocoTransformerMLPAgentCfg,
)
from loco_transformer.rpo_loco_transformer_env_cfg import (
    RPOLocoMLPEnvCfg,
    RPOLocoMLPEnvCfg_PLAY,
    RPOLocoTransformerEnvCfg,
    RPOLocoTransformerEnvCfg_PLAY,
    RPOLocoTransformerHistoryEnvCfg,
    RPOLocoTransformerHistoryEnvCfg_PLAY,
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


def test_history_variant_has_exactly_the_selected_ten_frame_terms():
    original_cfg = RPOLocoTransformerEnvCfg()
    history_cfg = RPOLocoTransformerHistoryEnvCfg()
    agent_cfg = LocoTransformerHistoryAgentCfg()

    historical_terms = ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel")
    current_terms = ("velocity_commands", "actions", "height_scan")
    for group_name in ("policy", "critic"):
        original_group = getattr(original_cfg.observations, group_name)
        history_group = getattr(history_cfg.observations, group_name)
        for term_name in historical_terms:
            assert getattr(history_group, term_name).history_length == 10
            assert getattr(history_group, term_name).flatten_history_dim is True
            assert getattr(original_group, term_name).history_length == 0
        for term_name in current_terms:
            assert getattr(history_group, term_name).history_length == 0

    assert agent_cfg.policy.actor_perception_range == (546, 777)
    assert agent_cfg.policy.critic_perception_range == (546, 777)
    assert agent_cfg.policy.height_map_shape == (11, 21)
    assert agent_cfg.experiment_name == "loco_transformer_history"
    assert agent_cfg.resume is False


def test_history_variant_preserves_original_task_configuration():
    original_cfg = RPOLocoTransformerEnvCfg()
    history_cfg = RPOLocoTransformerHistoryEnvCfg()

    assert _reward_weights(history_cfg.rewards) == _reward_weights(original_cfg.rewards)
    for field_name in (
        "curriculum",
        "size",
        "border_width",
        "num_rows",
        "num_cols",
        "horizontal_scale",
        "vertical_scale",
    ):
        assert getattr(history_cfg.scene.terrain.terrain_generator, field_name) == getattr(
            original_cfg.scene.terrain.terrain_generator, field_name
        )
    assert history_cfg.scene.height_scanner.update_period == original_cfg.scene.height_scanner.update_period
    assert history_cfg.decimation == original_cfg.decimation
    assert history_cfg.sim.dt == original_cfg.sim.dt


@pytest.mark.parametrize(
    ("task_id", "env_cfg_name"),
    [
        ("RPO-Loco-Transformer-History", "RPOLocoTransformerHistoryEnvCfg"),
        ("RPO-Loco-Transformer-History-Play", "RPOLocoTransformerHistoryEnvCfg_PLAY"),
    ],
)
def test_history_tasks_are_registered(task_id, env_cfg_name):
    spec = gym.spec(task_id)

    assert spec.kwargs["env_cfg_entry_point"].endswith(f":{env_cfg_name}")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(":LocoTransformerHistoryAgentCfg")


@pytest.mark.parametrize(
    "cfg_type",
    [
        RPOLocoTransformerEnvCfg_PLAY,
        RPOLocoTransformerHistoryEnvCfg_PLAY,
        RPOLocoMLPEnvCfg_PLAY,
    ],
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
