"""Configuration regression tests for Transformer, MLP, and playback variants."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

gym = pytest.importorskip("gymnasium")
pytest.importorskip("pxr", reason="Isaac Sim Python runtime is required for config tests")
pytest.importorskip("isaaclab")

from isaaclab.terrains import TerrainImporter

from loco_transformer.agents.loco_transformer_agent_cfg import (
    LocoTransformerAgentCfg,
    LocoTransformerHistoryAgentCfg,
    LocoTransformerHistoryRoughAgentCfg,
    LocoTransformerMLPAgentCfg,
)
from loco_transformer.rpo_loco_transformer_env_cfg import (
    RPOLocoMLPEnvCfg,
    RPOLocoMLPEnvCfg_PLAY,
    RPOLocoTransformerEnvCfg,
    RPOLocoTransformerEnvCfg_PLAY,
    RPOLocoTransformerHistoryEnvCfg,
    RPOLocoTransformerHistoryEnvCfg_PLAY,
    RPOLocoTransformerHistoryRoughEnvCfg,
    RPOLocoTransformerHistoryRoughEnvCfg_PLAY,
)
from loco_transformer.mdp.curriculums import terrain_levels_vel
from loco_transformer.mdp.symmetry import compute_symmetric_states
from loco_transformer.terrain_generator_cfg import (
    LOCO_HISTORY_ROUGH_TERRAINS_CFG,
    BalancedDirectionTerrainGenerator,
    terrain_difficulty_for_level,
)
from robolab.tasks.direct.base.rpo_env_cfg import RPORewardCfg
from robolab.tasks.direct.base.agents.rpo_agent_cfg import RPOFlatAgentCfg
from rsl_rl.rsl_rl.modules.actor_critic_cross_attn import CrossAttentionActorCritic


HEIGHT_MAP_SHAPE = (11, 21)


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
    assert env_cfg.scene.height_scanner.ray_alignment == "yaw"
    assert env_cfg.scene.height_scanner.pattern_cfg.ordering == "xy"
    assert env_cfg.scene.height_scanner.pattern_cfg.resolution == pytest.approx(0.1)
    assert env_cfg.scene.height_scanner.pattern_cfg.size == (2.0, 1.0)
    assert env_cfg.scene.contact_forces.update_period == pytest.approx(0.005)
    assert env_cfg.scene.height_scanner.update_period == pytest.approx(0.02)
    assert env_cfg.scene.left_feet_scanner.update_period == pytest.approx(0.02)
    assert env_cfg.scene.right_feet_scanner.update_period == pytest.approx(0.02)


def _height_scanner_ray_starts():
    """Generate rays with the exact GridPatternCfg used by the current task."""
    pattern_cfg = RPOLocoTransformerEnvCfg().scene.height_scanner.pattern_cfg
    starts, _ = pattern_cfg.func(pattern_cfg, "cpu")
    return starts


def _policy_height_map(height_scan):
    """Exercise the current policy's real ``view(-1, 11, 21)`` path."""
    policy_shape = SimpleNamespace(H=HEIGHT_MAP_SHAPE[0], W=HEIGHT_MAP_SHAPE[1])
    _, height_map = CrossAttentionActorCritic._split_flat_obs(
        policy_shape, height_scan, (0, height_scan.shape[-1])
    )
    return height_map


def _isolated_platform_map(x, y):
    starts = _height_scanner_ray_starts()
    distance = (starts[:, 0] - x).abs() + (starts[:, 1] - y).abs()
    ray_index = int(distance.argmin())
    assert distance[ray_index] < 1.0e-5, f"({x}, {y}) is not on the configured scan grid"

    # Isaac Lab height_scan = sensor_z - hit_z - offset.  A raised platform
    # therefore appears below the flat-ground baseline in observation value.
    height_scan = torch.zeros(1, starts.shape[0])
    height_scan[0, ray_index] = -1.0
    return _policy_height_map(height_scan), ray_index


def test_height_map_reshape_has_y_rows_and_x_columns():
    """Row 0 is right; last row left; column 0 back; last column front."""
    starts = _height_scanner_ray_starts().view(*HEIGHT_MAP_SHAPE, 3)

    assert tuple(starts.shape) == (11, 21, 3)
    assert torch.allclose(starts[:, :, 0], starts[0:1, :, 0].expand(11, -1))
    assert torch.allclose(starts[:, :, 1], starts[:, 0:1, 1].expand(-1, 21))
    assert starts[0, 0, 1].item() == pytest.approx(-0.5)  # first row: right
    assert starts[-1, 0, 1].item() == pytest.approx(+0.5)  # last row: left
    assert starts[0, 0, 0].item() == pytest.approx(-1.0)  # first column: back
    assert starts[0, -1, 0].item() == pytest.approx(+1.0)  # last column: front


@pytest.mark.parametrize(
    ("name", "x", "y", "expected_row", "expected_column"),
    [
        ("front", +0.6, 0.0, 5, 16),
        ("back", -0.6, 0.0, 5, 4),
        ("left", 0.0, +0.3, 8, 10),
        ("right", 0.0, -0.3, 2, 10),
        ("front_left", +0.6, +0.3, 8, 16),
    ],
)
def test_isolated_platform_keeps_direction_after_policy_reshape(
    name, x, y, expected_row, expected_column
):
    height_map, ray_index = _isolated_platform_map(x, y)
    nonzero = torch.nonzero(height_map[0], as_tuple=False)

    assert ray_index == expected_row * HEIGHT_MAP_SHAPE[1] + expected_column, name
    assert nonzero.tolist() == [[expected_row, expected_column]], name
    assert height_map[0, expected_row, expected_column].item() == pytest.approx(-1.0)


@pytest.mark.parametrize("yaw_degrees", [0.0, 37.0, 90.0, -123.0])
def test_yaw_alignment_keeps_front_left_in_the_same_robot_relative_cell(yaw_degrees):
    """Yaw changes the obstacle's world coordinates, not its policy-map cell."""
    starts = _height_scanner_ray_starts()
    local_target = torch.tensor([+0.6, +0.3])
    yaw = torch.deg2rad(torch.tensor(yaw_degrees))
    rotation = torch.tensor(
        [[torch.cos(yaw), -torch.sin(yaw)], [torch.sin(yaw), torch.cos(yaw)]]
    )

    world_starts = starts[:, :2] @ rotation.T
    world_target = local_target @ rotation.T
    hit_index = int(torch.linalg.vector_norm(world_starts - world_target, dim=1).argmin())

    height_scan = torch.zeros(1, starts.shape[0])
    height_scan[0, hit_index] = -1.0
    height_map = _policy_height_map(height_scan)
    assert torch.nonzero(height_map[0], as_tuple=False).tolist() == [[8, 16]]


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
    ("task_id", "env_cfg_name"),
    [
        ("RPO-Loco-Transformer-History-Rough", "RPOLocoTransformerHistoryRoughEnvCfg"),
        ("RPO-Loco-Transformer-History-Rough-Play", "RPOLocoTransformerHistoryRoughEnvCfg_PLAY"),
    ],
)
def test_history_rough_tasks_are_registered(task_id, env_cfg_name):
    spec = gym.spec(task_id)

    assert spec.kwargs["env_cfg_entry_point"].endswith(f":{env_cfg_name}")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(":LocoTransformerHistoryRoughAgentCfg")


def test_history_rough_cfg_preserves_model_and_mdp_but_enables_curriculum():
    history_cfg = RPOLocoTransformerHistoryEnvCfg()
    rough_cfg = RPOLocoTransformerHistoryRoughEnvCfg()
    agent_cfg = LocoTransformerHistoryRoughAgentCfg()
    terrain_cfg = rough_cfg.scene.terrain.terrain_generator

    assert agent_cfg.policy.actor_perception_range == (546, 777)
    assert agent_cfg.policy.critic_perception_range == (546, 777)
    assert agent_cfg.experiment_name == "loco_transformer_history_rough"
    assert agent_cfg.resume is False
    assert _reward_weights(rough_cfg.rewards) == _reward_weights(history_cfg.rewards)
    assert len(_reward_weights(rough_cfg.rewards)) == 29
    assert rough_cfg.events.to_dict() == history_cfg.events.to_dict()
    assert rough_cfg.terminations.to_dict() == history_cfg.terminations.to_dict()

    assert terrain_cfg.curriculum is True
    assert terrain_cfg.class_type is BalancedDirectionTerrainGenerator
    assert terrain_cfg.seed == 42
    assert terrain_cfg.num_rows == 10
    assert terrain_cfg.num_cols == 20
    assert terrain_cfg.size == (8.0, 8.0)
    assert terrain_cfg.border_width == 20.0
    assert terrain_cfg.horizontal_scale == pytest.approx(0.05)
    assert terrain_cfg.vertical_scale == pytest.approx(0.005)
    assert rough_cfg.scene.terrain.max_init_terrain_level == 2
    assert rough_cfg.curriculum.terrain_levels is not None

    proportions = {name: term.proportion for name, term in terrain_cfg.sub_terrains.items()}
    assert sum(proportions.values()) == pytest.approx(1.0)
    assert proportions["flat"] == pytest.approx(0.20)
    assert proportions["rough"] == pytest.approx(0.30)
    assert proportions["slope_up"] + proportions["slope_down"] == pytest.approx(0.25)
    assert proportions["stairs_up"] + proportions["stairs_down"] == pytest.approx(0.25)
    assert proportions["slope_up"] == proportions["slope_down"]
    assert proportions["stairs_up"] == proportions["stairs_down"]


def test_only_history_rough_agent_enables_robolab_rough_symmetry():
    base_symmetry = LocoTransformerAgentCfg().algorithm.symmetry_cfg
    history_symmetry = LocoTransformerHistoryAgentCfg().algorithm.symmetry_cfg
    rough_symmetry = LocoTransformerHistoryRoughAgentCfg().algorithm.symmetry_cfg
    mlp_symmetry = LocoTransformerMLPAgentCfg().algorithm.symmetry_cfg

    assert base_symmetry is None
    assert history_symmetry is None
    assert mlp_symmetry is None
    assert rough_symmetry is not None
    assert rough_symmetry.use_data_augmentation is True
    assert rough_symmetry.use_mirror_loss is True
    assert rough_symmetry.mirror_loss_coeff == pytest.approx(0.2)
    assert rough_symmetry.data_augmentation_func is compute_symmetric_states


def test_history_rough_play_is_deterministic_and_highest_difficulty():
    cfg = RPOLocoTransformerHistoryRoughEnvCfg_PLAY()
    terrain_cfg = cfg.scene.terrain.terrain_generator

    assert cfg.curriculum.terrain_levels is None
    assert terrain_cfg.curriculum is False
    assert terrain_cfg.difficulty_range == (1.0, 1.0)
    assert terrain_cfg.seed == 42
    assert terrain_cfg.num_rows == 1
    assert cfg.scene.terrain.max_init_terrain_level == 0


@pytest.mark.parametrize(
    "cfg_type",
    [
        RPOLocoTransformerEnvCfg_PLAY,
        RPOLocoTransformerHistoryEnvCfg_PLAY,
        RPOLocoTransformerHistoryRoughEnvCfg_PLAY,
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
        "push_robot",
    )

    assert all(getattr(cfg.events, name) is None for name in disabled_events)
    assert cfg.events.reset_base.params["pose_range"] == {}
    assert cfg.events.reset_base.params["velocity_range"] == {}
    assert cfg.events.reset_robot_joints.params["position_range"] == (1.0, 1.0)
    assert cfg.events.reset_robot_joints.params["velocity_range"] == (0.0, 0.0)
    assert cfg.observations.policy.enable_corruption is False
    assert cfg.commands.base_velocity.debug_vis is False
    assert cfg.commands.base_velocity.ranges.lin_vel_x == (0.0, 0.0)
    assert cfg.viewer.origin_type == "asset_root"
    assert cfg.viewer.asset_name == "robot"


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


def _terrain_mesh_at(name: str, difficulty: float):
    cfg = copy.deepcopy(LOCO_HISTORY_ROUGH_TERRAINS_CFG.sub_terrains[name])
    cfg.size = LOCO_HISTORY_ROUGH_TERRAINS_CFG.size
    if hasattr(cfg, "horizontal_scale"):
        cfg.horizontal_scale = LOCO_HISTORY_ROUGH_TERRAINS_CFG.horizontal_scale
        cfg.vertical_scale = LOCO_HISTORY_ROUGH_TERRAINS_CFG.vertical_scale
        cfg.slope_threshold = LOCO_HISTORY_ROUGH_TERRAINS_CFG.slope_threshold
    meshes, origin = cfg.function(difficulty, cfg)
    return meshes[0], origin


def test_rough_slope_and_stairs_geometry_increases_with_difficulty():
    rough_easy, _ = _terrain_mesh_at("rough", 0.0)
    rough_hard, _ = _terrain_mesh_at("rough", 1.0)
    slope_easy, _ = _terrain_mesh_at("slope_up", 0.0)
    slope_hard, _ = _terrain_mesh_at("slope_up", 1.0)
    stairs_easy, _ = _terrain_mesh_at("stairs_up", 0.0)
    stairs_hard, _ = _terrain_mesh_at("stairs_up", 1.0)

    assert np.ptp(rough_hard.vertices[:, 2]) > np.ptp(rough_easy.vertices[:, 2])
    assert np.ptp(slope_hard.vertices[:, 2]) > np.ptp(slope_easy.vertices[:, 2])
    assert np.ptp(stairs_hard.vertices[:, 2]) > np.ptp(stairs_easy.vertices[:, 2])

    center = (
        (rough_hard.vertices[:, 0] > 3.2)
        & (rough_hard.vertices[:, 0] < 4.8)
        & (rough_hard.vertices[:, 1] > 3.2)
        & (rough_hard.vertices[:, 1] < 4.8)
    )
    assert center.any()
    assert np.max(np.abs(rough_hard.vertices[center, 2])) == pytest.approx(0.0)


def test_up_down_terrain_pairs_have_opposite_height_direction():
    slope_up, _ = _terrain_mesh_at("slope_up", 1.0)
    slope_down, _ = _terrain_mesh_at("slope_down", 1.0)
    stairs_up, _ = _terrain_mesh_at("stairs_up", 1.0)
    stairs_down, _ = _terrain_mesh_at("stairs_down", 1.0)

    assert slope_up.vertices[:, 2].max() > 0.0
    assert slope_down.vertices[:, 2].min() < 0.0
    assert stairs_up.vertices[:, 2].max() > 0.0
    assert stairs_down.vertices[:, 2].min() < 0.0


def test_level_midpoint_difficulties_are_monotonic():
    difficulties = [terrain_difficulty_for_level(level) for level in range(10)]
    assert difficulties == sorted(difficulties)
    assert difficulties[0] == pytest.approx(0.05)
    assert difficulties[-1] == pytest.approx(0.95)


class _CurriculumScene:
    def __init__(self, asset, terrain, origins):
        self._asset = asset
        self.terrain = terrain
        self.env_origins = origins

    def __getitem__(self, name):
        assert name == "robot"
        return self._asset


def test_distance_curriculum_moves_success_failure_and_middle_as_expected():
    root_pos = torch.tensor([[5.0, 0.0, 0.0], [0.5, 0.0, 0.0], [3.0, 0.0, 0.0]])
    asset = SimpleNamespace(data=SimpleNamespace(root_pos_w=root_pos))
    terrain = SimpleNamespace(
        cfg=SimpleNamespace(terrain_generator=SimpleNamespace(size=(8.0, 8.0))),
        terrain_levels=torch.tensor([1, 1, 1]),
    )
    recorded = {}

    def update_env_origins(env_ids, move_up, move_down):
        recorded["ids"] = env_ids
        recorded["up"] = move_up.clone()
        recorded["down"] = move_down.clone()

    terrain.update_env_origins = update_env_origins
    env = SimpleNamespace(
        scene=_CurriculumScene(asset, terrain, torch.zeros(3, 3)),
        command_manager=SimpleNamespace(
            get_command=lambda name: torch.tensor(
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.1, 0.0, 0.0]]
            )
        ),
        max_episode_length_s=20.0,
    )

    mean_level = terrain_levels_vel(env, torch.arange(3))

    assert recorded["up"].tolist() == [True, False, False]
    assert recorded["down"].tolist() == [False, True, False]
    assert mean_level.item() == pytest.approx(1.0)


def test_isaac_level_update_clamps_zero_and_resamples_after_top():
    fake = SimpleNamespace(
        terrain_origins=torch.zeros(3, 1, 3),
        terrain_levels=torch.tensor([0, 2]),
        terrain_types=torch.tensor([0, 0]),
        env_origins=torch.zeros(2, 3),
        max_terrain_level=3,
        device="cpu",
    )

    TerrainImporter.update_env_origins(
        fake,
        torch.tensor([0, 1]),
        torch.tensor([False, True]),
        torch.tensor([True, False]),
    )

    assert fake.terrain_levels[0].item() == 0
    assert 0 <= fake.terrain_levels[1].item() < 3
