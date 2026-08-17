# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for the loco_transformer task.

Based on the RPO-Flat reward and terrain design:
- 28-term reward function (1:1 port from RPO-Flat)
- Procedural gravel terrain (70% flat + 30% random rough, matching GRAVEL_TERRAINS_CFG)
- Additional feet scanners for feet_height reward
"""

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp

##
# Terrain generator — matching RPO-Flat's GRAVEL_TERRAINS_CFG
##

TERRAIN_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.3,
            noise_range=(-0.02, 0.04),
            noise_step=0.02,
            border_width=0.25,
        ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.7,
        ),
    },
)


##
# Scene
##


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with RPO robot."""

    # -- terrain: generator-based (matched to RPO-Flat GRAVEL_TERRAINS_CFG) --
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TERRAIN_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # robot — assigned by the concrete config class in __post_init__
    robot: ArticulationCfg = MISSING

    # contact sensor — captures all body contact forces
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # height scanner — on torso_link for terrain perception (cross-attention input)
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 5.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(2.0, 1.0)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # per-foot scanners for feet_height reward (matching RPO-Flat)
    left_feet_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.025, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=[0.12, 0.04]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    right_feet_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.025, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=[0.12, 0.04]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )

    # dome light
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP.

    Two observation groups:
    - policy: observations with noise (for actor training)
    - critic: observations without noise (for value estimation)

    Height scan from raycaster on torso_link (21×11 = 231 rays, GridPatternCfg 0.1 m resolution).
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy group (with noise).

        - base_ang_vel (3)
        - projected_gravity (3)
        - velocity_commands (3)
        - joint_pos (23)
        - joint_vel (23)
        - last_action (23)
        - height_scan (231)  ← raycaster
        Total: 309 dims
        """

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.03, n_max=0.03),  # matched to RPO-Flat
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.75, n_max=1.75),  # matched to RPO-Flat
        )
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-5.0, 5.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for the critic group (without noise).

        Same structure as PolicyCfg but without observation noise.
        """

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            clip=(-5.0, 5.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP.

    Uses uniform velocity commands matching RPO-Flat ranges.
    """

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.6, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.57, 1.57),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class RewardsCfg:
    """Reward terms — 1:1 port from RPO-Flat (28 terms)."""

    # -- task rewards: velocity tracking --
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # -- regularization penalties --
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)
    energy = RewTerm(func=mdp.energy, weight=-1.0e-4)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-2.0e-4)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-2.0e-2)
    action_smoothness_l2 = RewTerm(func=mdp.action_smoothness_l2, weight=-2.0e-2)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "torso_link",
                    ".*_thigh_yaw_link",
                    ".*_thigh_roll_link",
                    ".*_arm_pitch_link",
                    ".*_arm_roll_link",
                    ".*_arm_yaw_link",
                    ".*_elbow_pitch_link",
                    ".*_elbow_yaw_link",
                ],
            ),
        },
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    termination_penalty = RewTerm(func=mdp.termination_penalty, weight=-200.0)

    # -- gait / foot behavior rewards --
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    feet_force = RewTerm(
        func=mdp.feet_force,
        weight=-3.0e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )
    feet_orientation_l2 = RewTerm(
        func=mdp.feet_orientation_l2,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    feet_distance = RewTerm(
        func=mdp.body_distance_y,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "min": 0.16,
            "max": 0.50,
        },
    )
    knee_distance = RewTerm(
        func=mdp.body_distance_y,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_knee_link"),
            "min": 0.18,
            "max": 0.35,
        },
    )
    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "command_name": "base_velocity",
        },
    )
    feet_height = RewTerm(
        func=mdp.feet_height,
        weight=0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "sensor_cfg1": SceneEntityCfg("left_feet_scanner"),
            "sensor_cfg2": SceneEntityCfg("right_feet_scanner"),
            "ankle_height": 0.04,
            "threshold": 0.02,
        },
    )

    # -- joint posture / deviation rewards --
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.03,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_thigh_yaw_joint", ".*_thigh_roll_joint"]
            ),
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*torso_joint",
                    ".*_arm_roll_joint",
                    ".*_arm_yaw_joint",
                    ".*_elbow_pitch_joint",
                    ".*_elbow_yaw_joint",
                ],
            ),
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.06,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_arm_pitch_joint"]),
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_thigh_pitch_joint",
                    ".*_knee_joint",
                    ".*_ankle_pitch_joint",
                    ".*_ankle_roll_joint",
                ],
            ),
        },
    )

    # -- posture / standing rewards --
    upward = RewTerm(func=mdp.upward, weight=0.4)
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-0.2,
        params={
            "command_name": "base_velocity",
            "pos_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_arm_.*",
                    ".*_elbow_.*",
                    ".*torso.*",
                    ".*_thigh_.*",
                    ".*_knee_.*",
                    ".*_ankle_.*",
                ],
            ),
            "vel_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_arm_.*",
                    ".*_elbow_.*",
                    ".*torso.*",
                    ".*_thigh_.*",
                    ".*_knee_.*",
                    ".*_ankle_.*",
                ],
            ),
            "pos_weight": 1.0,
            "vel_weight": 0.04,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms — matched to RPO-Flat."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # terminate on torso + thigh (yaw/roll) contact — matching RPO-Flat
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "torso_link",
                    ".*_thigh_yaw_link",
                    ".*_thigh_roll_link",
                ],
            ),
            "threshold": 1.0,
        },
    )

    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(60.0)},
    )

    root_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.3},
    )


@configclass
class EventCfg:
    """Configuration for domain randomization events.

    Matched to RPO-Flat ranges.
    """

    # startup randomizations — applied once at environment creation

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link", "base_link"]),
            "mass_distribution_params": (-3.0, 3.0),
            "operation": "add",
        },
    )

    scale_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_.*_link", "right_.*_link"]),
            "mass_distribution_params": (0.9, 1.1),  # matched to RPO-Flat
            "operation": "scale",
        },
    )

    randomize_rigid_body_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link", "base_link"]),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.025, 0.025), "z": (-0.05, 0.05)},
        },
    )

    scale_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"]),
            "stiffness_distribution_params": (0.9, 1.1),  # matched to RPO-Flat
            "damping_distribution_params": (0.9, 1.1),  # matched to RPO-Flat
            "operation": "scale",
        },
    )

    scale_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"]),
            "friction_distribution_params": (1.0, 1.0),
            "armature_distribution_params": (0.5, 1.5),  # matched to RPO-Flat
            "operation": "scale",
        },
    )

    # reset events — applied at every episode reset

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.2, 0.2),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # periodic events

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.2, 0.2),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            },
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP.

    Empty for the Flat variant — matches RPO-Flat (GRAVEL_TERRAINS_CFG has
    ``curriculum=False``). To enable terrain-based curriculum, set
    ``TERRAIN_CFG.curriculum = True`` and add appropriate curriculum terms.
    """

    pass


##
# Environment configuration
##


@configclass
class LocoTransformerEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the loco_transformer locomotion environment."""

    # Scene
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)

    # MDP
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization to set simulation and physics parameters."""
        # general settings
        self.decimation = 4  # policy runs at 50 Hz
        self.episode_length_s = 20.0

        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.sim.dt
        if self.scene.left_feet_scanner is not None:
            self.scene.left_feet_scanner.update_period = self.sim.dt
        if self.scene.right_feet_scanner is not None:
            self.scene.right_feet_scanner.update_period = self.sim.dt
