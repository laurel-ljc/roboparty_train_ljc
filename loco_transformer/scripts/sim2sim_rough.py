# Copyright (c) 2025-2026, Loco-Transformer Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""MuJoCo Sim2Sim playback on a procedural History-Rough-style course.

This is the rough-terrain counterpart of :mod:`sim2sim`.  Policy loading,
309/777-D observations, yaw-aligned height scanning, PD control, keyboard
commands, and the chase camera are shared with that script.  Only the MuJoCo
world is replaced by four wide, parallel test lanes on one flat ground plane:
ascending/descending stairs, a gentle up/down ramp, a rolling/cambered road,
and plum-blossom posts.  Every lane starts and ends at ground height.

Example::

    python loco_transformer/scripts/sim2sim_rough.py \
        --load_model logs/rsl_rl/loco_transformer/RUN/exported/policy.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

try:
    from . import sim2sim as base
except ImportError:  # Direct execution: python loco_transformer/scripts/...
    import sim2sim as base


GROUND_Z = 0.0
LANE_HALF_WIDTH = 2.0
OBSTACLE_START_X = 3.0
LANE_CENTERS = {
    "stairs": 7.5,
    "slope": 2.5,
    "rough": -2.5,
    "plum": -7.5,
}
TERRAIN_FRICTION = (1.0, 0.005, 0.0001)

COLOR_ROUGH = (0.42, 0.35, 0.27, 1.0)
COLOR_RAMP = (0.45, 0.50, 0.58, 1.0)
COLOR_STAIRS = (0.62, 0.45, 0.28, 1.0)
COLOR_POSTS = (0.34, 0.46, 0.54, 1.0)


def _add_box(
    spec: mujoco.MjSpec,
    name: str,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    rgba: tuple[float, float, float, float],
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Add a collidable group-0 box so the policy height rays can see it."""
    spec.worldbody.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=pos,
        size=size,
        euler=euler,
        rgba=rgba,
        friction=TERRAIN_FRICTION,
        condim=4,
        contype=1,
        conaffinity=15,
        group=0,
    )


def _add_solid_box(
    spec: mujoco.MjSpec,
    name: str,
    x_center: float,
    y_center: float,
    half_length: float,
    half_width: float,
    top_z: float,
    rgba: tuple[float, float, float, float],
) -> None:
    """Add a step extending from the ground plane to the requested top height."""
    if top_z <= GROUND_Z + 1.0e-6:
        return
    half_height = 0.5 * (top_z - GROUND_Z)
    _add_box(
        spec,
        name,
        (x_center, y_center, GROUND_Z + half_height),
        (half_length, half_width, half_height),
        rgba,
    )


def _add_ramp(
    spec: mujoco.MjSpec,
    name: str,
    x_start: float,
    x_end: float,
    z_start: float,
    z_end: float,
    y_center: float,
    half_width: float,
    rgba: tuple[float, float, float, float],
    roll: float = 0.0,
) -> None:
    """Add a thin box whose upper face joins two requested elevations."""
    length = x_end - x_start
    pitch = -np.arctan2(z_end - z_start, length)
    half_length = 0.5 * np.hypot(length, z_end - z_start)
    half_thickness = 0.055
    # Offset the box downward along its local normal so its top face passes
    # approximately through the two specified endpoints.
    center_x = 0.5 * (x_start + x_end) - half_thickness * np.sin(pitch)
    center_z = 0.5 * (z_start + z_end) - half_thickness * np.cos(pitch)
    _add_box(
        spec,
        name,
        (center_x, y_center, center_z),
        (half_length, half_width, half_thickness),
        rgba,
        # rpo.xml uses compiler eulerseq="zyx": yaw, pitch, then roll.
        euler=(0.0, pitch, roll),
    )


def _add_pyramid_stairs(spec: mujoco.MjSpec, difficulty: float) -> None:
    """Build one wide staircase that climbs and returns to the ground plane."""
    y_center = LANE_CENTERS["stairs"]
    step_width = 0.45
    step_height = 0.04 + 0.04 * difficulty
    steps_per_side = 8
    x_start = OBSTACLE_START_X

    for index in range(steps_per_side):
        # The first tread is represented by the shared ground plane so the
        # lane entrance has no lip above the horizon.
        top_z = index * step_height
        x = x_start + (index + 0.5) * step_width
        _add_solid_box(
            spec,
            f"stairs_up_{index}",
            x,
            y_center,
            0.5 * step_width + 0.002,
            LANE_HALF_WIDTH,
            top_z,
            COLOR_STAIRS,
        )

    plateau_start = x_start + steps_per_side * step_width
    plateau_length = 0.80
    peak = (steps_per_side - 1) * step_height
    _add_solid_box(
        spec,
        "stairs_platform",
        plateau_start + 0.5 * plateau_length,
        y_center,
        0.5 * plateau_length,
        LANE_HALF_WIDTH,
        peak,
        COLOR_STAIRS,
    )

    down_start = plateau_start + plateau_length
    for index in range(steps_per_side):
        top_z = (steps_per_side - index - 1) * step_height
        x = down_start + (index + 0.5) * step_width
        _add_solid_box(
            spec,
            f"stairs_down_{index}",
            x,
            y_center,
            0.5 * step_width + 0.002,
            LANE_HALF_WIDTH,
            top_z,
            COLOR_STAIRS,
        )


def _add_gentle_pyramid_ramp(spec: mujoco.MjSpec, difficulty: float) -> None:
    """Build a broad, shallow up/down ramp with both ends exactly at z=0."""
    y_center = LANE_CENTERS["slope"]
    peak = 0.25 + 0.20 * difficulty
    up_start = OBSTACLE_START_X
    up_end = up_start + 4.0
    plateau_length = 0.80
    down_start = up_end + plateau_length
    down_end = down_start + 4.0
    _add_ramp(
        spec, "slope_up", up_start, up_end, GROUND_Z, peak,
        y_center, LANE_HALF_WIDTH, COLOR_RAMP,
    )
    _add_solid_box(
        spec, "slope_center", up_end + 0.5 * plateau_length, y_center,
        0.5 * plateau_length, LANE_HALF_WIDTH, peak, COLOR_RAMP,
    )
    _add_ramp(
        spec, "slope_down", down_start, down_end, peak, GROUND_Z,
        y_center, LANE_HALF_WIDTH, COLOR_RAMP,
    )


def _add_rolling_road(spec: mujoco.MjSpec, difficulty: float) -> None:
    """Build two mild rolling humps whose first and last edges are level."""
    y_center = LANE_CENTERS["rough"]
    peak = 0.20 + 0.22 * difficulty
    points = (
        (OBSTACLE_START_X, GROUND_Z),
        (OBSTACLE_START_X + 2.5, peak),
        (OBSTACLE_START_X + 5.0, GROUND_Z),
        (OBSTACLE_START_X + 7.5, 0.70 * peak),
        (OBSTACLE_START_X + 10.0, GROUND_Z),
    )
    for index, ((x_start, z_start), (x_end, z_end)) in enumerate(zip(points, points[1:])):
        _add_ramp(
            spec,
            f"rough_road_{index}",
            x_start,
            x_end,
            z_start,
            z_end,
            y_center,
            LANE_HALF_WIDTH,
            COLOR_ROUGH,
        )


def _add_plum_blossom_posts(
    spec: mujoco.MjSpec,
    rng: np.random.Generator,
    difficulty: float,
    x_start: float = OBSTACLE_START_X,
) -> None:
    """Add low five-post clusters directly on the common ground plane."""
    y_center = LANE_CENTERS["plum"]
    cluster_spacing = 1.05
    offsets = (
        (0.00, 0.00),
        (-0.28, 0.48),
        (-0.28, -0.48),
        (0.28, 0.48),
        (0.28, -0.48),
    )
    cluster_count = 10
    radius = 0.24
    for cluster in range(cluster_count):
        center_x = x_start + 0.45 + cluster * cluster_spacing
        lateral_shift = (0.10 + 0.12 * difficulty) * (-1.0 if cluster % 2 else 1.0)
        for post, (dx, dy) in enumerate(offsets):
            top_z = rng.uniform(0.04, 0.07 + 0.07 * difficulty)
            half_height = 0.5 * top_z
            spec.worldbody.add_geom(
                name=f"plum_{cluster}_{post}",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=(center_x + dx, y_center + dy + lateral_shift, half_height),
                size=(radius, half_height, 0.0),
                rgba=COLOR_POSTS,
                friction=TERRAIN_FRICTION,
                condim=4,
                contype=1,
                conaffinity=15,
                group=0,
            )


def build_rough_model(
    model_path: Path,
    difficulty: float,
    seed: int,
    lane: str = "stairs",
) -> mujoco.MjModel:
    """Compile four parallel ground-level obstacle lanes and select a spawn lane."""
    spec = mujoco.MjSpec.from_file(str(model_path))
    ground = spec.geom("ground")
    if ground is None:
        raise ValueError(f"MuJoCo model has no 'ground' geom: {model_path}")
    ground.pos[2] = GROUND_Z
    ground.rgba[:] = (0.22, 0.25, 0.28, 1.0)

    if lane not in LANE_CENTERS:
        raise ValueError(f"Unknown lane '{lane}'. Expected one of {tuple(LANE_CENTERS)}.")
    robot_body = spec.body("base_link")
    if robot_body is None:
        raise ValueError(f"MuJoCo model has no 'base_link' body: {model_path}")
    robot_body.pos[1] = LANE_CENTERS[lane]

    rng = np.random.default_rng(seed)
    _add_pyramid_stairs(spec, difficulty)
    _add_gentle_pyramid_ramp(spec, difficulty)
    _add_rolling_road(spec, difficulty)
    _add_plum_blossom_posts(spec, rng, difficulty)

    model = spec.compile()
    print("[MuJoCo] Parallel ground-level terrain lanes compiled:")
    print(f"  difficulty={difficulty:.2f}, seed={seed}, spawn lane={lane}")
    print(f"  lane width={2.0 * LANE_HALF_WIDTH:.1f} m; all lane entrances/exits are at z=0")
    print("  lanes: stairs | gentle slope | rolling road | plum posts")
    return model


def _parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[2]
    default_model = workspace / "robolab" / "data" / "robots" / "roboparty" / "rpo" / "mjcf" / "rpo.xml"
    parser = argparse.ArgumentParser(
        description="Visualize an exported Loco-Transformer policy on a procedural rough MuJoCo course."
    )
    parser.add_argument(
        "--load-model", "--load_model", dest="load_model", type=Path, required=True,
        help="Path to exported/policy.pt from loco_transformer/scripts/play.py.",
    )
    parser.add_argument("--model", type=Path, default=default_model, help="Override the base RPO MuJoCo XML path.")
    parser.add_argument(
        "--difficulty", type=float, default=1.0,
        help="Terrain severity in [0, 1]; 1 resembles highest-difficulty Rough-Play.",
    )
    parser.add_argument(
        "--lane",
        choices=tuple(LANE_CENTERS),
        default="stairs",
        help="Lane where the robot spawns: stairs, slope, rough, or plum.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic rough-block/post height seed.")
    parser.add_argument(
        "--duration", type=float, default=0.0,
        help="Simulation duration in seconds; 0 keeps running until the viewer closes.",
    )
    parser.add_argument("--dt", type=float, default=0.001, help="MuJoCo physics time step.")
    parser.add_argument("--decimation", type=int, default=20, help="Physics steps per 50 Hz policy step.")
    parser.add_argument("--lin-vel-step", type=float, default=0.05, help="Keyboard linear-velocity increment.")
    parser.add_argument("--ang-vel-step", type=float, default=0.05, help="Keyboard angular-velocity increment.")
    parser.add_argument(
        "--gamepad",
        action="store_true",
        help="Add real-time Xbox/XInput commands to the keyboard command.",
    )
    parser.add_argument(
        "--gamepad-index",
        type=int,
        choices=range(4),
        default=0,
        help="XInput controller index used with --gamepad (default: 0).",
    )
    parser.add_argument("--no-real-time", action="store_true", help="Disable wall-clock pacing.")
    args = parser.parse_args()

    if not args.load_model.is_file():
        parser.error(f"policy file does not exist: {args.load_model}")
    if not args.model.is_file():
        parser.error(f"MuJoCo model does not exist: {args.model}")
    if not 0.0 <= args.difficulty <= 1.0:
        parser.error("--difficulty must be in [0, 1]")
    if args.duration < 0.0:
        parser.error("--duration must be non-negative")
    if args.dt <= 0.0 or args.decimation <= 0:
        parser.error("--dt and --decimation must be positive")
    return args


def run(args: argparse.Namespace) -> None:
    model = build_rough_model(args.model, args.difficulty, args.seed, args.lane)
    base.run_model(model, args)


if __name__ == "__main__":
    run(_parse_args())
