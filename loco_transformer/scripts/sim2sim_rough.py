# Copyright (c) 2025-2026, Loco-Transformer Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""MuJoCo Sim2Sim playback on a procedural History-Rough-style course.

This is the rough-terrain counterpart of :mod:`sim2sim`.  Policy loading,
309/777-D observations, yaw-aligned height scanning, PD control, keyboard
commands, and the chase camera are shared with that script.  Only the MuJoCo
world is replaced by a deterministic course containing rough ground, gentle
ramps, a steep/cambered road, ascending and descending pyramid stairs, and
plum-blossom stepping posts.

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


PIT_FLOOR_Z = -0.60
ROAD_HALF_WIDTH = 1.25
TERRAIN_FRICTION = (1.0, 0.005, 0.0001)

COLOR_START = (0.28, 0.42, 0.32, 1.0)
COLOR_ROUGH = (0.42, 0.35, 0.27, 1.0)
COLOR_RAMP = (0.45, 0.50, 0.58, 1.0)
COLOR_STEEP = (0.50, 0.38, 0.30, 1.0)
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
    half_length: float,
    half_width: float,
    top_z: float,
    rgba: tuple[float, float, float, float],
) -> None:
    """Add a box extending from the pit floor to the requested top height."""
    half_height = 0.5 * (top_z - PIT_FLOOR_Z)
    _add_box(
        spec,
        name,
        (x_center, 0.0, PIT_FLOOR_Z + half_height),
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
        (center_x, 0.0, center_z),
        (half_length, half_width, half_thickness),
        rgba,
        euler=(roll, pitch, 0.0),
    )


def _add_rough_section(
    spec: mujoco.MjSpec,
    rng: np.random.Generator,
    difficulty: float,
    x_start: float = 2.0,
    x_end: float = 6.0,
) -> None:
    """Build blocky rough ground with roughly +/-6 cm variation at level 1."""
    cell = 0.40
    columns = int(round((x_end - x_start) / cell))
    rows = 6
    for ix in range(columns):
        for iy in range(rows):
            top_z = rng.uniform(-0.06, 0.06) * difficulty
            x = x_start + (ix + 0.5) * cell
            y = (iy - 0.5 * (rows - 1)) * cell
            half_height = 0.5 * (top_z - PIT_FLOOR_Z)
            _add_box(
                spec,
                f"rough_{ix}_{iy}",
                (x, y, PIT_FLOOR_Z + half_height),
                (0.5 * cell + 0.003, 0.5 * cell + 0.003, half_height),
                COLOR_ROUGH,
            )


def _add_gentle_pyramid_ramp(spec: mujoco.MjSpec, difficulty: float) -> None:
    peak = 0.20 + 0.30 * difficulty
    _add_ramp(spec, "ramp_up", 6.0, 9.0, 0.0, peak, ROAD_HALF_WIDTH, COLOR_RAMP)
    _add_solid_box(spec, "ramp_platform", 9.5, 0.5, ROAD_HALF_WIDTH, peak, COLOR_RAMP)
    _add_ramp(spec, "ramp_down", 10.0, 13.0, peak, 0.0, ROAD_HALF_WIDTH, COLOR_RAMP)


def _add_steep_cambered_road(spec: mujoco.MjSpec, difficulty: float) -> None:
    """Build a narrower, steeper up/down road with alternating side camber."""
    peak = 0.45 + 0.45 * difficulty
    camber = np.deg2rad(3.0 + 5.0 * difficulty)
    _add_ramp(spec, "steep_up", 13.0, 15.5, 0.0, peak, 0.92, COLOR_STEEP, roll=camber)
    _add_solid_box(spec, "steep_platform", 16.0, 0.5, 0.92, peak, COLOR_STEEP)
    _add_ramp(spec, "steep_down", 16.5, 19.0, peak, 0.0, 0.92, COLOR_STEEP, roll=-camber)


def _add_pyramid_stairs(spec: mujoco.MjSpec, difficulty: float) -> None:
    """Build matched ascending/descending stairs like the Rough task pair."""
    step_width = 0.30
    step_height = 0.04 + 0.10 * difficulty
    steps_per_side = 9
    x_start = 19.0

    for index in range(steps_per_side):
        top_z = (index + 1) * step_height
        x = x_start + (index + 0.5) * step_width
        _add_solid_box(
            spec,
            f"stairs_up_{index}",
            x,
            0.5 * step_width + 0.002,
            ROAD_HALF_WIDTH,
            top_z,
            COLOR_STAIRS,
        )

    plateau_start = x_start + steps_per_side * step_width
    plateau_length = 0.90
    peak = steps_per_side * step_height
    _add_solid_box(
        spec,
        "stairs_platform",
        plateau_start + 0.5 * plateau_length,
        0.5 * plateau_length,
        ROAD_HALF_WIDTH,
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
            0.5 * step_width + 0.002,
            ROAD_HALF_WIDTH,
            top_z,
            COLOR_STAIRS,
        )


def _add_plum_blossom_posts(
    spec: mujoco.MjSpec,
    rng: np.random.Generator,
    difficulty: float,
    x_start: float,
) -> float:
    """Add five-post clusters over the lowered ground, returning their end x."""
    cluster_spacing = 0.95
    offsets = (
        (0.00, 0.00),
        (-0.22, 0.34),
        (-0.22, -0.34),
        (0.24, 0.34),
        (0.24, -0.34),
    )
    cluster_count = 8
    radius = 0.20
    for cluster in range(cluster_count):
        center_x = x_start + 0.45 + cluster * cluster_spacing
        lateral_shift = (0.12 + 0.12 * difficulty) * (-1.0 if cluster % 2 else 1.0)
        for post, (dx, dy) in enumerate(offsets):
            top_z = rng.uniform(0.0, 0.04 + 0.08 * difficulty)
            half_height = 0.5 * (top_z - PIT_FLOOR_Z)
            spec.worldbody.add_geom(
                name=f"plum_{cluster}_{post}",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=(center_x + dx, dy + lateral_shift, PIT_FLOOR_Z + half_height),
                size=(radius, half_height, 0.0),
                rgba=COLOR_POSTS,
                friction=TERRAIN_FRICTION,
                condim=4,
                contype=1,
                conaffinity=15,
                group=0,
            )
    return x_start + cluster_count * cluster_spacing


def build_rough_model(model_path: Path, difficulty: float, seed: int) -> mujoco.MjModel:
    """Compile the RPO model with a deterministic mixed-obstacle course."""
    spec = mujoco.MjSpec.from_file(str(model_path))
    ground = spec.geom("ground")
    if ground is None:
        raise ValueError(f"MuJoCo model has no 'ground' geom: {model_path}")
    ground.pos[2] = PIT_FLOOR_Z
    ground.rgba[:] = (0.16, 0.18, 0.20, 1.0)

    rng = np.random.default_rng(seed)
    _add_solid_box(spec, "start_platform", 0.0, 2.0, 1.7, 0.0, COLOR_START)
    _add_rough_section(spec, rng, difficulty)
    _add_gentle_pyramid_ramp(spec, difficulty)
    _add_steep_cambered_road(spec, difficulty)
    _add_pyramid_stairs(spec, difficulty)

    stairs_end = 19.0 + 9 * 0.30 + 0.90 + 9 * 0.30
    posts_end = _add_plum_blossom_posts(spec, rng, difficulty, stairs_end)
    _add_solid_box(spec, "finish_platform", posts_end + 2.0, 2.0, 1.7, 0.0, COLOR_START)

    model = spec.compile()
    print("[MuJoCo] Procedural History-Rough course compiled:")
    print(f"  difficulty={difficulty:.2f}, seed={seed}, course length~{posts_end + 4.0:.1f} m")
    print("  rough -> ramp up/down -> steep cambered road -> pyramid stairs up/down -> plum posts")
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
    parser.add_argument("--seed", type=int, default=42, help="Deterministic rough-block/post height seed.")
    parser.add_argument(
        "--duration", type=float, default=0.0,
        help="Simulation duration in seconds; 0 keeps running until the viewer closes.",
    )
    parser.add_argument("--dt", type=float, default=0.001, help="MuJoCo physics time step.")
    parser.add_argument("--decimation", type=int, default=20, help="Physics steps per 50 Hz policy step.")
    parser.add_argument("--lin-vel-step", type=float, default=0.05, help="Keyboard linear-velocity increment.")
    parser.add_argument("--ang-vel-step", type=float, default=0.05, help="Keyboard angular-velocity increment.")
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
    model = build_rough_model(args.model, args.difficulty, args.seed)
    base.run_model(model, args)


if __name__ == "__main__":
    run(_parse_args())
