# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Preview the exact terrain generator used by History-Rough training.

Examples:

.. code-block:: bash

    python loco_transformer/scripts/visualize_terrains.py
    python loco_transformer/scripts/visualize_terrains.py --terrain-type stairs --level all
    python loco_transformer/scripts/visualize_terrains.py --terrain-type slope --level 9
"""

"""Launch Isaac Sim first."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Preview RPO History-Rough curriculum terrains.")
parser.add_argument(
    "--terrain-type",
    choices=("all", "flat", "rough", "slope", "stairs"),
    default="all",
    help="Terrain family to include.",
)
parser.add_argument(
    "--level",
    default="all",
    help="Show all ten curriculum levels or one level in [0, 9].",
)
parser.add_argument("--seed", type=int, default=42, help="Deterministic terrain generator seed.")
parser.add_argument(
    "--color-scheme",
    choices=("height", "random", "none"),
    default="height",
    help="Terrain mesh color scheme.",
)
parser.add_argument(
    "--show-origins",
    action="store_true",
    help="Draw the generated tile origins.",
)
parser.add_argument(
    "--duration",
    type=float,
    default=0.0,
    help="Seconds to display; zero keeps the window open until it is closed.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.duration < 0.0:
    parser.error("--duration must be non-negative")
if args_cli.level != "all":
    try:
        selected_level = int(args_cli.level)
    except ValueError:
        parser.error("--level must be 'all' or an integer from 0 through 9")
    if not 0 <= selected_level <= 9:
        parser.error("--level must be in the range 0 through 9")
else:
    selected_level = None

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Terrain construction follows."""

import copy

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg

from loco_transformer.terrain_generator_cfg import (
    LOCO_HISTORY_ROUGH_TERRAINS_CFG,
    terrain_difficulty_for_level,
)


_TERRAIN_KEYS = {
    "all": ("flat", "rough", "slope_up", "slope_down", "stairs_up", "stairs_down"),
    "flat": ("flat",),
    "rough": ("rough",),
    "slope": ("slope_up", "slope_down"),
    "stairs": ("stairs_up", "stairs_down"),
}


def _make_preview_cfg():
    cfg = copy.deepcopy(LOCO_HISTORY_ROUGH_TERRAINS_CFG)
    cfg.seed = args_cli.seed
    cfg.color_scheme = args_cli.color_scheme

    selected_keys = _TERRAIN_KEYS[args_cli.terrain_type]
    cfg.sub_terrains = {key: cfg.sub_terrains[key] for key in selected_keys}
    total_proportion = sum(term.proportion for term in cfg.sub_terrains.values())
    for term in cfg.sub_terrains.values():
        term.proportion /= total_proportion

    # The default view deliberately preserves the exact 10 x 20 training
    # grid. Filtered views only need one column per selected direction/type.
    if args_cli.terrain_type != "all":
        cfg.num_cols = len(selected_keys)

    if selected_level is None:
        cfg.curriculum = True
        cfg.num_rows = 10
        level_description = "all levels 0-9 (row-wise easy to hard)"
    else:
        difficulty = terrain_difficulty_for_level(selected_level)
        cfg.curriculum = True
        cfg.num_rows = 1
        cfg.num_cols = len(selected_keys)
        cfg.difficulty_range = (difficulty, difficulty)
        # A single-level preview shows every selected direction/type once.
        for term in cfg.sub_terrains.values():
            term.proportion = 1.0 / len(cfg.sub_terrains)
        level_description = f"level {selected_level} (representative difficulty={difficulty:.2f})"

    return cfg, level_description


def _print_summary(cfg, level_description: str) -> None:
    print("\n[INFO] History-Rough terrain preview")
    print(f"  type filter     : {args_cli.terrain_type}")
    print(f"  level           : {level_description}")
    print(f"  seed            : {cfg.seed}")
    print(f"  grid            : {cfg.num_rows} rows x {cfg.num_cols} columns")
    print(f"  tile / border   : {cfg.size[0]:.1f} x {cfg.size[1]:.1f} m / {cfg.border_width:.1f} m")
    print(f"  resolution      : horizontal={cfg.horizontal_scale:.3f} m, vertical={cfg.vertical_scale:.3f} m")
    print(f"  color / origins : {cfg.color_scheme} / {args_cli.show_origins}")
    print("  generated types :")
    for name, term in cfg.sub_terrains.items():
        detail = ""
        if name == "rough":
            detail = (
                f", Perlin zScale={term.noise_scale} m "
                f"(about +/-{0.5 * term.noise_scale[-1]:.2f} m at max), "
                f"platform={term.platform_width:.1f} m"
            )
        elif name.startswith("slope"):
            detail = f", slope={term.slope_range}, platform={term.platform_width:.1f} m"
        elif name.startswith("stairs"):
            detail = (
                f", height={term.step_height_range} m, width={term.step_width:.2f} m, "
                f"platform={term.platform_width:.1f} m"
            )
        print(f"    - {name}: {term.proportion:.3f}{detail}")
    print()


def main() -> None:
    terrain_cfg, level_description = _make_preview_cfg()
    _print_summary(terrain_cfg, level_description)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args_cli.device))
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    importer_cfg = TerrainImporterCfg(
        num_envs=terrain_cfg.num_rows * terrain_cfg.num_cols,
        env_spacing=3.0,
        prim_path="/World/ground",
        max_init_terrain_level=None,
        terrain_type="generator",
        terrain_generator=terrain_cfg,
        debug_vis=args_cli.show_origins,
    )
    if args_cli.color_scheme in ("height", "random"):
        importer_cfg.visual_material = None
    TerrainImporter(importer_cfg)

    extent_x = terrain_cfg.num_rows * terrain_cfg.size[0]
    extent_y = terrain_cfg.num_cols * terrain_cfg.size[1]
    center = [0.5 * extent_x, 0.5 * extent_y, 0.0]
    camera_height = max(extent_x, extent_y) * 0.85
    sim.set_camera_view(
        eye=[center[0] - 0.08 * extent_x, center[1] - 0.08 * extent_y, camera_height],
        target=center,
    )
    sim.reset()
    print("[INFO] Terrain generation complete. Close the window to exit.")

    elapsed = 0.0
    while simulation_app.is_running():
        sim.step()
        elapsed += sim.get_physics_dt()
        if args_cli.duration > 0.0 and elapsed >= args_cli.duration:
            break


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
