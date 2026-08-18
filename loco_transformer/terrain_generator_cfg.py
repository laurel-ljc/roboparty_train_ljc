# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terrain configuration for the History-Rough locomotion task."""

from __future__ import annotations

import numpy as np

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.terrains.terrain_generator import TerrainGenerator
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass

from robolab.terrains.height_field import PerlinPlaneTerrainCfg
from robolab.terrains.height_field.hf_terrains import generate_perlin_noise


TERRAIN_SEED = 42
"""Default deterministic seed shared by training and terrain preview."""


@height_field_to_mesh
def perlin_rough_with_platform(difficulty: float, cfg: "PerlinRoughTerrainCfg") -> np.ndarray:
    """Generate curriculum-scaled Perlin roughness with a flat central spawn platform."""
    heights = generate_perlin_noise(difficulty, cfg)

    # ``height_field_to_mesh`` has already removed the border from cfg.size.
    # Flatten a centered square so resets do not start a robot on a local bump.
    platform_pixels = max(1, int(round(cfg.platform_width / cfg.horizontal_scale)))
    center_x, center_y = heights.shape[0] // 2, heights.shape[1] // 2
    half_low = platform_pixels // 2
    half_high = platform_pixels - half_low
    heights[
        center_x - half_low : center_x + half_high,
        center_y - half_low : center_y + half_high,
    ] = 0
    return heights


@configclass
class PerlinRoughTerrainCfg(PerlinPlaneTerrainCfg):
    """Perlin height-field terrain with a centered flat platform."""

    function = perlin_rough_with_platform
    platform_width: float = 2.0


class BalancedDirectionTerrainGenerator(TerrainGenerator):
    """Balance paired up/down terrain columns across adjacent levels.

    With 20 columns, a 25% terrain family occupies five columns, so a fixed
    per-column split cannot represent 12.5%/12.5% exactly. The standard
    column allocation is used on even rows and the paired directions are
    exchanged on odd rows. Across every two levels this yields an exact
    50/50 directional split without changing the family proportions.
    """

    _DIRECTION_PAIRS = {
        "slope_up": "slope_down",
        "slope_down": "slope_up",
        "stairs_up": "stairs_down",
        "stairs_down": "stairs_up",
    }

    def _generate_curriculum_terrains(self):
        proportions = np.array([cfg.proportion for cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)
        names = list(self.cfg.sub_terrains)
        configs = list(self.cfg.sub_terrains.values())

        column_indices = []
        cumulative = np.cumsum(proportions)
        for column in range(self.cfg.num_cols):
            column_indices.append(np.min(np.where(column / self.cfg.num_cols + 0.001 < cumulative)[0]))

        for column, base_index in enumerate(column_indices):
            for row in range(self.cfg.num_rows):
                terrain_name = names[base_index]
                if row % 2 == 1 and terrain_name in self._DIRECTION_PAIRS:
                    terrain_name = self._DIRECTION_PAIRS[terrain_name]
                terrain_cfg = configs[names.index(terrain_name)]

                lower, upper = self.cfg.difficulty_range
                difficulty = (row + self.np_rng.uniform()) / self.cfg.num_rows
                difficulty = lower + (upper - lower) * difficulty
                mesh, origin = self._get_terrain_mesh(difficulty, terrain_cfg)
                self._add_sub_terrain(mesh, origin, row, column, terrain_cfg)


LOCO_HISTORY_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    class_type=BalancedDirectionTerrainGenerator,
    seed=TERRAIN_SEED,
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    color_scheme="none",
    horizontal_scale=0.05,
    vertical_scale=0.005,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.20),
        "rough": PerlinRoughTerrainCfg(
            proportion=0.30,
            # RoboLab's centered Perlin output spans roughly half of zScale
            # in each direction, so 0.12 gives approximately +/- 0.06 m.
            noise_scale=(0.0, 0.12),
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
            border_width=1.0,
            platform_width=2.0,
        ),
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.125,
            slope_range=(0.0, 0.30),
            platform_width=2.0,
            border_width=1.0,
        ),
        "slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.125,
            slope_range=(0.0, 0.30),
            platform_width=2.0,
            border_width=1.0,
        ),
        "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.125,
            step_height_range=(0.02, 0.16),
            step_width=0.30,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.125,
            step_height_range=(0.02, 0.16),
            step_width=0.30,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


def terrain_difficulty_for_level(level: int, num_levels: int = 10) -> float:
    """Return the representative midpoint difficulty used to preview one level."""
    if not 0 <= level < num_levels:
        raise ValueError(f"Terrain level must be in [0, {num_levels - 1}], got {level}.")
    return (level + 0.5) / num_levels
