# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Left-right symmetry transforms for the 777-D History-Rough task."""

from __future__ import annotations

import torch
from tensordict import TensorDict


HISTORY_OBSERVATION_DIM = 777
ACTION_DIM = 23
HISTORY_LENGTH = 10
HEIGHT_MAP_SHAPE = (11, 21)

# Isaac Sim RPO joint order:
# left/right thigh yaw, torso, then ten left/right joint pairs.
JOINT_MIRROR_INDICES = (
    1,
    0,
    2,
    4,
    3,
    6,
    5,
    8,
    7,
    10,
    9,
    12,
    11,
    14,
    13,
    16,
    15,
    18,
    17,
    20,
    19,
    22,
    21,
)
JOINT_MIRROR_SIGNS = (
    -1,
    -1,
    -1,
    -1,
    -1,
    1,
    1,
    1,
    1,
    -1,
    -1,
    1,
    1,
    -1,
    -1,
    1,
    1,
    1,
    1,
    -1,
    -1,
    -1,
    -1,
)


def _validate_last_dim(tensor: torch.Tensor, expected: int, name: str) -> None:
    if tensor.ndim == 0 or tensor.shape[-1] != expected:
        actual = "scalar" if tensor.ndim == 0 else tensor.shape[-1]
        raise ValueError(f"{name} must have last dimension {expected}, got {actual}.")


def _apply_xyz_signs(values: torch.Tensor, signs: tuple[int, int, int]) -> torch.Tensor:
    frames = values.reshape(*values.shape[:-1], -1, 3)
    sign_tensor = values.new_tensor(signs)
    return (frames * sign_tensor).reshape(values.shape)


def _mirror_joint_values(values: torch.Tensor) -> torch.Tensor:
    """Mirror one or more contiguous 23-D joint frames without reversing time."""
    frames = values.reshape(*values.shape[:-1], -1, ACTION_DIM)
    indices = torch.tensor(JOINT_MIRROR_INDICES, device=values.device, dtype=torch.long)
    signs = values.new_tensor(JOINT_MIRROR_SIGNS)
    return (frames[..., indices] * signs).reshape(values.shape)


def mirror_history_observation(observation: torch.Tensor) -> torch.Tensor:
    """Apply the RPO left-right transform to a flat 777-D observation."""
    _validate_last_dim(observation, HISTORY_OBSERVATION_DIM, "History-Rough observation")
    mirrored = observation.clone()

    # Term-major layout. Each historical term contains ten old-to-new frames.
    mirrored[..., 0:30] = _apply_xyz_signs(observation[..., 0:30], (-1, 1, -1))
    mirrored[..., 30:60] = _apply_xyz_signs(observation[..., 30:60], (1, -1, 1))
    mirrored[..., 60:63] = _apply_xyz_signs(observation[..., 60:63], (1, -1, -1))
    mirrored[..., 63:293] = _mirror_joint_values(observation[..., 63:293])
    mirrored[..., 293:523] = _mirror_joint_values(observation[..., 293:523])
    mirrored[..., 523:546] = _mirror_joint_values(observation[..., 523:546])

    # GridPatternCfg ordering="xy": x is the inner 21-point dimension and
    # y is the outer 11-point dimension. Left-right symmetry flips y only.
    height_map = observation[..., 546:777].reshape(*observation.shape[:-1], *HEIGHT_MAP_SHAPE)
    mirrored[..., 546:777] = height_map.flip(dims=(-2,)).reshape(*observation.shape[:-1], -1)
    return mirrored


def mirror_actions(actions: torch.Tensor) -> torch.Tensor:
    """Apply the RPO left-right transform to 23-D joint actions."""
    _validate_last_dim(actions, ACTION_DIM, "RPO actions")
    return _mirror_joint_values(actions)


@torch.no_grad()
def compute_symmetric_states(
    env,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Return ``[original, left-right mirrored]`` RSL-RL training batches."""
    del env  # Kept in the signature for RSL-RL callback compatibility.

    if obs is None:
        observations_augmented = None
    else:
        if "policy" not in obs.keys():
            raise KeyError("Symmetry augmentation requires a 'policy' observation group.")
        observations_mirrored = obs.clone()
        observations_mirrored["policy"] = mirror_history_observation(obs["policy"])
        if "critic" in obs.keys():
            observations_mirrored["critic"] = mirror_history_observation(obs["critic"])
        observations_augmented = torch.cat((obs, observations_mirrored), dim=0)

    if actions is None:
        actions_augmented = None
    else:
        actions_augmented = torch.cat((actions, mirror_actions(actions)), dim=0)

    return observations_augmented, actions_augmented


__all__ = [
    "ACTION_DIM",
    "HEIGHT_MAP_SHAPE",
    "HISTORY_LENGTH",
    "HISTORY_OBSERVATION_DIM",
    "JOINT_MIRROR_INDICES",
    "JOINT_MIRROR_SIGNS",
    "compute_symmetric_states",
    "mirror_actions",
    "mirror_history_observation",
]
