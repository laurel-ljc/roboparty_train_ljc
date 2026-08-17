"""Simulator-independent tensor helpers shared by Loco-Transformer rewards."""

from __future__ import annotations

import torch


def joint_pos_limit_violation(
    joint_pos: torch.Tensor, soft_joint_pos_limits: torch.Tensor
) -> torch.Tensor:
    """Return non-negative per-joint violations of lower and upper soft limits."""
    lower_violation = -torch.clamp(
        joint_pos - soft_joint_pos_limits[..., 0], max=0.0
    )
    upper_violation = torch.clamp(
        joint_pos - soft_joint_pos_limits[..., 1], min=0.0
    )
    return lower_violation + upper_violation
