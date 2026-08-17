# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Thin subclass of ManagerBasedRLEnv for the loco_transformer task.

Adds a 3-step action buffer used by ``action_smoothness_l2`` and handles
per-foot ray-caster scanners required by ``feet_height``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from isaaclab.utils.buffers import CircularBuffer


class LocoTransformerEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv with action buffer and foot-scanner support.

    The 3-step ``action_buffer`` enables the ``action_smoothness_l2`` reward
    which penalizes second-order action differences:
        || a_t - 2*a_{t-1} + a_{t-2} ||^2.
    """

    cfg: ManagerBasedRLEnvCfg

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # 3-step circular buffer for action_smoothness_l2
        num_actions = self.action_manager.total_action_dim
        self.action_buffer = CircularBuffer(
            max_len=3, batch_size=self.num_envs, device=self.device
        )
        # pre-fill with zeros to match the robolab BaseEnv pattern
        for _ in range(3):
            self.action_buffer.append(
                torch.zeros(self.num_envs, num_actions, device=self.device)
            )

    def step(self, action: torch.Tensor):
        """Execute one timestep, maintaining the action buffer."""
        self.action_buffer.append(action.to(self.device))
        return super().step(action)

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments and clear their action buffer entries."""
        super()._reset_idx(env_ids)
        if len(env_ids) > 0:
            self.action_buffer.reset(env_ids)
