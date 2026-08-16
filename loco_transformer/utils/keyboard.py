# Copyright (c) 2025-2026, Loco-Transformer Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard velocity-command controller for Isaac Lab manager-based environments."""

from __future__ import annotations

import weakref

import carb
import omni.appwindow
import torch


class VelocityCommandKeyboard:
    """Write SE(2) velocity commands into an Isaac Lab command manager.

    The controller intentionally targets ``command_manager`` instead of RoboLab's
    direct-environment ``command_generator``.  Commands are clamped to the ranges
    used to train the Loco-Transformer policy.
    """

    def __init__(
        self,
        env,
        command_name: str = "base_velocity",
        lin_vel_step: float = 0.05,
        ang_vel_step: float = 0.05,
        lin_vel_x_range: tuple[float, float] = (-0.6, 1.0),
        lin_vel_y_range: tuple[float, float] = (-0.5, 0.5),
        ang_vel_z_range: tuple[float, float] = (-1.57, 1.57),
    ) -> None:
        self.env = env.unwrapped
        self.command_name = command_name
        self.lin_vel_step = lin_vel_step
        self.ang_vel_step = ang_vel_step
        self._ranges = (lin_vel_x_range, lin_vel_y_range, ang_vel_z_range)
        self._command = [0.0, 0.0, 0.0]

        if not hasattr(self.env, "command_manager"):
            raise TypeError("VelocityCommandKeyboard requires a manager-based environment with command_manager.")

        app_window = omni.appwindow.get_default_app_window()
        if app_window is None:
            raise RuntimeError("Cannot initialize keyboard control: no Omniverse app window is available.")
        self._input = carb.input.acquire_input_interface()
        self._keyboard = app_window.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
        self._write_command(announce=False)
        print("[Keyboard] Loco-Transformer velocity control:")
        print("  W/S: forward/backward     A/D: strafe left/right")
        print("  Q/E: turn left/right      X: stop")
        print("  R: reset the environments")

    def __del__(self) -> None:
        keyboard_sub = getattr(self, "_keyboard_sub", None)
        if keyboard_sub is not None:
            try:
                self._input.unsubscribe_from_keyboard_events(self._keyboard, keyboard_sub)
            except Exception:
                # Omniverse may already have torn down its input interface.
                pass
            self._keyboard_sub = None

    def _on_keyboard_event(self, event, *args) -> bool:
        del args
        if event.type not in (carb.input.KeyboardEventType.KEY_PRESS, carb.input.KeyboardEventType.KEY_REPEAT):
            return True

        key = event.input.name
        if key == "W":
            self._adjust(0, self.lin_vel_step)
        elif key == "S":
            self._adjust(0, -self.lin_vel_step)
        elif key == "A":
            self._adjust(1, self.lin_vel_step)
        elif key == "D":
            self._adjust(1, -self.lin_vel_step)
        elif key == "Q":
            self._adjust(2, self.ang_vel_step)
        elif key == "E":
            self._adjust(2, -self.ang_vel_step)
        elif key == "X":
            self._command[:] = (0.0, 0.0, 0.0)
            self._write_command()
        elif key == "R":
            # Setting the episode length to its limit requests a reset safely on
            # the next environment step; calling env.reset() in an input callback
            # can race the simulator thread.
            self.env.episode_length_buf[:] = self.env.max_episode_length
            print("[Keyboard] Environment reset requested")
        return True

    def _adjust(self, index: int, delta: float) -> None:
        low, high = self._ranges[index]
        self._command[index] = min(high, max(low, self._command[index] + delta))
        self._write_command()

    def advance(self) -> None:
        """Reapply the command after environment resets/resampling."""
        self._write_command(announce=False)

    def _write_command(self, announce: bool = True) -> None:
        command = self.env.command_manager.get_command(self.command_name)
        values = torch.tensor(self._command, device=command.device, dtype=command.dtype)
        command[:, :3] = values
        if announce:
            print(
                f"[Keyboard] vx={self._command[0]:+.2f} m/s, "
                f"vy={self._command[1]:+.2f} m/s, wz={self._command[2]:+.2f} rad/s"
            )
