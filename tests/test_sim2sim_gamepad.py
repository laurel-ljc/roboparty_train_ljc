"""Regression tests for Sim2Sim keyboard and Xbox command inputs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("mujoco")
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "loco_transformer" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
import sim2sim  # noqa: E402


def _gamepad(**values) -> sim2sim._XInputGamepad:
    gamepad = sim2sim._XInputGamepad()
    for name, value in values.items():
        setattr(gamepad, name, value)
    return gamepad


def test_keyboard_behavior_remains_latched_and_resettable():
    keyboard = sim2sim.KeyboardCommands(linear_step=0.05, angular_step=0.10)

    keyboard.on_key(ord("I"))
    keyboard.on_key(ord("J"))
    keyboard.on_key(ord("U"))
    assert keyboard.snapshot() == pytest.approx([0.05, 0.05, 0.10])

    keyboard.on_key(ord("P"))
    assert keyboard.snapshot() == pytest.approx([0.0, 0.0, 0.0])
    keyboard.on_key(ord("R"))
    assert keyboard.consume_reset() is True
    assert keyboard.consume_reset() is False


def test_left_stick_center_and_standard_deadzone_are_zero():
    assert sim2sim._gamepad_velocity_command(_gamepad()) == pytest.approx([0.0, 0.0, 0.0])
    assert sim2sim._gamepad_velocity_command(
        _gamepad(sThumbLX=7849, sThumbLY=0)
    ) == pytest.approx([0.0, 0.0, 0.0])


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"sThumbLY": 32767}, (1.0, 0.0, 0.0)),
        ({"sThumbLY": -32768}, (-0.6, 0.0, 0.0)),
        ({"sThumbLX": -32768}, (0.0, 0.5, 0.0)),
        ({"sThumbLX": 32767}, (0.0, -0.5, 0.0)),
    ],
)
def test_left_stick_extremes_match_training_ranges(state, expected):
    assert sim2sim._gamepad_velocity_command(_gamepad(**state)) == pytest.approx(expected)


def test_triggers_turn_left_right_and_cancel_each_other():
    left = sim2sim._gamepad_velocity_command(_gamepad(bLeftTrigger=255))
    right = sim2sim._gamepad_velocity_command(_gamepad(bRightTrigger=255))
    both = sim2sim._gamepad_velocity_command(
        _gamepad(bLeftTrigger=255, bRightTrigger=255)
    )

    assert left == pytest.approx([0.0, 0.0, 1.57])
    assert right == pytest.approx([0.0, 0.0, -1.57])
    assert both == pytest.approx([0.0, 0.0, 0.0])


def test_keyboard_and_gamepad_commands_add_then_clip():
    keyboard = np.asarray([0.8, 0.4, 1.0], dtype=np.float32)
    gamepad = np.asarray([0.6, 0.4, 1.0], dtype=np.float32)
    assert sim2sim._combine_velocity_commands(keyboard, gamepad) == pytest.approx(
        [1.0, 0.5, 1.57]
    )

    keyboard = np.asarray([-0.5, -0.4, -1.0], dtype=np.float32)
    gamepad = np.asarray([-0.5, -0.4, -1.0], dtype=np.float32)
    assert sim2sim._combine_velocity_commands(keyboard, gamepad) == pytest.approx(
        [-0.6, -0.5, -1.57]
    )


def test_disconnect_clears_gamepad_command_and_reconnect_reads_fresh_state():
    forward = _gamepad(sThumbLY=32767)
    backward = _gamepad(sThumbLY=-32768)
    responses = iter(
        [
            (sim2sim.XINPUT_ERROR_SUCCESS, forward),
            (1167, forward),  # ERROR_DEVICE_NOT_CONNECTED; stale state must be ignored.
            (sim2sim.XINPUT_ERROR_SUCCESS, backward),
        ]
    )
    controller = sim2sim.GamepadCommands(
        index=0,
        poll_state=lambda _index: next(responses),
    )

    assert controller.snapshot() == pytest.approx([1.0, 0.0, 0.0])
    assert controller.snapshot() == pytest.approx([0.0, 0.0, 0.0])
    assert controller.snapshot() == pytest.approx([-0.6, 0.0, 0.0])


@pytest.mark.parametrize("index", [-1, 4])
def test_gamepad_index_must_be_an_xinput_slot(index):
    with pytest.raises(ValueError, match=r"\[0, 3\]"):
        sim2sim.GamepadCommands(index=index, poll_state=lambda _index: None)
