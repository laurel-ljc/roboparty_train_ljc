"""Unit tests for reward math that is independent of the simulator state."""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _load_reward_math_module():
    module_path = (
        Path(__file__).parents[1]
        / "loco_transformer"
        / "mdp"
        / "reward_math.py"
    )
    spec = importlib.util.spec_from_file_location("loco_reward_math", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


joint_pos_limit_violation = _load_reward_math_module().joint_pos_limit_violation


def test_joint_pos_limit_violation_is_non_negative_on_both_sides():
    limits = torch.tensor([[[-1.0, 1.0], [-0.5, 0.5], [-2.0, 2.0]]])
    joint_pos = torch.tensor([[0.0, -0.75, 2.25]])

    violation = joint_pos_limit_violation(joint_pos, limits)

    torch.testing.assert_close(violation, torch.tensor([[0.0, 0.25, 0.25]]))
    assert torch.all(violation >= 0.0)


def test_joint_pos_limit_violation_is_zero_at_limits():
    limits = torch.tensor([[[-1.0, 1.0], [-0.5, 0.5]]])
    joint_pos = torch.tensor([[-1.0, 0.5]])

    violation = joint_pos_limit_violation(joint_pos, limits)

    torch.testing.assert_close(violation, torch.zeros_like(joint_pos))
