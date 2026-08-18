"""Pure observation-layout tests for Loco-Transformer MuJoCo playback."""

import importlib.util
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

_SIM2SIM_PATH = Path(__file__).parents[1] / "loco_transformer" / "scripts" / "sim2sim.py"
_SPEC = importlib.util.spec_from_file_location("loco_transformer_sim2sim", _SIM2SIM_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_SIM2SIM = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SIM2SIM)

CURRENT_OBSERVATION_DIM = _SIM2SIM.CURRENT_OBSERVATION_DIM
HISTORY_OBSERVATION_DIM = _SIM2SIM.HISTORY_OBSERVATION_DIM
ProprioceptiveHistory = _SIM2SIM.ProprioceptiveHistory
ProprioceptiveState = _SIM2SIM.ProprioceptiveState
_assemble_observation = _SIM2SIM._assemble_observation
_detect_policy_observation_dim = _SIM2SIM._detect_policy_observation_dim


def _state(value: float) -> ProprioceptiveState:
    return ProprioceptiveState(
        angular_velocity=np.full(3, value, dtype=np.float32),
        projected_gravity=np.full(3, value + 1, dtype=np.float32),
        joint_position=np.full(23, value + 2, dtype=np.float32),
        joint_velocity=np.full(23, value + 3, dtype=np.float32),
    )


def test_history_observation_layout_reset_fill_and_shift():
    history = ProprioceptiveHistory()
    command = np.asarray([10, 11, 12], dtype=np.float32)
    action = np.arange(23, dtype=np.float32)
    height = np.arange(231, dtype=np.float32)

    first = _assemble_observation(_state(1), command, action, height, history)
    assert first.shape == (HISTORY_OBSERVATION_DIM,)
    np.testing.assert_array_equal(first[0:30].reshape(10, 3), np.full((10, 3), 1))
    np.testing.assert_array_equal(first[30:60].reshape(10, 3), np.full((10, 3), 2))
    np.testing.assert_array_equal(first[60:63], command)
    np.testing.assert_array_equal(first[63:293].reshape(10, 23), np.full((10, 23), 3))
    np.testing.assert_array_equal(first[293:523].reshape(10, 23), np.full((10, 23), 4))
    np.testing.assert_array_equal(first[523:546], action)
    np.testing.assert_array_equal(first[546:777], height)

    second = _assemble_observation(_state(5), command, action, height, history)
    angular_history = second[0:30].reshape(10, 3)
    np.testing.assert_array_equal(angular_history[:-1], np.full((9, 3), 1))
    np.testing.assert_array_equal(angular_history[-1], np.full(3, 5))

    history.clear()
    reset = _assemble_observation(_state(7), command, action, height, history)
    np.testing.assert_array_equal(reset[0:30].reshape(10, 3), np.full((10, 3), 7))


def test_original_observation_layout_remains_309_dimensional():
    command = np.asarray([10, 11, 12], dtype=np.float32)
    action = np.arange(23, dtype=np.float32)
    height = np.arange(231, dtype=np.float32)

    observation = _assemble_observation(_state(1), command, action, height)

    assert observation.shape == (CURRENT_OBSERVATION_DIM,)
    np.testing.assert_array_equal(observation[0:3], np.full(3, 1))
    np.testing.assert_array_equal(observation[3:6], np.full(3, 2))
    np.testing.assert_array_equal(observation[6:9], command)
    np.testing.assert_array_equal(observation[78:309], height)


class _FixedInputPolicy(torch.nn.Module):
    def __init__(self, observation_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(observation_dim, 23)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.linear(observation)


@pytest.mark.parametrize("observation_dim", [CURRENT_OBSERVATION_DIM, HISTORY_OBSERVATION_DIM])
def test_policy_observation_dimension_detection(observation_dim):
    policy = torch.jit.script(_FixedInputPolicy(observation_dim))

    assert _detect_policy_observation_dim(policy) == observation_dim
