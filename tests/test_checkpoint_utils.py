from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "loco_transformer" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import checkpoint_utils  # noqa: E402


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--load_run", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    checkpoint_utils.add_finetune_args(parser)
    return parser


@pytest.mark.parametrize("option", ["--finetune-path", "--finetune_path"])
def test_finetune_path_aliases_are_normalized(option):
    checkpoint = Path(__file__).resolve()
    parser = _make_parser()
    args = parser.parse_args([option, str(checkpoint)])

    result = checkpoint_utils.validate_finetune_args(args, parser)

    assert result == str(checkpoint.resolve())
    assert args.finetune_path == str(checkpoint.resolve())


def test_finetune_path_must_exist():
    parser = _make_parser()
    missing_checkpoint = SCRIPTS_DIR / "definitely_missing_checkpoint.pt"
    args = parser.parse_args(["--finetune-path", str(missing_checkpoint)])

    with pytest.raises(SystemExit, match="2"):
        checkpoint_utils.validate_finetune_args(args, parser)


@pytest.mark.parametrize(
    "conflicting_args",
    [
        ["--resume"],
        ["--load_run", "some-run"],
        ["--checkpoint", "model_9000.pt"],
    ],
)
def test_finetune_rejects_resume_options(conflicting_args):
    checkpoint = Path(__file__).resolve()
    parser = _make_parser()
    args = parser.parse_args(["--finetune-path", str(checkpoint), *conflicting_args])

    with pytest.raises(SystemExit, match="2"):
        checkpoint_utils.validate_finetune_args(args, parser)


@pytest.mark.parametrize("runner_class", ["AMPRunner", "DistillationRunner"])
def test_finetune_rejects_non_on_policy_runner(runner_class):
    with pytest.raises(ValueError, match="only OnPolicyRunner"):
        checkpoint_utils.validate_finetune_runner(runner_class)


class _FakeRunner:
    def __init__(self):
        self.current_learning_iteration = 0
        self.events = []

    def load(self, path, **kwargs):
        self.events.append(("load", path, kwargs))
        self.current_learning_iteration = 9000

    def save(self, path, infos=None):
        self.events.append(("save", path, infos))


def test_resume_restores_optimizer_and_iteration():
    runner = _FakeRunner()

    source_iteration = checkpoint_utils.load_checkpoint_for_training(
        runner,
        "resume.pt",
        mode="resume",
        map_location="cuda:0",
    )

    assert source_iteration is None
    assert runner.current_learning_iteration == 9000
    assert runner.events == [("load", "resume.pt", {"map_location": "cuda:0"})]


def test_finetune_loads_weights_only_resets_iteration_and_saves_snapshot():
    runner = _FakeRunner()
    source = SCRIPTS_DIR / "source.pt"
    snapshot = SCRIPTS_DIR / "model_loaded.pt"

    source_iteration = checkpoint_utils.load_checkpoint_for_training(
        runner,
        str(source),
        mode="finetune",
        map_location="cuda:0",
        snapshot_path=str(snapshot),
    )

    assert source_iteration == 9000
    assert runner.current_learning_iteration == 0
    assert runner.events[0] == (
        "load",
        str(source),
        {"load_optimizer": False, "map_location": "cuda:0"},
    )
    assert runner.events[1] == (
        "save",
        str(snapshot),
        {
            "finetune_source": {
                "checkpoint": str(source.resolve()),
                "iteration": 9000,
            }
        },
    )


def test_resume_checkpoint_lookup_skips_newest_empty_run():
    log_root = str(Path("logs") / "experiment")

    class _Entry:
        def __init__(self, name):
            self.name = name

        def is_dir(self):
            return True

    entries = [_Entry("2026-08-18"), _Entry("2026-08-19"), _Entry("2026-08-20")]
    timestamps = {
        str(Path(log_root) / "2026-08-18"): 1,
        str(Path(log_root) / "2026-08-19"): 2,
        str(Path(log_root) / "2026-08-20"): 3,
    }
    files = {
        str(Path(log_root) / "2026-08-18"): ["model_500.pt"],
        str(Path(log_root) / "2026-08-19"): ["model_9000.pt", "model_10000.pt"],
        str(Path(log_root) / "2026-08-20"): [],
    }

    with (
        patch.object(checkpoint_utils.os, "scandir", return_value=entries),
        patch.object(checkpoint_utils.os.path, "getmtime", side_effect=timestamps.__getitem__),
        patch.object(checkpoint_utils.os, "listdir", side_effect=files.__getitem__),
    ):
        result = checkpoint_utils.get_resume_checkpoint_path(
            log_root,
            ".*",
            "model_.*.pt",
        )

    assert result == str(Path(log_root) / "2026-08-19" / "model_10000.pt")


def test_training_entrypoint_no_longer_contains_explicit_resume_path():
    train_source = (SCRIPTS_DIR / "train.py").read_text(encoding="utf-8")
    for removed_name in ("resume_path", "resume-path", "explicit_resume_path"):
        assert removed_name not in train_source
