# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play script for the RPO-Loco-Transformer task family.

Loads a trained checkpoint from the registered task's experiment directory,
exports ``policy.pt`` and ``policy.onnx``, then launches the Isaac Sim window
with keyboard velocity control.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play and export a Loco-Transformer policy with RSL-RL.")
parser.add_argument("--task", type=str, default="RPO-Loco-Transformer-Play", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--plane", action="store_true", help="Use plane terrain")
parser.add_argument("--push_robot", action="store_true", help="Push robot during playing")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playing.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--lin-vel-step", type=float, default=0.05, help="Keyboard linear-velocity increment.")
parser.add_argument("--ang-vel-step", type=float, default=0.05, help="Keyboard angular-velocity increment.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import gymnasium as gym
import os
import re
import time
import torch
import copy

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import loco_transformer  # noqa: F401


def get_play_checkpoint_path(log_path: str, run_pattern: str, checkpoint_pattern: str) -> str:
    """Find the newest checkpoint while skipping run directories without models."""
    if not os.path.isdir(log_path):
        raise ValueError(f"Experiment directory does not exist: {log_path}")
    runs = [entry.path for entry in os.scandir(log_path) if entry.is_dir() and re.match(run_pattern, entry.name)]
    for run_path in sorted(runs, key=os.path.getmtime, reverse=True):
        checkpoints = [name for name in os.listdir(run_path) if re.match(checkpoint_pattern, name)]
        if checkpoints:
            checkpoints.sort(key=lambda name: f"{name:0>15}")
            return os.path.join(run_path, checkpoints[-1])
    raise ValueError(
        f"No checkpoint in '{log_path}' matched run '{run_pattern}' and model '{checkpoint_pattern}'."
    )


class CrossAttentionPolicyExporter(torch.nn.Module):
    """Flattened-tensor inference wrapper for ``CrossAttentionActorCritic``."""

    def __init__(self, policy, normalizer=None):
        super().__init__()
        self.actor = copy.deepcopy(policy.actor)
        self.encoder = copy.deepcopy(policy.encoder)
        self.normalizer = copy.deepcopy(normalizer) if normalizer is not None else torch.nn.Identity()
        self.num_actor_obs = policy.num_actor_obs
        self.perception_start, self.perception_end = policy.actor_perception_range
        self.height_map_shape = policy.height_map_shape

    def forward(self, obs):
        obs = self.normalizer(obs)
        proprio = torch.cat(
            (obs[:, : self.perception_start], obs[:, self.perception_end :]), dim=-1
        )
        height_map = obs[:, self.perception_start : self.perception_end].reshape(
            -1, self.height_map_shape[0], self.height_map_shape[1]
        )
        embedding, _ = self.encoder(proprio, height_map)
        return self.actor(torch.cat((proprio, embedding), dim=-1))

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        self.to("cpu").eval()
        example = torch.zeros(1, self.num_actor_obs)
        with torch.inference_mode():
            traced = torch.jit.trace(self, example, strict=False)
            traced.save(os.path.join(path, "policy.pt"))
            torch.onnx.export(
                self,
                example,
                os.path.join(path, "policy.onnx"),
                export_params=True,
                opset_version=18,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
            )


def main():
    # load configurations from gym registry
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg = load_cfg_from_registry(
        args_cli.task, "env_cfg_entry_point"
    )
    agent_cfg: RslRlBaseRunnerCfg = load_cfg_from_registry(args_cli.task, args_cli.agent)

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    # The existing Loco-Transformer checkpoints were trained with the original
    # 256-128-64 MLP heads and without observation normalization.  Keep playback
    # pinned to that architecture even if the registered agent config drifts.
    agent_cfg.policy.actor_hidden_dims = [256, 128, 64]
    agent_cfg.policy.critic_hidden_dims = [256, 128, 64]
    agent_cfg.policy.actor_obs_normalization = False
    agent_cfg.policy.critic_obs_normalization = False
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.scene.env_spacing = 2.5

    # rsl-rl 3.x compatibility
    if version.parse(installed_version) < version.parse("5.0.0"):
        for key in ("optimizer", "share_cnn_encoders"):
            if hasattr(agent_cfg.algorithm, key):
                delattr(agent_cfg.algorithm, key)

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # The Loco-Transformer task is manager-based.  Configure the nested
    # observation/command terms instead of the direct-environment attributes
    # used by RoboLab's original play script.
    if hasattr(env_cfg.observations, "policy"):
        env_cfg.observations.policy.enable_corruption = False
    if not args_cli.push_robot and hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
    env_cfg.episode_length_s = 40.0
    command_cfg = env_cfg.commands.base_velocity
    command_cfg.heading_command = False
    command_cfg.rel_heading_envs = 0.0
    command_cfg.rel_standing_envs = 0.0
    # Prevent periodic random command resampling from overwriting keyboard input.
    command_cfg.resampling_time_range = (1.0e9, 1.0e9)
    # Avoid remote arrow-marker assets and visual ambiguity: keyboard commands
    # remain available, but desired/current velocity arrows are not spawned.
    command_cfg.debug_vis = False
    command_cfg.ranges.lin_vel_x = (0.0, 0.0)
    command_cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_cfg.ranges.ang_vel_z = (0.0, 0.0)

    if args_cli.plane:
        env_cfg.scene.terrain.terrain_generator = None
        env_cfg.scene.terrain.terrain_type = "plane"

    if env_cfg.scene.terrain.terrain_generator is not None:
        terrain_generator = env_cfg.scene.terrain.terrain_generator
        # Registered Play tasks may deliberately choose a particular grid
        # shape (History-Rough-Play uses one row and twenty terrain types).
        # Only compact a training task when it is launched through play.py.
        if not args_cli.task.endswith("-Play"):
            terrain_generator.num_rows = 5
            terrain_generator.num_cols = 5
        terrain_generator.curriculum = False
        terrain_generator.difficulty_range = (1.0, 1.0)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_play_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    command = env.unwrapped.command_manager.get_command("base_velocity")
    command[:, :3] = 0.0

    # convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path, map_location=agent_cfg.device)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if not os.path.exists(export_model_dir):
        os.makedirs(export_model_dir, exist_ok=True)
    if all(hasattr(policy_nn, name) for name in ("encoder", "actor_perception_range", "height_map_shape")):
        CrossAttentionPolicyExporter(policy_nn, normalizer).export(export_model_dir)
    else:
        # Standard MLP policy — use built-in exporters
        export_policy_as_jit(policy_nn, normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(
            policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx"
        )
    print(f"[INFO] Exported policy files to: {export_model_dir}")

    # keyboard control
    if not args_cli.headless:
        from loco_transformer.utils import VelocityCommandKeyboard

        keyboard = VelocityCommandKeyboard(
            env,
            lin_vel_step=args_cli.lin_vel_step,
            ang_vel_step=args_cli.ang_vel_step,
        )  # keep a strong reference for the lifetime of the play loop

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            if not args_cli.headless:
                keyboard.advance()
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
