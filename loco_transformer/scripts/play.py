# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play script for the RPO-Loco-Transformer task.

Loads a trained checkpoint from ``logs/rsl_rl/loco_transformer``, exports
``policy.pt`` and ``policy.onnx``, then launches the Isaac Sim window with
keyboard velocity control.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a trained RPO-Loco-Transformer agent.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
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
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import loco_transformer  # noqa: F401


# ------------------------------------------------------------------
# Cross-attention policy exporters
# ------------------------------------------------------------------

class CrossAttnTorchPolicyExporter(torch.nn.Module):
    """Export CrossAttentionActorCritic policy to TorchScript JIT."""

    def __init__(self, policy, normalizer=None):
        super().__init__()
        self.actor = copy.deepcopy(policy.actor)
        self.encoder = copy.deepcopy(policy.encoder)
        self.num_actor_obs = policy.num_actor_obs
        self.actor_perception_range = policy.actor_perception_range
        self.H = policy.H
        self.W = policy.W
        self.state_dependent_std = policy.state_dependent_std
        if normalizer and not isinstance(normalizer, torch.nn.Identity):
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward(self, x):
        flat_obs = self.normalizer(x)
        s, e = self.actor_perception_range
        proprio = torch.cat([flat_obs[:, :s], flat_obs[:, e:]], dim=-1)
        height_scan = flat_obs[:, s:e]
        height_map = height_scan.view(-1, self.H, self.W)
        embedding = self.encoder(proprio, height_map)[0]
        actor_input = torch.cat([proprio, embedding], dim=-1)
        actions = self.actor(actor_input)
        if self.state_dependent_std:
            actions = actions[..., 0, :]
        return actions

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class CrossAttnOnnxPolicyExporter(torch.nn.Module):
    """Export CrossAttentionActorCritic policy to ONNX."""

    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.actor = copy.deepcopy(policy.actor)
        self.encoder = copy.deepcopy(policy.encoder)
        self.num_actor_obs = policy.num_actor_obs
        self.actor_perception_range = policy.actor_perception_range
        self.H = policy.H
        self.W = policy.W
        self.state_dependent_std = policy.state_dependent_std
        if normalizer and not isinstance(normalizer, torch.nn.Identity):
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward(self, x):
        flat_obs = self.normalizer(x)
        s, e = self.actor_perception_range
        proprio = torch.cat([flat_obs[:, :s], flat_obs[:, e:]], dim=-1)
        height_scan = flat_obs[:, s:e]
        height_map = height_scan.view(-1, self.H, self.W)
        embedding = self.encoder(proprio, height_map)[0]
        actor_input = torch.cat([proprio, embedding], dim=-1)
        actions = self.actor(actor_input)
        if self.state_dependent_std:
            actions = actions[..., 0, :]
        return actions

    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        opset_version = 18
        obs = torch.zeros(1, self.num_actor_obs)
        torch.onnx.export(
            self,
            obs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=opset_version,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={},
        )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Load a trained policy, export models, and run interactive play with keyboard control."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.scene.env_spacing = 2.5

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # rsl-rl 3.x compatibility
    if version.parse(installed_version) < version.parse("5.0.0"):
        for key in ("optimizer", "share_cnn_encoders"):
            if hasattr(agent_cfg.algorithm, key):
                delattr(agent_cfg.algorithm, key)

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # disable domain randomization for play
    env_cfg.observations.policy.enable_corruption = False
    if not args_cli.push_robot:
        env_cfg.events.push_robot = None
    env_cfg.episode_length_s = 40.0
    env_cfg.commands.heading_command = False
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.rel_heading_envs = 0.0

    if args_cli.plane:
        env_cfg.scene.terrain.terrain_generator = None
        env_cfg.scene.terrain.terrain_type = "plane"

    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False
        env_cfg.scene.terrain.terrain_generator.difficulty_range = (1.0, 1.0)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # zero out commands — keyboard will control them instead
    if hasattr(env.unwrapped, "command_generator"):
        # DirectRLEnv-style (robolab tasks)
        env.unwrapped.command_generator.command[:, 0] = 0.0
        env.unwrapped.command_generator.command[:, 1] = 0.0
        env.unwrapped.command_generator.command[:, 2] = 0.0
    elif hasattr(env.unwrapped, "command_manager"):
        # ManagerBasedRLEnv-style (loco_transformer)
        cmd = env.unwrapped.command_manager.get_command("base_velocity")
        cmd[:, 0] = 0.0
        cmd[:, 1] = 0.0
        cmd[:, 2] = 0.0

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
        print("[INFO] Recording videos during play.")
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
    if hasattr(policy_nn, "encoder") and hasattr(policy_nn, "actor_perception_range"):
        # Cross-attention policy — use custom exporters
        torch_policy_exporter = CrossAttnTorchPolicyExporter(policy_nn, normalizer)
        torch_policy_exporter.export(path=export_model_dir, filename="policy.pt")
        onnx_policy_exporter = CrossAttnOnnxPolicyExporter(policy_nn, normalizer, verbose=False)
        onnx_policy_exporter.export(path=export_model_dir, filename="policy.onnx")
        print(f"[INFO] Exported cross-attention policy to {export_model_dir}")
    else:
        # Standard MLP policy — use built-in exporters
        export_policy_as_jit(policy_nn, normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(
            policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx"
        )
        print(f"[INFO] Exported standard policy to {export_model_dir}")

    # keyboard control
    if not args_cli.headless:
        from robolab.utils.keyboard import Keyboard
        keyboard = Keyboard(env)  # noqa: F841

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
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
