# Copyright (c) 2025-2026, Loco-Transformer Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive MuJoCo Sim2Sim playback for a Loco-Transformer policy.

The script consumes the flattened ``policy.pt`` exported by ``play.py`` and
reconstructs the 309-dimensional training observation, including the 21 x 11
yaw-aligned height scan.
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch


NUM_ACTIONS = 23
ACTION_SCALE = 0.25
ANGULAR_VELOCITY_SCALE = 0.25
HEIGHT_SCAN_OFFSET = 0.5
HEIGHT_SCAN_SHAPE = (11, 21)  # (y rows, x columns), matching the training policy
HEIGHT_SCAN_RESOLUTION = 0.1

# MuJoCo stores the joints in the depth-first URDF order below.  Isaac Sim's
# articulation/action order is breadth-first; this is the same verified mapping
# used by RoboLab's RPO Sim2Sim scripts.
ISAAC_TO_MUJOCO = np.asarray(
    [0, 6, 12, 1, 7, 13, 18, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22],
    dtype=np.int32,
)

MUJOCO_JOINT_NAMES = (
    "left_thigh_yaw_joint",
    "left_thigh_roll_joint",
    "left_thigh_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_thigh_yaw_joint",
    "right_thigh_roll_joint",
    "right_thigh_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "torso_joint",
    "left_arm_pitch_joint",
    "left_arm_roll_joint",
    "left_arm_yaw_joint",
    "left_elbow_pitch_joint",
    "left_elbow_yaw_joint",
    "right_arm_pitch_joint",
    "right_arm_roll_joint",
    "right_arm_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_yaw_joint",
)

DEFAULT_POS = np.asarray(
    [0, 0, -0.1, 0.3, -0.2, 0, 0, 0, -0.1, 0.3, -0.2, 0, 0, 0.18, 0.06, 0, 0.78, 0,
     0.18, -0.06, 0, 0.78, 0],
    dtype=np.float64,
)
KP = np.asarray(
    [100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40, 150, 40, 40, 40, 30, 20,
     40, 40, 40, 30, 20],
    dtype=np.float64,
)
KD = np.asarray(
    [3.3, 3.3, 3.3, 5.0, 2.0, 2.0, 3.3, 3.3, 3.3, 5.0, 2.0, 2.0, 5.0, 2.0, 2.0,
     2.0, 1.5, 1.0, 2.0, 2.0, 2.0, 1.5, 1.0],
    dtype=np.float64,
)
# Match RPO_CFG instead of the overly permissive 200 Nm used by the reference script.
EFFORT_LIMIT = np.asarray(
    [120, 120, 120, 120, 27, 27, 120, 120, 120, 120, 27, 27, 120,
     27, 27, 27, 27, 27, 27, 27, 27, 27, 27],
    dtype=np.float64,
)


class KeyboardCommands:
    """Thread-safe command state updated by the MuJoCo viewer callback.

    I/K, J/L, U/O and P are deliberately used instead of MuJoCo's built-in
    Space, Backspace, Tab, arrow, function and number-key shortcuts.
    """

    _RANGES = ((-0.6, 1.0), (-0.5, 0.5), (-1.57, 1.57))

    def __init__(self, linear_step: float, angular_step: float) -> None:
        self.linear_step = linear_step
        self.angular_step = angular_step
        self._value = np.zeros(3, dtype=np.float32)
        self._reset_requested = False
        self._lock = threading.Lock()

    def on_key(self, keycode: int) -> None:
        try:
            key = chr(keycode).upper()
        except (TypeError, ValueError):
            return

        changes = {
            "I": (0, self.linear_step),
            "K": (0, -self.linear_step),
            "J": (1, self.linear_step),
            "L": (1, -self.linear_step),
            "U": (2, self.angular_step),
            "O": (2, -self.angular_step),
        }
        with self._lock:
            if key in changes:
                index, delta = changes[key]
                low, high = self._RANGES[index]
                self._value[index] = np.clip(self._value[index] + delta, low, high)
            elif key == "P":
                self._value[:] = 0.0
            elif key == "R":
                self._value[:] = 0.0
                self._reset_requested = True
            else:
                return
            value = self._value.copy()
        print(f"[Keyboard] vx={value[0]:+.2f} m/s, vy={value[1]:+.2f} m/s, wz={value[2]:+.2f} rad/s")

    def snapshot(self) -> np.ndarray:
        with self._lock:
            return self._value.copy()

    def consume_reset(self) -> bool:
        with self._lock:
            requested = self._reset_requested
            self._reset_requested = False
            return requested


def _joint_addresses(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    qpos_addresses = []
    dof_addresses = []
    for name in MUJOCO_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo model is missing required joint: {name}")
        qpos_addresses.append(model.jnt_qposadr[joint_id])
        dof_addresses.append(model.jnt_dofadr[joint_id])
    return np.asarray(qpos_addresses), np.asarray(dof_addresses)


def _reset_robot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_addresses: np.ndarray,
) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[qpos_addresses] = DEFAULT_POS
    mujoco.mj_forward(model, data)


def _projected_gravity(data: mujoco.MjData) -> np.ndarray:
    quat_wxyz = np.asarray(data.sensor("orientation").data, dtype=np.float64)
    rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, quat_wxyz)
    return rotation.reshape(3, 3).T @ np.asarray([0.0, 0.0, -1.0])


def _height_scan_offsets() -> np.ndarray:
    rows, columns = HEIGHT_SCAN_SHAPE
    x = (np.arange(columns) - (columns - 1) / 2.0) * HEIGHT_SCAN_RESOLUTION
    y = (np.arange(rows) - (rows - 1) / 2.0) * HEIGHT_SCAN_RESOLUTION
    grid_x, grid_y = np.meshgrid(x, y, indexing="xy")
    return np.column_stack((grid_x.ravel(), grid_y.ravel()))


def _height_scan(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    local_offsets: np.ndarray,
    torso_body_id: int,
) -> np.ndarray:
    torso_pos = data.xpos[torso_body_id]
    rotation = np.asarray(data.xmat[torso_body_id]).reshape(3, 3)
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    yaw_rotation = np.asarray([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    world_offsets = local_offsets @ yaw_rotation.T

    geom_group = np.asarray([1, 0, 0, 0, 0, 0], dtype=np.uint8)  # terrain is MuJoCo geom group 0
    ray_direction = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    geom_id = np.empty(1, dtype=np.int32)
    heights = np.empty(len(world_offsets), dtype=np.float32)
    for index, offset in enumerate(world_offsets):
        start = np.asarray(torso_pos, dtype=np.float64).copy()
        start[:2] += offset
        start[2] += 5.0
        distance = mujoco.mj_ray(model, data, start, ray_direction, geom_group, 1, torso_body_id, geom_id)
        if distance < 0.0:
            heights[index] = -5.0
        else:
            hit_z = start[2] - distance
            heights[index] = np.clip(torso_pos[2] - hit_z - HEIGHT_SCAN_OFFSET, -5.0, 5.0)
    return heights


def _observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_addresses: np.ndarray,
    dof_addresses: np.ndarray,
    last_action: np.ndarray,
    command: np.ndarray,
    scan_offsets: np.ndarray,
    torso_body_id: int,
) -> np.ndarray:
    joint_pos_mujoco = data.qpos[qpos_addresses] - DEFAULT_POS
    joint_vel_mujoco = data.qvel[dof_addresses]
    joint_pos_isaac = joint_pos_mujoco[ISAAC_TO_MUJOCO]
    joint_vel_isaac = joint_vel_mujoco[ISAAC_TO_MUJOCO]
    angular_velocity = np.asarray(data.sensor("angular-velocity").data) * ANGULAR_VELOCITY_SCALE

    obs = np.concatenate(
        (
            angular_velocity,
            _projected_gravity(data),
            command,
            joint_pos_isaac,
            joint_vel_isaac,
            last_action,
            _height_scan(model, data, scan_offsets, torso_body_id),
        )
    ).astype(np.float32)
    if obs.shape != (309,):
        raise RuntimeError(f"Expected a 309-dimensional policy observation, got {obs.shape}.")
    return obs


def _policy_action(policy: torch.jit.ScriptModule, observation: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        output = policy(torch.from_numpy(observation).unsqueeze(0))
    if isinstance(output, (tuple, list)):
        output = output[0]
    action = output.squeeze(0).detach().cpu().numpy().astype(np.float64)
    if action.shape != (NUM_ACTIONS,):
        raise RuntimeError(f"Expected {NUM_ACTIONS} policy actions, got {action.shape}.")
    return action


def run(args: argparse.Namespace) -> None:
    model = mujoco.MjModel.from_xml_path(str(args.model))
    model.opt.timestep = args.dt
    data = mujoco.MjData(model)
    qpos_addresses, dof_addresses = _joint_addresses(model)
    torso_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    if torso_body_id < 0:
        raise ValueError("MuJoCo model is missing required body: torso_link")
    _reset_robot(model, data, qpos_addresses)

    policy = torch.jit.load(str(args.load_model), map_location="cpu")
    policy.eval()
    keyboard = KeyboardCommands(args.lin_vel_step, args.ang_vel_step)
    scan_offsets = _height_scan_offsets()
    action = np.zeros(NUM_ACTIONS, dtype=np.float64)
    target_pos = DEFAULT_POS.copy()
    start_sim_time = data.time

    print("[MuJoCo] Interactive velocity control (keys chosen to avoid viewer shortcuts):")
    print("  I/K: forward/backward     J/L: strafe left/right")
    print("  U/O: turn left/right      P: stop      R: reset robot")
    print("  Close the viewer window or press Esc to exit.")

    with mujoco.viewer.launch_passive(model, data, key_callback=keyboard.on_key) as viewer:
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
            viewer.cam.distance = 4.0
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -20.0

        while viewer.is_running():
            wall_step_start = time.perf_counter()
            if keyboard.consume_reset():
                _reset_robot(model, data, qpos_addresses)
                action.fill(0.0)
                target_pos[:] = DEFAULT_POS
                start_sim_time = data.time

            command = keyboard.snapshot()
            obs = _observation(
                model,
                data,
                qpos_addresses,
                dof_addresses,
                action,
                command,
                scan_offsets,
                torso_body_id,
            )
            action = _policy_action(policy, obs)
            target_pos[:] = DEFAULT_POS
            target_pos[ISAAC_TO_MUJOCO] += action * ACTION_SCALE

            for _ in range(args.decimation):
                q = data.qpos[qpos_addresses]
                dq = data.qvel[dof_addresses]
                torque = (target_pos - q) * KP - dq * KD
                data.ctrl[:] = np.clip(torque, -EFFORT_LIMIT, EFFORT_LIMIT)
                mujoco.mj_step(model, data)

            viewer.sync()
            if args.duration > 0.0 and data.time - start_sim_time >= args.duration:
                break
            if not args.no_real_time:
                sleep_time = args.dt * args.decimation - (time.perf_counter() - wall_step_start)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)


def _parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[2]
    asset_dir = workspace / "robolab" / "data" / "robots" / "roboparty" / "rpo" / "mjcf"
    parser = argparse.ArgumentParser(description="Visualize an exported Loco-Transformer policy in MuJoCo.")
    parser.add_argument("--load-model", "--load_model", dest="load_model", type=Path, required=True,
                        help="Path to exported/policy.pt from loco_transformer/scripts/play.py.")
    parser.add_argument("--model", type=Path, default=None, help="Override the RPO MuJoCo XML path.")
    parser.add_argument("--terrain", action="store_true", help="Use rpo_terrain.xml instead of the plane model.")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Simulation duration in seconds; 0 keeps running until the viewer closes.")
    parser.add_argument("--dt", type=float, default=0.001, help="MuJoCo physics time step.")
    parser.add_argument("--decimation", type=int, default=20, help="Physics steps per 50 Hz policy step.")
    parser.add_argument("--lin-vel-step", type=float, default=0.05, help="Keyboard linear-velocity increment.")
    parser.add_argument("--ang-vel-step", type=float, default=0.05, help="Keyboard angular-velocity increment.")
    parser.add_argument("--no-real-time", action="store_true", help="Disable wall-clock pacing.")
    args = parser.parse_args()
    if args.model is None:
        args.model = asset_dir / ("rpo_terrain.xml" if args.terrain else "rpo.xml")
    if not args.load_model.is_file():
        parser.error(f"policy file does not exist: {args.load_model}")
    if not args.model.is_file():
        parser.error(f"MuJoCo model does not exist: {args.model}")
    if args.dt <= 0.0 or args.decimation <= 0:
        parser.error("--dt and --decimation must be positive")
    return args


if __name__ == "__main__":
    run(_parse_args())
