# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MuJoCo sim2sim deployment for the RPO-Loco-Transformer task.

Loads an ONNX policy exported by ``play.py``, runs a MuJoCo simulation with
PD control, and accepts keyboard velocity commands so the user can see the
training results in real time.
"""

import argparse
import os
import sys
import threading
import time

import mujoco
import mujoco_viewer
import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from robolab.assets import ISAAC_DATA_DIR

# ------------------------------------------------------------------
# Keyboard velocity command (updated by listener thread)
# ------------------------------------------------------------------


class CmdState:
    """Mutable velocity command shared with the keyboard listener."""

    def __init__(self):
        self.vx = 0.0
        self.vy = 0.0
        self.dyaw = 0.0
        self.lin_vel_step = 0.05
        self.ang_vel_step = 0.05


cmd = CmdState()


def _start_keyboard_listener():
    """Start a background thread that listens for WASD+QE key presses."""
    try:
        from pynput import keyboard as pynput_keyboard
    except ImportError:
        print("[WARN] pynput not installed. Run: pip install pynput")
        print("[WARN] Using default zero-velocity command. Edit cmd.vx/vy/dyaw in code.")
        return None

    def on_press(key):
        try:
            k = key.char
        except AttributeError:
            return  # ignore special keys

        if k == "w":
            cmd.vx += cmd.lin_vel_step
            _print_cmd()
        elif k == "s":
            cmd.vx -= cmd.lin_vel_step
            _print_cmd()
        elif k == "a":
            cmd.vy += cmd.lin_vel_step
            _print_cmd()
        elif k == "d":
            cmd.vy -= cmd.lin_vel_step
            _print_cmd()
        elif k == "q":
            cmd.dyaw += cmd.ang_vel_step
            _print_cmd()
        elif k == "e":
            cmd.dyaw -= cmd.ang_vel_step
            _print_cmd()
        elif k == "x":
            cmd.vx = 0.0
            cmd.vy = 0.0
            cmd.dyaw = 0.0
            _print_cmd()

    def _print_cmd():
        print(f"\r[Keyboard] vx={cmd.vx:+.2f}  vy={cmd.vy:+.2f}  dyaw={cmd.dyaw:+.2f}", end="")

    listener = pynput_keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()
    return listener


# ------------------------------------------------------------------
# Ray casting helpers (match Isaac Lab RayCaster GridPattern)
# ------------------------------------------------------------------


def get_rays(model, data, pos, num_points, offset_xy, scanner_z):
    """Cast rays downward from ``scanner_z + 20`` and return distance from scanner_z to ground.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        pos: Torso world position (x, y, z).
        num_points: Number of grid points.
        offset_xy: (num_points, 2) XY offsets in world frame.
        scanner_z: World Z of the scanner origin.

    Returns:
        dist: (num_points,) distances from scanner_z to ground at each grid point.
    """
    dist = np.zeros(num_points, dtype=np.float64)
    ray_vec = np.array([0, 0, -1], dtype=np.float64)
    geomgroup = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    geomid = np.zeros(1, dtype=np.int32)
    for i in range(num_points):
        pt = pos.copy()
        pt[:2] += offset_xy[i]
        pt[2] = scanner_z + 20.0  # cast from well above
        dist[i] = mujoco.mj_ray(model, data, pt, ray_vec, geomgroup, 1, -1, geomid)
        dist[i] -= 20.0  # remove safety margin → distance from scanner_z to ground
    return dist


def get_obs(data, default_pos):
    """Extract proprioceptive observation from MuJoCo data.

    Returns:
        q: joint positions (full dof).
        dq: joint velocities (full dof).
        quat: base orientation quaternion (x, y, z, w).
        v: base linear velocity in body frame.
        omega: base angular velocity in body frame.
        gvec: projected gravity in body frame.
    """
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)
    omega = data.sensor("angular-velocity").data.astype(np.double)
    gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """PD position controller: tau = (target_q - q) * kp + (target_dq - dq) * kd."""
    return (target_q - q) * kp + (target_dq - dq) * kd


# ------------------------------------------------------------------
# Main simulation loop
# ------------------------------------------------------------------


def run_mujoco(policy, cfg, headless=False):
    """Run MuJoCo simulation with the given ONNX policy.

    Args:
        policy: ONNX inference session.
        cfg: Configuration object (see :class:`Sim2simCfg`).
        headless: If True, render off-screen and save video.
    """
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    data.qpos[-cfg.robot_config.num_actions :] = cfg.robot_config.default_pos
    mujoco.mj_step(model, data)

    os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
    os.environ["MUJOCO_GL"] = "glfw"

    # -- rendering setup --
    if headless:
        renderer = mujoco.Renderer(model, width=1920, height=1080)
        cam = mujoco.MjvCamera()
        cam.distance = 4.0
        cam.azimuth = 45.0
        cam.elevation = -20.0
        cam.lookat = [0, 0, 1]
    else:
        viewer = mujoco_viewer.MujocoViewer(model, data, mode="window", width=1920, height=1080)
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 45.0
        viewer.cam.elevation = -20.0
        viewer.cam.lookat = [0, 0, 1]

    # -- pre-compute grid offsets for height scan (world frame, to be rotated by yaw) --
    num_scan_x = cfg.robot_config.num_scan_x  # 21
    num_scan_y = cfg.robot_config.num_scan_y  # 11
    num_scan_points = num_scan_x * num_scan_y
    offset_local = np.zeros((num_scan_points, 2), dtype=np.double)
    start_x = -(num_scan_x - 1) / 2 * cfg.robot_config.scan_resolution
    start_y = -(num_scan_y - 1) / 2 * cfg.robot_config.scan_resolution
    for j in range(num_scan_y):
        for i in range(num_scan_x):
            offset_local[j * num_scan_x + i] = np.array(
                [start_x + i * cfg.robot_config.scan_resolution,
                 start_y + j * cfg.robot_config.scan_resolution]
            )

    # -- state variables --
    target_pos = np.zeros(cfg.robot_config.num_actions, dtype=np.double)
    action = np.zeros(cfg.robot_config.num_actions, dtype=np.double)
    tau = np.zeros(cfg.robot_config.num_actions, dtype=np.double)
    count_lowlevel = 0

    # -- data collection for plotting --
    time_data = []
    commanded_joint_pos_data = []
    actual_joint_pos_data = []
    tau_data = []
    commanded_lin_vel_x_data = []
    commanded_lin_vel_y_data = []
    commanded_ang_vel_z_data = []
    actual_lin_vel_data = []
    actual_ang_vel_data = []

    total_steps = int(cfg.sim_config.sim_duration / cfg.sim_config.dt)
    for step in tqdm(range(total_steps), desc="Simulating..."):

        # obtain proprioceptive observation
        q, dq, quat, v, omega, gvec = get_obs(data, cfg.robot_config.default_pos)
        q_joints = q[-cfg.robot_config.num_actions :]
        dq_joints = dq[-cfg.robot_config.num_actions :]

        # policy runs at decimated rate
        if count_lowlevel % cfg.sim_config.decimation == 0:
            # reorder joints from URDF order to USD order
            q_obs = np.zeros(cfg.robot_config.num_actions, dtype=np.double)
            dq_obs = np.zeros(cfg.robot_config.num_actions, dtype=np.double)
            q_rel = q_joints - cfg.robot_config.default_pos
            for i in range(len(cfg.robot_config.usd2urdf)):
                q_obs[i] = q_rel[cfg.robot_config.usd2urdf[i]]
                dq_obs[i] = dq_joints[cfg.robot_config.usd2urdf[i]]

            # build proprioceptive observation (78 dims) — must match Isaac Lab ordering
            obs = np.zeros(cfg.robot_config.num_single_obs, dtype=np.float32)
            obs[0:3] = omega * 0.25                # base_ang_vel, scaled
            obs[3:6] = gvec                         # projected_gravity
            obs[6] = cmd.vx                          # velocity_commands
            obs[7] = cmd.vy
            obs[8] = cmd.dyaw
            obs[9:32] = q_obs                       # joint_pos_rel
            obs[32:55] = dq_obs                     # joint_vel_rel
            obs[55:78] = action                      # last_action

            # height scan via MuJoCo ray casting
            r = R.from_quat(quat)
            yaw = r.as_euler("zyx")[0]
            cy, sy = np.cos(yaw), np.sin(yaw)
            rot_mat = np.array([[cy, -sy], [sy, cy]])
            current_offset_xy = offset_local @ rot_mat.T

            # scanner world Z: torso_pos_z + rotated scanner_offset (5.0m local Z)
            scanner_offset_local = np.array([0.0, 0.0, cfg.robot_config.scanner_z_offset])
            scanner_offset_world = r.apply(scanner_offset_local)
            scanner_z = data.qpos[2] + scanner_offset_world[2]

            dist = get_rays(model, data, data.qpos[:3].copy(), num_scan_points, current_offset_xy, scanner_z)
            height_scan_vals = np.clip(
                dist - cfg.robot_config.height_scan_offset,
                cfg.robot_config.height_scan_clip[0],
                cfg.robot_config.height_scan_clip[1],
            )

            # full policy input: [proprio (78) | height_scan (231)] = 309 dims
            policy_input = np.concatenate(
                [obs.reshape(1, -1).astype(np.float32),
                 height_scan_vals.reshape(1, -1).astype(np.float32)],
                axis=1,
            )

            # ONNX inference
            ort_inputs = {policy.get_inputs()[0].name: policy_input}
            action[:] = policy.run(None, ort_inputs)[0][0]

            # scale action to target joint positions
            target_q = action * cfg.robot_config.action_scale
            for i in range(len(cfg.robot_config.usd2urdf)):
                target_pos[cfg.robot_config.usd2urdf[i]] = target_q[i]
            target_pos = target_pos + cfg.robot_config.default_pos

            # -- collect data for plotting --
            time_data.append(step * cfg.sim_config.dt)
            commanded_joint_pos_data.append(target_pos.copy())
            actual_joint_pos_data.append(q_joints.copy())
            tau_data.append(tau.copy())
            commanded_lin_vel_x_data.append(cmd.vx)
            commanded_lin_vel_y_data.append(cmd.vy)
            commanded_ang_vel_z_data.append(cmd.dyaw)
            actual_lin_vel_data.append(v[:2].copy())
            actual_ang_vel_data.append(omega[2].copy())

            # render
            if headless:
                renderer.update_scene(data, camera=cam)
                renderer.render()
            else:
                viewer.render()

        # PD control at full simulation rate
        target_vel = np.zeros(cfg.robot_config.num_actions, dtype=np.double)
        tau = pd_control(
            target_pos, q_joints, cfg.robot_config.kps,
            target_vel, dq_joints, cfg.robot_config.kds,
        )
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
        data.ctrl = tau
        mujoco.mj_step(model, data)

        count_lowlevel += 1

    if not headless:
        viewer.close()

    # -- generate plots --
    print("\nSimulation finished. Generating plots...")

    time_arr = np.array(time_data)
    commanded_joint_pos_arr = np.array(commanded_joint_pos_data)
    actual_joint_pos_arr = np.array(actual_joint_pos_data)
    commanded_lin_vel_x_arr = np.array(commanded_lin_vel_x_data)
    commanded_lin_vel_y_arr = np.array(commanded_lin_vel_y_data)
    commanded_ang_vel_z_arr = np.array(commanded_ang_vel_z_data)
    actual_lin_vel_arr = np.array(actual_lin_vel_data)
    actual_ang_vel_arr = np.array(actual_ang_vel_data)

    try:
        import matplotlib.pyplot as plt

        fig1, axes1 = plt.subplots(6, 4, figsize=(16, 14), sharex=True)
        axes1 = axes1.flatten()
        for i in range(cfg.robot_config.num_actions):
            ax = axes1[i]
            ax.plot(time_arr, commanded_joint_pos_arr[:, i], label="Commanded", linestyle="--")
            ax.plot(time_arr, actual_joint_pos_arr[:, i], label="Actual")
            ax.set_title(f"Joint {i + 1}")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Position [rad]")
            ax.legend(fontsize="small")
            ax.grid(True)
        for i in range(cfg.robot_config.num_actions, len(axes1)):
            fig1.delaxes(axes1[i])
        fig1.suptitle("Commanded vs Actual Joint Positions", fontsize=16)
        plt.tight_layout()
        fig1.savefig("loco_transformer_joint_positions.png")
        print("  → loco_transformer_joint_positions.png")

        fig2, axes2 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        axes2[0].plot(time_arr, commanded_lin_vel_x_arr, label="Commanded Vx", linestyle="--")
        axes2[0].plot(time_arr, actual_lin_vel_arr[:, 0], label="Actual Vx")
        axes2[0].set_title("Base Linear Velocity X")
        axes2[0].set_ylabel("Velocity [m/s]")
        axes2[0].legend()
        axes2[0].grid(True)

        axes2[1].plot(time_arr, commanded_lin_vel_y_arr, label="Commanded Vy", linestyle="--")
        axes2[1].plot(time_arr, actual_lin_vel_arr[:, 1], label="Actual Vy")
        axes2[1].set_title("Base Linear Velocity Y")
        axes2[1].set_ylabel("Velocity [m/s]")
        axes2[1].legend()
        axes2[1].grid(True)

        axes2[2].plot(time_arr, commanded_ang_vel_z_arr, label="Commanded Dyaw", linestyle="--")
        axes2[2].plot(time_arr, actual_ang_vel_arr, label="Actual Dyaw")
        axes2[2].set_title("Base Angular Velocity Z")
        axes2[2].set_xlabel("Time [s]")
        axes2[2].set_ylabel("Angular Velocity [rad/s]")
        axes2[2].legend()
        axes2[2].grid(True)
        fig2.suptitle("Commanded vs Actual Base Velocities", fontsize=16)
        plt.tight_layout()
        fig2.savefig("loco_transformer_base_velocities.png")
        print("  → loco_transformer_base_velocities.png")
    except ImportError:
        print("[WARN] matplotlib not available — skipping plots.")

    print("Done.")


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------


class Sim2simCfg:
    """Configuration for the RPO-Loco-Transformer MuJoCo sim2sim deployment."""

    class sim_config:
        mujoco_model_path = os.path.join(ISAAC_DATA_DIR, "robots", "roboparty", "rpo", "mjcf", "rpo.xml")
        sim_duration = 30.0  # seconds
        dt = 0.001           # simulation timestep
        decimation = 20      # policy runs at 50 Hz (0.001 * 20 = 0.02 s)

    class robot_config:
        # PD gains (23 dof)
        kps = np.array(
            [100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40,
             150, 40, 40, 40, 30, 20, 40, 40, 40, 30, 20],
            dtype=np.double,
        )
        kds = np.array(
            [3.3, 3.3, 3.3, 5.0, 2.0, 2.0, 3.3, 3.3, 3.3, 5.0, 2.0, 2.0,
             5.0, 2.0, 2.0, 2.0, 1.5, 1.0, 2.0, 2.0, 2.0, 1.5, 1.0],
            dtype=np.double,
        )
        default_pos = np.array(
            [0, 0, -0.1, 0.3, -0.2, 0, 0, 0, -0.1, 0.3, -0.2, 0,
             0, 0.18, 0.06, 0, 0.78, 0, 0.18, -0.06, 0, 0.78, 0],
            dtype=np.double,
        )
        tau_limit = 200.0 * np.ones(23, dtype=np.double)

        # observation dimensions
        num_single_obs = 78     # proprioceptive only
        num_actions = 23
        action_scale = 0.25     # matches JointPositionActionCfg

        # joint name mapping: URDF index → USD index
        # URDF order:
        #   left_thigh_yaw(0), right_thigh_yaw(1), torso(2),
        #   left_thigh_roll(3), right_thigh_roll(4), left_arm_pitch(5), right_arm_pitch(6),
        #   left_thigh_pitch(7), right_thigh_pitch(8), left_arm_roll(9), right_arm_roll(10),
        #   left_knee(11), right_knee(12), left_arm_yaw(13), right_arm_yaw(14),
        #   left_ankle_pitch(15), right_ankle_pitch(16), left_elbow_pitch(17), right_elbow_pitch(18),
        #   left_ankle_roll(19), right_ankle_roll(20), left_elbow_yaw(21), right_elbow_yaw(22)
        usd2urdf = [
            0, 6, 12, 1, 7, 13, 18, 2, 8, 14, 19, 3,
            9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22,
        ]

        # height scan parameters — must match Isaac Lab env config
        num_scan_x = 21          # 2.0 / 0.1 + 1
        num_scan_y = 11          # 1.0 / 0.1 + 1
        scan_resolution = 0.1    # GridPatternCfg.resolution
        scanner_z_offset = 5.0   # RayCasterCfg offset.z — scanner mounted 5m above torso
        height_scan_offset = 0.5 # observation offset parameter
        height_scan_clip = (-5.0, 5.0)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RPO-Loco-Transformer MuJoCo sim2sim deployment.")
    parser.add_argument("--load_model", type=str, required=True, help="Path to the ONNX policy file (policy.onnx).")
    parser.add_argument("--terrain", action="store_true", help="Use terrain MJCF instead of plane.")
    parser.add_argument("--headless", action="store_true", help="Run without GUI and save video.")
    parser.add_argument("--no-keyboard", action="store_true", help="Disable keyboard listener.")
    args = parser.parse_args()

    # override model path for terrain
    if args.terrain:
        Sim2simCfg.sim_config.mujoco_model_path = os.path.join(
            ISAAC_DATA_DIR, "robots", "roboparty", "rpo", "mjcf", "rpo_terrain.xml"
        )

    # start keyboard listener (background thread)
    listener = None
    if not args.no_keyboard:
        print("\n[Keyboard] Velocity control (WASD + QE + X):")
        print("  W/S: Forward / Backward")
        print("  A/D: Strafe left / right")
        print("  Q/E: Turn left / right")
        print("  X:   Stop (zero velocity)")
        print()
        listener = _start_keyboard_listener()
        time.sleep(0.3)  # let the listener thread settle

    try:
        # load ONNX policy
        print(f"[INFO] Loading ONNX policy from: {args.load_model}")
        policy = ort.InferenceSession(args.load_model)
        run_mujoco(policy, Sim2simCfg(), args.headless)
    finally:
        if listener is not None:
            listener.stop()
