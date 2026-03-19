# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import sys
import threading

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
parser.add_argument(
    "--arm_ee_pose",
    type=float,
    nargs=6,
    default=[0.30, 0.0, 0.25, 0.0, 1.57, 0.0],
    metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
    help="Initial arm end-effector pose in the robot base frame (meters/radians).",
)
parser.add_argument(
    "--arm_terminal_control",
    action="store_true",
    default=False,
    help="Enable realtime terminal control of the arm end-effector pose.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# add local package source path when the project is not installed as a site package
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_python_source_dir = os.path.join(_repo_root, "source", "robot_lab")
if _python_source_dir not in sys.path and os.path.isdir(_python_source_dir):
    sys.path.insert(0, _python_source_dir)
    
import robot_lab.tasks  # noqa: F401  # isort: skip

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl_utils import camera_follow

# PLACEHOLDER: Extension template (do not remove this comment)

# EE_FRAME_NAME = "gripperMover"
EE_FRAME_NAME = "gripper_link"
ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


def euler_xyz_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert XYZ Euler angles to quaternion in wxyz order."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float32)


class SharedArmTarget:
    def __init__(self, pose_xyzrpy: list[float]):
        self.pos = np.array(pose_xyzrpy[:3], dtype=np.float32)
        self.quat = euler_xyz_to_quat_wxyz(*pose_xyzrpy[3:])
        self.lock = threading.Lock()

    def set_target(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float):
        with self.lock:
            self.pos[:] = [x, y, z]
            self.quat[:] = euler_xyz_to_quat_wxyz(roll, pitch, yaw)

    def get_target(self):
        with self.lock:
            return self.pos.copy(), self.quat.copy()


def arm_terminal_input_loop(shared_target: SharedArmTarget):
    print("\n[ARM CONTROL] Type target pose in robot base frame as:")
    print("  x y z roll pitch yaw")
    print("Example:")
    print("  0.30 0.00 0.25 0.0 1.57 0.0\n")

    while True:
        try:
            text = input("arm target> ").strip()
            if not text:
                continue
            vals = [float(v) for v in text.replace(",", " ").split()]
            if len(vals) != 6:
                print("[WARN] Need 6 numbers: x y z roll pitch yaw")
                continue
            shared_target.set_target(*vals)
            print("[INFO] Arm target updated.")
        except EOFError:
            print("[INFO] Arm terminal input closed.")
            break
        except KeyboardInterrupt:
            print("[INFO] Arm terminal input interrupted.")
            break
        except Exception as exc:
            print(f"[WARN] Invalid arm target: {exc}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 64

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # spawn the robot randomly in the grid (instead of their terrain levels)
    # env_cfg.scene.terrain.max_init_terrain_level = None
    
    # spawn robot on the easiest terrain level to avoid starting over pits/gaps during play
    # (helps deterministic evaluation with feet touching ground after reset)
    env_cfg.scene.terrain.max_init_terrain_level = 0
    # reduce the number of terrains to save memory
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # disable randomization for play
    env_cfg.observations.policy.enable_corruption = False
    # remove random pushing
    env_cfg.events.randomize_apply_external_force_torque = None
    # env_cfg.events.push_robot = None

    # keep reset at init_state (stand-still) and disable dynamics/domain randomization during evaluation
    env_cfg.events.randomize_push_robot = None
    env_cfg.events.randomize_reset_base = None
    env_cfg.events.randomize_actuator_gains = None
    env_cfg.events.randomize_rigid_body_material = None
    env_cfg.events.randomize_rigid_body_mass_base = None
    env_cfg.events.randomize_rigid_body_mass_others = None
    env_cfg.events.randomize_com_positions = None

    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.command_levels_ang_vel = None

    if args_cli.keyboard or args_cli.arm_terminal_control:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None

    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        config = Se2KeyboardCfg(
            v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1],
            v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
            omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
        )
        controller = Se2Keyboard(config)
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: torch.tensor(controller.advance(), dtype=torch.float32).unsqueeze(0).to(env.device),
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)        
# @@ -174,97 +259,148 @@ def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agen
    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
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
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    arm_joint_ids = None
    ee_body_id = None
    ee_jacobi_idx = None
    diff_ik = None
    arm_target = None

    if args_cli.arm_terminal_control:
        robot = env.unwrapped.scene["robot"]
        arm_joint_ids, arm_joint_names = robot.find_joints(ARM_JOINT_NAMES, preserve_order=True)
        ee_body_ids, ee_body_names = robot.find_bodies(EE_FRAME_NAME, preserve_order=True)
        ee_body_id = ee_body_ids[0]
        ee_jacobi_idx = ee_body_id - 1 if robot.is_fixed_base else ee_body_id

        diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        diff_ik = DifferentialIKController(diff_ik_cfg, num_envs=env.unwrapped.num_envs, device=env.unwrapped.device)
        arm_target = SharedArmTarget(args_cli.arm_ee_pose)
        init_pos_b, init_quat_b = arm_target.get_target()
        ik_command = torch.tensor(
            np.concatenate([init_pos_b, init_quat_b])[None, :], dtype=torch.float32, device=env.unwrapped.device
        )
        diff_ik.reset()
        diff_ik.set_command(ik_command)

        arm_input_thread = threading.Thread(target=arm_terminal_input_loop, args=(arm_target,), daemon=True)
        arm_input_thread.start()

        print(f"[INFO] Arm differential IK enabled for joints: {arm_joint_names}")
        print(f"[INFO] Arm EE body: {ee_body_names[0]} (body_id={ee_body_id}, jacobian_index={ee_jacobi_idx})")
        print(f"[INFO] Initial arm target xyzrpy: {args_cli.arm_ee_pose}")

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)

            if diff_ik is not None:
                robot = env.unwrapped.scene["robot"]
                ee_pose_w = robot.data.body_pose_w[:, ee_body_id]
                root_pose_w = robot.data.root_pose_w
                # print("root_pose_w: ", root_pose_w)
                # print("ee_pose_w: ", ee_pose_w)
                ee_pos_b, ee_quat_b = subtract_frame_transforms(
                    root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
                )
                # print("ee_pos_b: ", ee_pos_b)
                # print("ee_quat_b: ", ee_quat_b)
                # jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids]
                # joint_pos_arm = robot.data.joint_pos[:, arm_joint_ids]
                jacobians = robot.root_physx_view.get_jacobians()   # (N, bodies, 6, cols)

                if robot.is_fixed_base:
                    jacobian = jacobians[:, ee_jacobi_idx, :, arm_joint_ids]
                else:
                    arm_joint_ids_jac = [jid + 6 for jid in arm_joint_ids]
                    jacobian = jacobians[:, ee_jacobi_idx, :, arm_joint_ids_jac]

                joint_pos_arm = robot.data.joint_pos[:, arm_joint_ids]                


                target_pos_b, target_quat_b = arm_target.get_target()
                ik_command = torch.tensor(
                    np.concatenate([target_pos_b, target_quat_b])[None, :],
                    dtype=torch.float32,
                    device=env.unwrapped.device,
                )
                # print("target_pos_b: ", target_pos_b, "target_quat_b: ", target_quat_b)
                diff_ik.set_command(ik_command) 
                joint_pos_des_arm = diff_ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos_arm)
                # joint_pos_des_arm = torch.zeros_like(joint_pos_arm)    
                # joint_pos_des_arm[:, 2] = -0.8    
                robot.set_joint_position_target(joint_pos_des_arm, joint_ids=arm_joint_ids)

            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        if args_cli.keyboard:
            camera_follow(env)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()