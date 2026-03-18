# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

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
    default=None,
    metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
    help="Fixed arm end-effector pose in base_link frame (meters/radians).",
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

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from scipy.spatial.transform import Rotation as R

from isaaclab.controllers.pink_ik import PinkIKController, PinkIKControllerCfg, NullSpacePostureTask
from pink.tasks import FrameTask
import pinocchio as pin

# add local package source path when the project is not installed as a site package
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_python_source_dir = os.path.join(_repo_root, "source", "robot_lab")
if _python_source_dir not in sys.path and os.path.isdir(_python_source_dir):
    sys.path.insert(0, _python_source_dir)
    
import robot_lab.tasks  # noqa: F401  # isort: skip

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl_utils import camera_follow

# PLACEHOLDER: Extension template (do not remove this comment)

def _rpy_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _rotvec_from_rot(rot: np.ndarray) -> np.ndarray:
    cos_theta = np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float64)
    skew = (rot - rot.T) / (2.0 * np.sin(theta))
    axis = np.array([skew[2, 1], skew[0, 2], skew[1, 0]], dtype=np.float64)
    return axis * theta


def _homogeneous(rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot
    mat[:3, 3] = trans
    return mat


def _axis_angle_rot(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    v = 1.0 - c
    return np.array(
        [
            [x * x * v + c, x * y * v - z * s, x * z * v + y * s],
            [y * x * v + z * s, y * y * v + c, y * z * v - x * s],
            [z * x * v - y * s, z * y * v + x * s, z * z * v + c],
        ],
        dtype=np.float64,
    )


def _z1_fk_in_base(q: np.ndarray) -> np.ndarray:
    mount_t = np.array([0.24, 0.0, 0.11], dtype=np.float64)
    chain = [
        (np.array([0.0, 0.0, 0.0585]), np.array([0.0, 0.0, 1.0]), q[0]),
        (np.array([0.0, 0.0, 0.0450]), np.array([0.0, 1.0, 0.0]), q[1]),
        (np.array([-0.35, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), q[2]),
        (np.array([0.218, 0.0, 0.057]), np.array([0.0, 1.0, 0.0]), q[3]),
        (np.array([0.07, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), q[4]),
        (np.array([0.0492, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), q[5]),
    ]
    tf = _homogeneous(np.eye(3), mount_t)
    for trans, axis, angle in chain:
        tf = tf @ _homogeneous(np.eye(3), trans) @ _homogeneous(_axis_angle_rot(axis, angle), np.zeros(3))
    tf = tf @ _homogeneous(np.eye(3), np.array([0.051, 0.0, 0.0]))
    return tf


def _solve_z1_ik(target_pose_xyzrpy: list[float]) -> np.ndarray:
    target_pos = np.array(target_pose_xyzrpy[:3], dtype=np.float64)
    target_rot = _rpy_to_rot(*target_pose_xyzrpy[3:])
    q = np.array([0.01, 1.48, -0.63, -0.84, 0.0, 1.57], dtype=np.float64)
    q_min = np.array([-2.61799387799, 0.0, -2.87979326579, -1.51843644923, -1.34390352403, -2.79252680319])
    q_max = np.array([2.61799387799, 2.96705972839, 0.0, 1.51843644923, 1.34390352403, 2.79252680319])
    damping = 1e-3
    eps = 1e-5
    for _ in range(120):
        tf = _z1_fk_in_base(q)
        pos_err = target_pos - tf[:3, 3]
        rot_err = _rotvec_from_rot(target_rot @ tf[:3, :3].T)
        err = np.concatenate([pos_err, rot_err])
        if np.linalg.norm(pos_err) < 1e-3 and np.linalg.norm(rot_err) < 5e-3:
            return q
        jac = np.zeros((6, 6), dtype=np.float64)
        for i in range(6):
            q_perturb = q.copy()
            q_perturb[i] += eps
            tf_p = _z1_fk_in_base(q_perturb)
            pos_diff = (tf_p[:3, 3] - tf[:3, 3]) / eps
            rot_diff = _rotvec_from_rot(tf_p[:3, :3] @ tf[:3, :3].T) / eps
            jac[:, i] = np.concatenate([pos_diff, rot_diff])
        jjt = jac @ jac.T
        dq = jac.T @ np.linalg.solve(jjt + damping * np.eye(6), err)
        q = np.clip(q + dq, q_min, q_max)
    raise RuntimeError(f"Arm IK did not converge for target pose: {target_pose_xyzrpy}")

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

    arm_pink_controller = None
    arm_joint_ids = None
    ee_task = None
    arm_target_pose = None

    if args_cli.arm_ee_pose is not None:
        sim_env = env.unwrapped
        robot = sim_env.scene["robot"]

        arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        arm_joint_ids, arm_joint_names = robot.find_joints(arm_joint_names, preserve_order=True)

        # 末端任务：控制 Z1 的工具末端 frame
        # 这里 frame 名必须和 URDF 里的 link/frame 名一致
        ee_task = FrameTask(
            frame="link06",          # <- 这里要改成你 Z1 URDF 里真实的末端 frame 名
            position_cost=1.0,
            orientation_cost=0.5,
        )

        posture_task = NullSpacePostureTask(cost=1e-3)

        pink_cfg = PinkIKControllerCfg(
            urdf_path="/home/leakycauldron/robot_lab-2.3.2-lx/source/robot_lab/data/Robots/unitree/b2_description/urdf/b2_description_b2z1.urdf",          # 必填
            mesh_path="/home/leakycauldron/robot_lab-2.3.2-lx/source/robot_lab/data/Robots/unitree",           # 没有 mesh 可先试 None，但通常建议给
            joint_names=arm_joint_names,               # 被 Pink 控的 joints
            all_joint_names=robot.data.joint_names,    # USD asset 全部 joint 名
            articulation_name="robot",
            base_link_name="base_link",
            variable_input_tasks=[ee_task, posture_task],
            fixed_input_tasks=[],
            show_ik_warnings=True,
            fail_on_joint_limit_violation=False,
        )

        # arm_pink_controller = PinkIKController(
        #     cfg=pink_cfg,
        #     robot_cfg=robot.cfg,
        #     device=str(sim_env.device),
        #     controlled_joint_indices=arm_joint_ids.tolist() if hasattr(arm_joint_ids, "tolist") else list(arm_joint_ids),
        # )

        arm_pink_controller = PinkIKController(
            cfg=pink_cfg,
            robot_cfg=robot.cfg,
            device=str(sim_env.device),
            controlled_joint_indices=arm_joint_ids,
        )

        # 保存目标 pose: xyz + quat(wxyz)
        xyz = np.array(args_cli.arm_ee_pose[:3], dtype=np.float64)
        quat_xyzw = R.from_euler("xyz", args_cli.arm_ee_pose[3:]).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)

        arm_target_pose = (xyz, quat_wxyz)

        print(f"[INFO] Pink IK target EE pose (base_link): {args_cli.arm_ee_pose}")

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
    # reset 后
    # while simulation_app.is_running():
    #     with torch.inference_mode():
    #         actions = policy(obs)
    #         curr_joint_pos = robot.data.joint_pos[0].detach().cpu().numpy()

    #         ee_task.set_target(
    #             translation=np.array([0.45, 0.0, 0.25]),
    #             rotation=np.array([1.0, 0.0, 0.0, 0.0]),  # wxyz
    #         )

    #         q_des = arm_pink_controller.compute(curr_joint_pos, dt)
    #         robot.set_joint_position_target(q_des.unsqueeze(0), joint_ids=arm_joint_ids)

    #         env.step(torch.zeros_like(actions))    
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)

            if arm_pink_controller is not None:
                robot = env.unwrapped.scene["robot"]

                # 取当前全部 joint 位置（单环境）
                curr_joint_pos = robot.data.joint_pos[0].detach().cpu().numpy()

                # 给 Pink 的末端任务设目标
                target_xyz, target_quat_wxyz = arm_target_pose
                # ee_task.set_target(
                #     translation=target_xyz,
                #     rotation=target_quat_wxyz,
                # )
                target_rot = R.from_quat(
                    [target_quat_wxyz[1], target_quat_wxyz[2], target_quat_wxyz[3], target_quat_wxyz[0]]
                ).as_matrix()

                target_tf = pin.SE3(target_rot, target_xyz)
                ee_task.set_target(target_tf)

                # 可选：把 null-space 姿态目标设成当前 arm 姿态，更稳
                curr_arm_joint_pos = curr_joint_pos[arm_joint_ids]
                arm_pink_controller.update_null_space_joint_targets(curr_arm_joint_pos)

                # 算出 arm joints 的目标关节角
                arm_joint_target = arm_pink_controller.compute(curr_joint_pos=curr_joint_pos, dt=dt)

                # shape -> [num_envs, 6]
                arm_joint_target = arm_joint_target.unsqueeze(0).repeat(env.unwrapped.num_envs, 1)

                robot.set_joint_position_target(arm_joint_target, joint_ids=arm_joint_ids)

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
    try:
        # run the main function
        main()
    except SyntaxError as err:
        print("[ERROR] 检测到 Python 语法错误，通常是任务配置文件里混入了非代码文本。")
        print(f"[ERROR] {err.filename}:{err.lineno}: {err.msg}")
        raise
    finally:
        # close sim app
        simulation_app.close()
