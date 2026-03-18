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
    default=[0.45, 0.0, 0.20, 0.0, 1.57, 0.0],
    metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
    help="Initial arm end-effector pose in base_link frame (meters/radians).",
)
parser.add_argument(
    "--arm_terminal_control",
    action="store_true",
    default=False,
    help="Enable terminal real-time control for the arm Pink IK target.",
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

import threading
from dataclasses import dataclass

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
@dataclass
class ArmTargetState:
    xyz: np.ndarray
    quat_wxyz: np.ndarray
    updated: bool = False

def _rpy_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    quat_xyzw = R.from_euler("xyz", [roll, pitch, yaw]).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)


def _quat_wxyz_to_rpy(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_xyzw = np.array(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
        dtype=np.float64,
    )
    return R.from_quat(quat_xyzw).as_euler("xyz")


def _print_arm_terminal_help():
    print("\n[ARM TERMINAL CONTROL]")
    print("Input format:")
    print("  x y z roll pitch yaw")
    print("Example:")
    print("  0.45 0.00 0.20 0.0 1.57 0.0")
    print("Special commands:")
    print("  p            -> print current target")
    print("  h            -> help")
    print("  q            -> stop terminal input thread (simulation keeps running)")
    print("")

def _arm_terminal_input_loop(target_state: ArmTargetState, target_lock: threading.Lock, stop_event: threading.Event):
    _print_arm_terminal_help()
    while not stop_event.is_set():
        try:
            line = input("[arm target xyz rpy] > ").strip()
        except EOFError:
            print("[INFO] Arm terminal input closed (EOF).")
            break
        except KeyboardInterrupt:
            print("[INFO] Arm terminal input interrupted.")
            break

        if not line:
            continue

        if line.lower() in {"h", "help"}:
            _print_arm_terminal_help()
            continue

        if line.lower() in {"q", "quit", "exit"}:
            print("[INFO] Arm terminal input thread exiting.")
            break

        if line.lower() in {"p", "print"}:
            with target_lock:
                xyz = target_state.xyz.copy()
                quat = target_state.quat_wxyz.copy()
            rpy = _quat_wxyz_to_rpy(quat)
            print(
                f"[INFO] Current arm target: "
                f"x={xyz[0]:.4f}, y={xyz[1]:.4f}, z={xyz[2]:.4f}, "
                f"roll={rpy[0]:.4f}, pitch={rpy[1]:.4f}, yaw={rpy[2]:.4f}"
            )
            continue

        parts = line.replace(",", " ").split()
        if len(parts) != 6:
            print("[WARN] Expected 6 values: x y z roll pitch yaw")
            continue

        try:
            vals = [float(x) for x in parts]
        except ValueError:
            print("[WARN] Invalid number format.")
            continue

        xyz = np.array(vals[:3], dtype=np.float64)
        quat_wxyz = _rpy_to_quat_wxyz(*vals[3:])

        with target_lock:
            target_state.xyz = xyz
            target_state.quat_wxyz = quat_wxyz
            target_state.updated = True

        print(
            f"[INFO] Updated arm target -> "
            f"x={xyz[0]:.4f}, y={xyz[1]:.4f}, z={xyz[2]:.4f}, "
            f"roll={vals[3]:.4f}, pitch={vals[4]:.4f}, yaw={vals[5]:.4f}"
        )

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

    env_cfg.scene.num_envs = 1
    env_cfg.terminations.time_out = None
    env_cfg.commands.base_velocity.debug_vis = False
    env_cfg.observations.policy.velocity_commands = ObsTerm(
        func=lambda env: torch.zeros((env.num_envs, 3), device=env.device),
    )    
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
    target_state = None
    target_lock = None
    target_stop_event = None
    target_input_thread = None

    # if args_cli.z1_urdf is None:
    #     raise ValueError("--z1_urdf must be provided for Pink IK control.")
    # if args_cli.z1_mesh_dir is None:
    #     raise ValueError("--z1_mesh_dir must be provided for Pink IK control.")

    sim_env = env.unwrapped
    robot = sim_env.scene["robot"]

    arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    arm_joint_ids, arm_joint_names = robot.find_joints(arm_joint_names, preserve_order=True)

    ee_task = FrameTask(
        frame="link06",
        position_cost=1.0,
        orientation_cost=0.0,
    )
    posture_task = NullSpacePostureTask(cost=1e-3)

    pink_cfg = PinkIKControllerCfg(
        urdf_path="/home/leakycauldron/robot_lab-2.3.2-lx/source/robot_lab/data/Robots/unitree/b2_description/urdf/b2_description_b2z1.urdf",
        mesh_path="/home/leakycauldron/robot_lab-2.3.2-lx/source/robot_lab/data/Robots/unitree",
        joint_names=arm_joint_names,
        all_joint_names=robot.data.joint_names,
        articulation_name="robot",
        base_link_name="base_link",
        variable_input_tasks=[ee_task, posture_task],
        fixed_input_tasks=[],
        show_ik_warnings=True,
        fail_on_joint_limit_violation=False,
    )

    arm_pink_controller = PinkIKController(
        cfg=pink_cfg,
        robot_cfg=robot.cfg,
        device=str(sim_env.device),
        controlled_joint_indices=arm_joint_ids.tolist() if hasattr(arm_joint_ids, "tolist") else list(arm_joint_ids),
    )

    init_xyz = np.array(args_cli.arm_ee_pose[:3], dtype=np.float64)
    init_quat_wxyz = _rpy_to_quat_wxyz(*args_cli.arm_ee_pose[3:])
    target_state = ArmTargetState(
        xyz=init_xyz,
        quat_wxyz=init_quat_wxyz,
        updated=True,
    )
    target_lock = threading.Lock()

    print(f"[INFO] Pink arm-only control enabled for joints: {arm_joint_names}")
    print(f"[INFO] End-effector frame: link06")
    print(f"[INFO] Base link frame: base_link")
    print(f"[INFO] Initial target pose xyzrpy: {args_cli.arm_ee_pose}")

    if args_cli.arm_terminal_control:
        target_stop_event = threading.Event()
        target_input_thread = threading.Thread(
            target=_arm_terminal_input_loop,
            args=(target_state, target_lock, target_stop_event),
            daemon=True,
        )
        target_input_thread.start()

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

                # 当前整机 joint positions（Pink 内部会只取 controlled_joint_indices 对应的 arm joints）
                curr_joint_pos = robot.data.joint_pos[0].detach().cpu().numpy()
                print("-"*50)
                print("curr_joint_pos: ", curr_joint_pos)

                with target_lock:
                    target_xyz = target_state.xyz.copy()
                    target_quat_wxyz = target_state.quat_wxyz.copy()
                    target_state.updated = False

                # # 更新末端任务目标
                # ee_task.set_target(
                #     translation=target_xyz,
                #     rotation=target_quat_wxyz,
                # )
                # target_rot = R.from_quat(
                #     [target_quat_wxyz[1], target_quat_wxyz[2], target_quat_wxyz[3], target_quat_wxyz[0]]
                # ).as_matrix()

                # target_tf = pin.SE3(target_rot, target_xyz)
                # ee_task.set_target(target_tf)
                base_link_id = robot.find_bodies(["base_link"], preserve_order=True)[0][0]
                ee_link_id = robot.find_bodies(["link06"], preserve_order=True)[0][0]
                print("base_link_id:", base_link_id)
                print("ee_link_id:", ee_link_id)
                base_pos_w = robot.data.body_pos_w[0, base_link_id].detach().cpu().numpy()
                base_quat_w = robot.data.body_quat_w[0, base_link_id].detach().cpu().numpy()
                print("base_pos_w: ", base_pos_w)
                print("base_quat_w: ", base_quat_w)

                base_rot_w = R.from_quat([
                    base_quat_w[1], base_quat_w[2], base_quat_w[3], base_quat_w[0]
                ]).as_matrix()

                target_rot_b = R.from_quat([
                    target_quat_wxyz[1], target_quat_wxyz[2], target_quat_wxyz[3], target_quat_wxyz[0]
                ]).as_matrix()

                target_pos_w = base_rot_w @ target_xyz + base_pos_w
                target_rot_w = base_rot_w @ target_rot_b

                target_tf_w = pin.SE3(target_rot_w, target_pos_w)
                ee_task.set_target(target_tf_w)

                # 更新 null-space posture target，防止冗余姿态乱漂
                # curr_arm_joint_pos = curr_joint_pos[arm_joint_ids]
                # arm_pink_controller.update_null_space_joint_targets(curr_arm_joint_pos)

                # Pink 计算 arm joint targets
                arm_joint_target = arm_pink_controller.compute(
                    curr_joint_pos=curr_joint_pos,
                    dt=dt,
                )

                arm_joint_target = arm_joint_target.unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
                print("target_pos_w:", target_pos_w)
                print("target_rot_w:", target_rot_w)

                robot = env.unwrapped.scene["robot"]
                ee_link_id = robot.find_bodies(["link06"], preserve_order=True)[0][0]
                ee_pos = robot.data.body_pos_w[0, ee_link_id]
                ee_quat = robot.data.body_quat_w[0, ee_link_id]
                print("curr_ee_xyz:", ee_pos.cpu().numpy())
                print("ee_quat:", ee_quat)

                print("arm_joint_target: ", arm_joint_target)
                robot.set_joint_position_target(arm_joint_target, joint_ids=arm_joint_ids)

                ee_pos_w = robot.data.body_pos_w[0, ee_link_id].detach().cpu().numpy()
                ee_quat_w = robot.data.body_quat_w[0, ee_link_id].detach().cpu().numpy()

                ee_pos_b = base_rot_w.T @ (ee_pos_w - base_pos_w)

                ee_rot_w = R.from_quat([
                    ee_quat_w[1], ee_quat_w[2], ee_quat_w[3], ee_quat_w[0]
                ]).as_matrix()
                ee_rot_b = base_rot_w.T @ ee_rot_w
                ee_rpy_b = R.from_matrix(ee_rot_b).as_euler("xyz")

                print("target_xyz_base:", target_xyz)
                print("curr_ee_xyz_base:", ee_pos_b)
                print("pos_err_base:", target_xyz - ee_pos_b)
                print("curr_ee_rpy_base:", ee_rpy_b)


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

    if target_stop_event is not None:
        target_stop_event.set()
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
