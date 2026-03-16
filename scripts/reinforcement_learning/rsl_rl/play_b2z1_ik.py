# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Play B2 checkpoint while controlling Z1 arm with realtime IK targets.

Examples:
python scripts/reinforcement_learning/rsl_rl/play_b2z1_ik.py \
  --task RobotLab-Isaac-Velocity-Flat-Unitree-B2-v0 \
  --checkpoint /abs/path/to/model_4999.pt --num_envs 1

UDP target input (default 0.0.0.0:5555):
- Position only: x y z
- Pose: x y z qw qx qy qz
"""

import argparse
# import socket
import sys
import time

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play B2 checkpoint with additional Z1 IK controller.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=2000, help="Recorded video length in steps.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--task", type=str, required=True, help="Task name.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent config entry.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
# parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard for base control.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in realtime if possible.")
# parser.add_argument("--udp_host", type=str, default="0.0.0.0", help="UDP host for EE target input.")
# parser.add_argument("--udp_port", type=int, default=5555, help="UDP port for EE target input.")
# parser.add_argument("--ee_body", type=str, default="link06", help="End-effector body name.")
# parser.add_argument("--ik_pos_gain", type=float, default=0.25, help="Position gain for IK task error.")
# parser.add_argument("--ik_rot_gain", type=float, default=0.15, help="Rotation gain for IK task error.")
# parser.add_argument("--ik_damping", type=float, default=0.05, help="DLS damping lambda.")
parser.add_argument(
    "--arm_fixed_joint_pos",
    nargs=6,
    type=float,
    default=[1.57, 1.48, -0.63, -0.84, 0.0, 1.57],
    help="Fixed arm joint positions [joint1..joint6]. Default points arm to the left.",
)
parser.add_argument(
    "--arm_joint_names",
    nargs="+",
    default=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
    help="Arm joint names used for IK.",
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import gymnasium as gym
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
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# add local package source path when the project is not installed as a site package
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_python_source_dir = os.path.join(_repo_root, "source", "robot_lab")
if _python_source_dir not in sys.path and os.path.isdir(_python_source_dir):
    sys.path.insert(0, _python_source_dir)
    
import robot_lab.tasks  # noqa: F401  # isort: skip
from robot_lab.assets.unitree import UNITREE_B2Z1_CFG
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl_utils import camera_follow


def _normalize_quat(q: torch.Tensor) -> torch.Tensor:
    return q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-8)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.stack((q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]), dim=-1)


def _orientation_error(target_q: torch.Tensor, current_q: torch.Tensor) -> torch.Tensor:
    target_q = _normalize_quat(target_q)
    current_q = _normalize_quat(current_q)
    q_err = _quat_mul(target_q, _quat_conjugate(current_q))
    sign = torch.where(q_err[:, :1] < 0.0, -1.0, 1.0)
    q_err = q_err * sign
    return 2.0 * q_err[:, 1:4]


class UdpTargetServer:
    """Non-blocking UDP server for EE target updates."""

    def __init__(self, host: str, port: int):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((host, port))
        self._sock.setblocking(False)
        print(f"[INFO] IK target UDP server listening on {host}:{port}")

    def poll(self):
        try:
            data, _ = self._sock.recvfrom(512)
        except BlockingIOError:
            return None
        msg = data.decode("utf-8").strip().split()
        if len(msg) not in (3, 7):
            print(f"[WARN] Ignore UDP target '{data.decode('utf-8').strip()}'. Expect 'x y z' or 'x y z qw qx qy qz'")
            return None
        vals = [float(v) for v in msg]
        if len(vals) == 3:
            return vals, None
        return vals[:3], vals[3:7]


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Use B2+Z1 articulation while keeping 12-dim leg action interface for B2 checkpoints.
    env_cfg.scene.robot = UNITREE_B2Z1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    if hasattr(env_cfg.actions, "joint_pos"):
        env_cfg.actions.joint_pos.joint_names = [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ]

    # Deterministic play setup.
    env_cfg.scene.terrain.max_init_terrain_level = 0
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 3
        env_cfg.scene.terrain.terrain_generator.num_cols = 3
        env_cfg.scene.terrain.terrain_generator.curriculum = False
    env_cfg.observations.policy.enable_corruption = False
    for term_name in [
        "randomize_apply_external_force_torque",
        "randomize_push_robot",
        "randomize_reset_base",
        "randomize_reset_joints",
        "randomize_actuator_gains",
        "randomize_rigid_body_material",
        "randomize_rigid_body_mass_base",
        "randomize_rigid_body_mass_others",
        "randomize_com_positions",
    ]:
        if hasattr(env_cfg.events, term_name):
            setattr(env_cfg.events, term_name, None)
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

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    sim_env = env.unwrapped
    robot = sim_env.scene["robot"]
    arm_joint_ids, arm_joint_names = robot.find_joints(args_cli.arm_joint_names)
    # ee_body_ids, ee_body_names = robot.find_bodies(args_cli.ee_body)
    # ee_body_id = int(ee_body_ids[0])
    print(f"[INFO] Arm joints: {arm_joint_names} -> {arm_joint_ids}")
    # print(f"[INFO] EE body: {ee_body_names[0]} -> {ee_body_id}")

    # udp = UdpTargetServer(args_cli.udp_host, args_cli.udp_port)
    arm_fixed_joint_pos = torch.tensor(args_cli.arm_fixed_joint_pos, dtype=torch.float32, device=env.unwrapped.device)
    arm_fixed_joint_pos = arm_fixed_joint_pos.unsqueeze(0).repeat(sim_env.num_envs, 1)

    obs = env.get_observations()
    # target_pos_w = robot.data.body_pos_w[:, ee_body_id].clone()
    # target_pos_w[:, 2] += 0.05
    # target_quat_w = robot.data.body_quat_w[:, ee_body_id].clone()

    while simulation_app.is_running():
        start_t = time.time()
        # cmd = udp.poll()
        # if cmd is not None:
        #     cmd_pos, cmd_quat = cmd
        #     target_pos_w[:, 0] = cmd_pos[0]
        #     target_pos_w[:, 1] = cmd_pos[1]
        #     target_pos_w[:, 2] = cmd_pos[2]
        #     if cmd_quat is not None:
        #         quat_tensor = torch.tensor(cmd_quat, dtype=target_quat_w.dtype, device=target_quat_w.device)
        #         target_quat_w[:] = _normalize_quat(quat_tensor.unsqueeze(0)).repeat(target_quat_w.shape[0], 1)

        with torch.inference_mode():
            leg_actions = policy(obs)

            # ee_pos = robot.data.body_pos_w[:, ee_body_id]
            # ee_quat = robot.data.body_quat_w[:, ee_body_id]
            # pos_err = target_pos_w - ee_pos
            # rot_err = _orientation_error(target_quat_w, ee_quat)
            # task_err = torch.cat((args_cli.ik_pos_gain * pos_err, args_cli.ik_rot_gain * rot_err), dim=-1)

            # jacobians = robot.root_physx_view.get_jacobians()  # [N, num_bodies, 6, num_joints]
            # J = jacobians[:, ee_body_id, :6, :][:, :, arm_joint_ids]  # [N, 6, na]

            # lambda_sq = args_cli.ik_damping * args_cli.ik_damping
            # eye6 = torch.eye(6, device=J.device, dtype=J.dtype).unsqueeze(0).repeat(J.shape[0], 1, 1)
            # JJt = torch.bmm(J, J.transpose(1, 2))
            # inv = torch.linalg.inv(JJt + lambda_sq * eye6)
            # dq = torch.bmm(torch.bmm(J.transpose(1, 2), inv), task_err.unsqueeze(-1)).squeeze(-1)

            # arm_q = robot.data.joint_pos[:, arm_joint_ids]
            # arm_q_target = arm_q + dq
            # robot.set_joint_position_target(arm_q_target, joint_ids=arm_joint_ids)
            
            # Keep arm at fixed pose (minimal stable debug mode).
            robot.set_joint_position_target(arm_fixed_joint_pos, joint_ids=arm_joint_ids)

            obs, _, dones, _ = env.step(leg_actions)

            try:
                policy_nn = runner.alg.policy
            except AttributeError:
                policy_nn = runner.alg.actor_critic
            policy_nn.reset(dones)

        if args_cli.real_time:
            dt = sim_env.step_dt
            sleep_t = dt - (time.time() - start_t)
            if sleep_t > 0:
                time.sleep(sleep_t)
        
        if args_cli.keyboard:
            camera_follow(env)


    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()