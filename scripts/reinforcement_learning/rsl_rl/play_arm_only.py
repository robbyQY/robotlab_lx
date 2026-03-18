#!/usr/bin/env python3
# Copyright (c) 2024-2026
# Minimal B2+Z1 arm-only Pink IK demo for Isaac Lab

import argparse
import threading
import time
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

# -----------------------------------------------------------------------------
# Launch Isaac Sim first
# -----------------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="B2+Z1 arm-only Pink IK control")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--headless", action="store_true", default=False)

parser.add_argument(
    "--asset_path",
    type=str,
    required=True,
    help="Absolute path to the combined B2+Z1 URDF, e.g. .../b2_description_b2z1.urdf",
)
parser.add_argument(
    "--urdf_package_root",
    type=str,
    required=True,
    help="Directory containing package folders referenced by package:// paths, e.g. root with b2_description/ and z1_description/",
)
parser.add_argument(
    "--fix_base",
    action="store_true",
    default=True,
    help="Fix the B2 base for arm-only debugging.",
)
parser.add_argument(
    "--dt",
    type=float,
    default=1.0 / 120.0,
    help="Simulation timestep.",
)
parser.add_argument(
    "--ee_pose",
    type=float,
    nargs=6,
    default=[0.30, 0.00, 0.20, 0.0, 1.57, 0.0],
    metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
    help="Initial end-effector target pose in base_link frame.",
)
parser.add_argument(
    "--position_only",
    action="store_true",
    default=False,
    help="Ignore EE orientation and track position only.",
)
parser.add_argument(
    "--print_every",
    type=int,
    default=20,
    help="Print debug info every N sim steps.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Now import Isaac Lab / Pink bits
# -----------------------------------------------------------------------------
import pinocchio as pin
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import DCMotorCfg
from isaaclab.controllers import PinkIKController, PinkIKControllerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from pink.tasks import FrameTask


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
@dataclass
class ArmTargetState:
    xyz_b: np.ndarray
    quat_wxyz_b: np.ndarray


def quat_wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)


def quat_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)


def rpy_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    q_xyzw = R.from_euler("xyz", [roll, pitch, yaw]).as_quat()
    return quat_xyzw_to_wxyz(q_xyzw)


def quat_wxyz_to_rpy(q_wxyz: np.ndarray) -> np.ndarray:
    q_xyzw = quat_wxyz_to_xyzw(q_wxyz)
    return R.from_quat(q_xyzw).as_euler("xyz")


def print_help():
    print()
    print("[ARM TERMINAL CONTROL]")
    print("Input EE target in base_link frame:")
    print("  x y z roll pitch yaw")
    print("Example:")
    print("  0.30 0.00 0.20 0.0 1.57 0.0")
    print("Commands:")
    print("  p      print current target")
    print("  h      help")
    print("  q      quit input thread")
    print()


def input_thread_fn(target_state: ArmTargetState, lock: threading.Lock, stop_evt: threading.Event):
    print_help()
    while not stop_evt.is_set():
        try:
            line = input("[ee target in base_link] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not line:
            continue

        if line.lower() in {"h", "help"}:
            print_help()
            continue

        if line.lower() in {"q", "quit", "exit"}:
            break

        if line.lower() in {"p", "print"}:
            with lock:
                xyz = target_state.xyz_b.copy()
                rpy = quat_wxyz_to_rpy(target_state.quat_wxyz_b.copy())
            print(f"[TARGET] xyz={xyz}, rpy={rpy}")
            continue

        vals = line.replace(",", " ").split()
        if len(vals) != 6:
            print("[WARN] Expected 6 values: x y z roll pitch yaw")
            continue

        try:
            vals = [float(v) for v in vals]
        except ValueError:
            print("[WARN] Invalid numeric input.")
            continue

        xyz = np.array(vals[:3], dtype=np.float64)
        quat = rpy_to_quat_wxyz(*vals[3:])

        with lock:
            target_state.xyz_b = xyz
            target_state.quat_wxyz_b = quat

        print(f"[UPDATED] xyz={xyz}, rpy={np.array(vals[3:], dtype=np.float64)}")


# -----------------------------------------------------------------------------
# Robot config
# -----------------------------------------------------------------------------
UNITREE_B2_ARM_ONLY_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=args_cli.fix_base,
        merge_fixed_joints=False,   # important for Pink / URDF consistency
        replace_cylinders_with_capsules=False,
        asset_path=args_cli.asset_path,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=0.0,
            )
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.58),
        joint_pos={
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 0.8,
            "RR_thigh_joint": 0.8,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
            "joint1": 0.01,
            "joint2": 1.48,
            "joint3": -0.63,
            "joint4": -0.84,
            "joint5": 0.0,
            "joint6": 1.57,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip": DCMotorCfg(
            joint_names_expr=[".*_hip_joint"],
            effort_limit=200.0,
            saturation_effort=200.0,
            velocity_limit=23.0,
            stiffness=160.0,
            damping=5.0,
            friction=0.0,
        ),
        "thigh": DCMotorCfg(
            joint_names_expr=[".*_thigh_joint"],
            effort_limit=200.0,
            saturation_effort=200.0,
            velocity_limit=23.0,
            stiffness=160.0,
            damping=5.0,
            friction=0.0,
        ),
        "calf": DCMotorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=320.0,
            saturation_effort=320.0,
            velocity_limit=14.0,
            stiffness=160.0,
            damping=5.0,
            friction=0.0,
        ),
        "arm": DCMotorCfg(
            joint_names_expr=["joint[1-6]"],
            effort_limit=60.0,
            saturation_effort=60.0,
            velocity_limit=3.2,
            stiffness=80.0,
            damping=4.0,
            friction=0.0,
        ),
    },
)


@configclass
class ArmOnlySceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot = UNITREE_B2_ARM_ONLY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    sim_cfg = sim_utils.SimulationCfg(dt=args_cli.dt, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 1.5], [0.0, 0.0, 0.5])

    scene_cfg = ArmOnlySceneCfg(num_envs=1, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.reset()

    robot = scene["robot"]

    # joint ids
    leg_joint_names = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    ]
    arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

    leg_joint_ids, _ = robot.find_joints(leg_joint_names, preserve_order=True)
    arm_joint_ids, _ = robot.find_joints(arm_joint_names, preserve_order=True)

    leg_joint_ids = leg_joint_ids.tolist() if hasattr(leg_joint_ids, "tolist") else list(leg_joint_ids)
    arm_joint_ids = arm_joint_ids.tolist() if hasattr(arm_joint_ids, "tolist") else list(arm_joint_ids)

    base_link_ids, _ = robot.find_bodies(["base_link"], preserve_order=True)
    ee_link_ids, _ = robot.find_bodies(["link06"], preserve_order=True)

    base_link_id = int(base_link_ids[0])
    ee_link_id = int(ee_link_ids[0])

    print("[INFO] arm_joint_ids:", arm_joint_ids)
    print("[INFO] leg_joint_ids:", leg_joint_ids)
    print("[INFO] base_link_id:", base_link_id, "ee_link_id:", ee_link_id)
    print("[INFO] joint names:", robot.data.joint_names)

    # fixed standing leg target
    leg_target = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.8, 0.8, 0.8, 0.8, -1.5, -1.5, -1.5, -1.5]],
        dtype=torch.float32,
        device=robot.device,
    )

    # Pink task
    ee_task = FrameTask(
        frame="link06",
        position_cost=1.0,
        orientation_cost=0.0 if args_cli.position_only else 0.2,
    )

    pink_cfg = PinkIKControllerCfg(
        urdf_path=args_cli.asset_path,
        mesh_path=args_cli.urdf_package_root,
        joint_names=arm_joint_names,
        all_joint_names=robot.data.joint_names,
        articulation_name="robot",
        base_link_name="base_link",
        variable_input_tasks=[ee_task],
        fixed_input_tasks=[],
        show_ik_warnings=True,
        fail_on_joint_limit_violation=False,
    )

    arm_pink_controller = PinkIKController(
        cfg=pink_cfg,
        robot_cfg=robot.cfg,
        device=str(robot.device),
        controlled_joint_indices=arm_joint_ids,
    )

    # target in base_link frame
    target_state = ArmTargetState(
        xyz_b=np.array(args_cli.ee_pose[:3], dtype=np.float64),
        quat_wxyz_b=rpy_to_quat_wxyz(*args_cli.ee_pose[3:]),
    )
    target_lock = threading.Lock()
    stop_evt = threading.Event()

    th = threading.Thread(target=input_thread_fn, args=(target_state, target_lock, stop_evt), daemon=True)
    th.start()

    step_count = 0
    try:
        while simulation_app.is_running():
            # keep legs fixed
            robot.set_joint_position_target(leg_target, joint_ids=leg_joint_ids)

            # current target in base frame
            with target_lock:
                target_xyz_b = target_state.xyz_b.copy()
                target_quat_wxyz_b = target_state.quat_wxyz_b.copy()

            # read base pose in world
            base_pos_w = robot.data.body_pos_w[0, base_link_id].detach().cpu().numpy()
            base_quat_wxyz = robot.data.body_quat_w[0, base_link_id].detach().cpu().numpy()

            base_rot_w = R.from_quat(quat_wxyz_to_xyzw(base_quat_wxyz)).as_matrix()
            target_rot_b = R.from_quat(quat_wxyz_to_xyzw(target_quat_wxyz_b)).as_matrix()

            # convert base-frame target -> world-frame target for Pink FrameTask
            target_pos_w = base_rot_w @ target_xyz_b + base_pos_w
            target_rot_w = base_rot_w @ target_rot_b
            target_tf_w = pin.SE3(target_rot_w, target_pos_w)
            ee_task.set_target(target_tf_w)

            # Pink solve
            curr_joint_pos = robot.data.joint_pos[0].detach().cpu().numpy()
            arm_joint_target = arm_pink_controller.compute(curr_joint_pos=curr_joint_pos, dt=args_cli.dt)
            arm_joint_target = arm_joint_target.unsqueeze(0).repeat(scene.num_envs, 1)

            # send arm target
            robot.set_joint_position_target(arm_joint_target, joint_ids=arm_joint_ids)

            # step sim
            scene.write_data_to_sim()
            sim.step()
            scene.update(args_cli.dt)

            # debug print after physics step
            if step_count % args_cli.print_every == 0:
                ee_pos_w = robot.data.body_pos_w[0, ee_link_id].detach().cpu().numpy()
                ee_quat_wxyz = robot.data.body_quat_w[0, ee_link_id].detach().cpu().numpy()

                ee_rot_w = R.from_quat(quat_wxyz_to_xyzw(ee_quat_wxyz)).as_matrix()
                ee_pos_b = base_rot_w.T @ (ee_pos_w - base_pos_w)
                ee_rot_b = base_rot_w.T @ ee_rot_w
                ee_rpy_b = R.from_matrix(ee_rot_b).as_euler("xyz")

                pos_err_b = target_xyz_b - ee_pos_b

                print("-" * 80)
                print("[TARGET base] xyz:", target_xyz_b)
                print("[TARGET base] rpy:", quat_wxyz_to_rpy(target_quat_wxyz_b))
                print("[CURRENT base] xyz:", ee_pos_b)
                print("[CURRENT base] rpy:", ee_rpy_b)
                print("[ERR base] xyz:", pos_err_b)
                print("[TARGET world] xyz:", target_pos_w)
                print("[CURRENT world] xyz:", ee_pos_w)
                print("[ARM joints curr]:", curr_joint_pos[arm_joint_ids])
                print("[ARM joints cmd ]:", arm_joint_target[0].detach().cpu().numpy())

            step_count += 1

    finally:
        stop_evt.set()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()