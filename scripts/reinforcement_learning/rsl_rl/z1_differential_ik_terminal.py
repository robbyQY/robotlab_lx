# z1_diffik_terminal.py
#
# Minimal Z1 Differential IK terminal controller for Isaac Lab / Isaac Sim
#
# Terminal input:
#   x y z roll pitch yaw
# Example:
#   0.30 0.00 0.25 0.0 1.57 0.0

import argparse
import math
import threading
import numpy as np
import torch

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# launch app first
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--robot_urdf", type=str, required=True)
parser.add_argument("--mesh_dir", type=str, required=True)  # kept for compatibility, unused

AppLauncher.add_app_launcher_args(parser)

args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# imports after app launch
# -----------------------------------------------------------------------------
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms

# -----------------------------------------------------------------------------
# robot constants
# -----------------------------------------------------------------------------
EE_FRAME_NAME = "gripperMover"
BASE_LINK_NAME = "link01"

ARM_JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
]

# -----------------------------------------------------------------------------
# helper functions
# -----------------------------------------------------------------------------
def euler_xyz_to_quat_wxyz(roll, pitch, yaw):
    """XYZ Euler -> quaternion (w, x, y, z)."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z], dtype=np.float32)

def quat_to_rpy(q):
    qw, qx, qy, qz = q

    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

class SharedTarget:
    def __init__(self):
        self.pos = np.array([0.30, 0.00, 0.25], dtype=np.float32)
        self.quat = euler_xyz_to_quat_wxyz(0.0, 1.57, 0.0)
        self.lock = threading.Lock()

    def set_target(self, x, y, z, r, p, yw):
        with self.lock:
            self.pos[:] = [x, y, z]
            self.quat[:] = euler_xyz_to_quat_wxyz(r, p, yw)

    def get_target(self):
        with self.lock:
            return self.pos.copy(), self.quat.copy()

def terminal_input_loop(shared_target: SharedTarget):
    print("\nType target pose as:")
    print("  x y z roll pitch yaw")
    print("Example:")
    print("  0.30 0.00 0.25 0.0 1.57 0.0\n")

    while True:
        try:
            text = input("target> ").strip()
            if not text:
                continue
            vals = [float(v) for v in text.split()]
            if len(vals) != 6:
                print("Need 6 numbers: x y z roll pitch yaw")
                continue
            shared_target.set_target(*vals)
            print("Target updated.")
        except EOFError:
            break
        except Exception as e:
            print(f"Invalid input: {e}")

# -----------------------------------------------------------------------------
# scene config
# -----------------------------------------------------------------------------
@configclass
class MySceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot = ArticulationCfg(
        prim_path="/World/Z1",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=args.robot_urdf,
            fix_base=True,
            merge_fixed_joints=True,
            activate_contact_sensors=False,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=1.0,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=800.0,
                    damping=80.0,
                )
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "joint1": 0.0,
                "joint2": 0.0,
                "joint3": -0.8,
                "joint4": 0.0,
                "joint5": 0.0,
                "joint6": 0.0,
                "jointGripper": 0.0,
            },
        ),
        actuators={
            "z1_arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-6]", "jointGripper"],
                # effort_limit=60.0,
                # velocity_limit=3.2,
                stiffness=80.0,
                damping=40.0,
                # friction=0.0,
            ),
        },
    )

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.5, 1.5, 1.0], [0.0, 0.0, 0.2])

    scene_cfg = MySceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.reset()

    robot = scene["robot"]

    # force a known initial joint pose
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:] = 0.0
    joint_vel[:] = 0.0

    name_to_idx = {name: i for i, name in enumerate(robot.data.joint_names)}
    joint_pos[0, name_to_idx["joint1"]] = 0.0
    joint_pos[0, name_to_idx["joint2"]] = 0.0
    joint_pos[0, name_to_idx["joint3"]] = -0.8
    joint_pos[0, name_to_idx["joint4"]] = 0.0
    joint_pos[0, name_to_idx["joint5"]] = 0.0
    joint_pos[0, name_to_idx["joint6"]] = 0.0
    joint_pos[0, name_to_idx["jointGripper"]] = 0.0

    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)
    robot.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())

    print("All joint names:", robot.data.joint_names)
    print("All body names:", robot.data.body_names)

    # resolve arm joints and EE body
    arm_joint_ids, arm_joint_names = robot.find_joints(ARM_JOINT_NAMES)
    ee_body_ids, ee_body_names = robot.find_bodies(EE_FRAME_NAME)
    ee_body_id = ee_body_ids[0]

    # fixed-base robot: Jacobian index is body_id - 1
    ee_jacobi_idx = ee_body_id - 1 if robot.is_fixed_base else ee_body_id

    print("\nSimulation started.")
    print("EE frame:", EE_FRAME_NAME)
    print("Base link:", BASE_LINK_NAME)
    print("Controlled joints:", ARM_JOINT_NAMES)
    print("EE body:", ee_body_names[0], ee_body_id)
    print("EE jacobian index:", ee_jacobi_idx)

    # Differential IK controller
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
    )
    diff_ik = DifferentialIKController(diff_ik_cfg, num_envs=1, device=sim.device)

    shared_target = SharedTarget()
    input_thread = threading.Thread(target=terminal_input_loop, args=(shared_target,), daemon=True)
    input_thread.start()

    # initial target in BASE frame
    target_pos_b_np, target_quat_b_np = shared_target.get_target()
    ik_command = torch.tensor(
        np.concatenate([target_pos_b_np, target_quat_b_np])[None, :],
        dtype=torch.float32,
        device=sim.device,
    )
    diff_ik.reset()
    diff_ik.set_command(ik_command)

    while simulation_app.is_running():
        dt = sim.get_physics_dt()
        robot.update(dt)

        # current EE pose in world
        ee_pose_w = robot.data.body_pose_w[:, ee_body_id]         # (1, 7)
        root_pose_w = robot.data.root_pose_w                      # (1, 7)

        # convert EE world pose -> base frame pose
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7],
        )

        # get arm Jacobian and current arm joint positions
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids]
        joint_pos_arm = robot.data.joint_pos[:, arm_joint_ids]

        # update target from terminal (interpreted in BASE frame)
        target_pos_b_np, target_quat_b_np = shared_target.get_target()
        ik_command = torch.tensor(
            np.concatenate([target_pos_b_np, target_quat_b_np])[None, :],
            dtype=torch.float32,
            device=sim.device,
        )
        diff_ik.set_command(ik_command)

        # solve IK -> desired arm joint positions
        joint_pos_des_arm = diff_ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos_arm)

        # debug print
        ee_pos_b_np = ee_pos_b[0].detach().cpu().numpy()
        ee_quat_b_np = ee_quat_b[0].detach().cpu().numpy()
        roll, pitch, yaw = quat_to_rpy(ee_quat_b_np)

        # print("-" * 50)
        # print("EE xyz base:", ee_pos_b_np)
        # print("EE quat base [w x y z]:", ee_quat_b_np)
        # print("EE rpy base [deg]:", [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)])
        # print("curr arm q :", joint_pos_arm[0].detach().cpu().numpy())
        # print("q_des_arm  :", joint_pos_des_arm[0].detach().cpu().numpy())
        # print("delta arm  :", (joint_pos_des_arm[0] - joint_pos_arm[0]).detach().cpu().numpy())
        # joint_pos_des_arm[0, arm_joint_ids] = torch.tensor(
        #     [0.8,  0.8, -0.8,  0.0,
        #     0.0, 0.0],
        #     device=joint_pos_des_arm.device,
        #     dtype=joint_pos_des_arm.dtype
        # )     
        # apply only to the arm joints
        robot.set_joint_position_target(joint_pos_des_arm, joint_ids=arm_joint_ids)
        robot.write_data_to_sim()

        sim.step()
        scene.update(dt)

    simulation_app.close()

if __name__ == "__main__":
    main()