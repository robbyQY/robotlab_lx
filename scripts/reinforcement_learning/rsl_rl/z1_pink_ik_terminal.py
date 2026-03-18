# z1_pink_terminal.py
#
# Minimal Z1 Pink IK terminal controller for Isaac Lab / Isaac Sim
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
parser.add_argument("--mesh_dir", type=str, required=True)

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

from isaaclab.controllers.pink_ik import PinkIKController, PinkIKControllerCfg
from pink.tasks import FrameTask
import pinocchio as pin
from scipy.spatial.transform import Rotation as R

# -----------------------------------------------------------------------------
# robot constants from your URDF
# -----------------------------------------------------------------------------
EE_FRAME_NAME = "gripperMover"
# BASE_LINK_NAME = "link00"
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
    """XYZ euler -> quaternion (w, x, y, z)."""
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

def target_se3_from_pos_quat(pos, quat_wxyz):
    # scipy uses xyzw, while your code stores wxyz
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
    rot = R.from_quat(quat_xyzw).as_matrix()
    return pin.SE3(rot, np.array(pos, dtype=np.float64))

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
                # stiffness=800.0,
                # damping=80.0,
                effort_limit=600.0,
                # saturation_effort=60.0,
                velocity_limit=3.2,
                stiffness=800.0,
                damping=4.0,
                friction=0.0,                
            ),
        },
    )
import math
import numpy as np
import torch

# -------------------------
# helpers
# -------------------------
def quat_wxyz_to_rot(q):
    qw, qx, qy, qz = q
    R = np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)
    return R

def rot_to_quat_wxyz(R):
    # simple robust conversion
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return np.array([qw, qx, qy, qz], dtype=np.float64)

def world_pose_to_base_pose(target_pos_w, target_quat_w, base_pos_w, base_quat_w):
    R_w_b = quat_wxyz_to_rot(base_quat_w)     # base orientation in world
    R_w_t = quat_wxyz_to_rot(target_quat_w)   # target orientation in world

    # world -> base
    R_b_w = R_w_b.T
    p_b_t = R_b_w @ (target_pos_w - base_pos_w)
    R_b_t = R_b_w @ R_w_t
    q_b_t = rot_to_quat_wxyz(R_b_t)

    return p_b_t, q_b_t

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
    # wait one frame so handles/data are ready
    sim.step()
    scene.update(sim.get_physics_dt())

    print("All joint names:", robot.data.joint_names)
    print("All body names:", robot.data.body_names)

    # joint ids for the 6 arm joints
    arm_joint_ids, arm_joint_names = robot.find_joints(ARM_JOINT_NAMES)

    # all joints in USD articulation
    all_joint_names = list(robot.data.joint_names)

    ee_task = FrameTask(
        frame=EE_FRAME_NAME,
        position_cost=1.0,
        orientation_cost=0.2,   # start lower for debugging
        gain=0.5,               # slightly gentler
        lm_damping=1e-4,
    )

    pink_cfg = PinkIKControllerCfg(
        urdf_path=args.robot_urdf,
        mesh_path=args.mesh_dir,
        num_hand_joints=0,
        variable_input_tasks=[ee_task],
        fixed_input_tasks=[],
        joint_names=ARM_JOINT_NAMES,
        all_joint_names=all_joint_names,
        articulation_name="robot",
        base_link_name=BASE_LINK_NAME,   # must exist in your articulation
        show_ik_warnings=True,
        fail_on_joint_limit_violation=False,
        xr_enabled=False,
    )

    pink = PinkIKController(
        cfg=pink_cfg,
        robot_cfg=scene_cfg.robot,
        device=sim.device,
        controlled_joint_indices=arm_joint_ids,
    )

    shared_target = SharedTarget()
    # input_thread = threading.Thread(target=terminal_input_loop, args=(shared_target,), daemon=True)
    # input_thread.start()
    shared_target.set_target(
        0.35,   # x
        0.0,  # y
        0.30,   # z
        0.0,    # roll
        0.0,   # pitch
        0.0     # yaw
    )
    print("\nSimulation started.")
    print("EE frame:", EE_FRAME_NAME)
    print("Base link:", BASE_LINK_NAME)
    print("Controlled joints:", ARM_JOINT_NAMES)
    # ee_body_id = robot.find_bodies(EE_FRAME_NAME)[0]
    # ee_body_id = robot.find_bodies(EE_FRAME_NAME)[0][0].item()   # get scalar int
    import math

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


    ee_body_ids, ee_body_names = robot.find_bodies(EE_FRAME_NAME)
    ee_body_id = ee_body_ids[0]

    base_body_ids, base_body_names = robot.find_bodies(BASE_LINK_NAME)
    base_body_id = base_body_ids[0]

    print("EE body:", ee_body_names[0], ee_body_id)
    print("Base body:", base_body_names[0], base_body_id)

    # while simulation_app.is_running():
    #     dt = sim.get_physics_dt()

    #     robot.update(dt)
    #     # --- EE pose ---
    #     ee_state = robot.data.body_state_w[0, ee_body_id]   # (13,)
    #     ee_pos = ee_state[0:3]
    #     ee_quat = ee_state[3:7]

    #     roll, pitch, yaw = quat_to_rpy(ee_quat.detach().cpu().numpy())

    #     print("-" * 50)
    #     print("EE xyz:", ee_pos.detach().cpu().numpy())
    #     print("EE quat [w x y z]:", ee_quat.detach().cpu().numpy())
    #     print("EE rpy [rad]:", [roll, pitch, yaw])
    #     print("EE rpy [deg]:", [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)])

    #     target_pos_np, target_quat_np = shared_target.get_target()
    #     target_pos = torch.tensor(target_pos_np, dtype=torch.float32, device=sim.device).unsqueeze(0)
    #     target_quat = torch.tensor(target_quat_np, dtype=torch.float32, device=sim.device).unsqueeze(0)

    #     # set Pink target
    #     target_se3 = target_se3_from_pos_quat(target_pos_np, target_quat_np)
    #     ee_task.set_target(target_se3)

    #     joint_pos_all = robot.data.joint_pos.clone()                    # torch, shape (1, 7)
    #     curr_arm_q = joint_pos_all[0, arm_joint_ids].detach().cpu().numpy()   # shape (6,)
    #     q_des_arm = pink.compute(curr_arm_q, dt)
    #     print("curr q:", curr_joint_pos_1d[:6])
    #     print("q_des :", q_des_arm.detach().cpu().numpy())
    #     print("delta :", q_des_arm.detach().cpu().numpy() - curr_joint_pos_1d[:6])

    #     q_des_full = joint_pos_all.clone()                             # torch, shape (1, 7)
    #     q_des_full[0, arm_joint_ids] = q_des_arm.to(q_des_full.device, dtype=q_des_full.dtype)
    #     # q_des_full[0, arm_joint_ids] = torch.tensor(
    #     #     [0.0,  0.0, -0.8,  0.0,
    #     #     0.0, 0.0],
    #     #     device=q_des_full.device,
    #     #     dtype=q_des_full.dtype
    #     # )        
    #     print(q_des_full[0, arm_joint_ids])
    #     print(q_des_full)
    #     robot.set_joint_position_target(q_des_full)

    #     # push to sim
    #     robot.write_data_to_sim()
    #     sim.step()
    #     scene.update(dt)

    while simulation_app.is_running():
        dt = sim.get_physics_dt()

        robot.update(dt)

        # current EE pose in world
        ee_state = robot.data.body_state_w[0, ee_body_id]
        ee_pos_w = ee_state[0:3].detach().cpu().numpy()
        ee_quat_w = ee_state[3:7].detach().cpu().numpy()

        # current base pose in world
        base_state = robot.data.body_state_w[0, base_body_id]
        base_pos_w = base_state[0:3].detach().cpu().numpy()
        base_quat_w = base_state[3:7].detach().cpu().numpy()

        roll, pitch, yaw = quat_to_rpy(ee_quat_w)

        print("-" * 50)
        print("EE xyz world:", ee_pos_w)
        print("EE quat world [w x y z]:", ee_quat_w)
        print("EE rpy deg:", [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)])

        # user target is assumed in WORLD frame
        target_pos_w, target_quat_w = shared_target.get_target()

        # convert WORLD target -> BASE target for Pink
        target_pos_b, target_quat_b = world_pose_to_base_pose(
            np.array(target_pos_w, dtype=np.float64),
            np.array(target_quat_w, dtype=np.float64),
            np.array(base_pos_w, dtype=np.float64),
            np.array(base_quat_w, dtype=np.float64),
        )

        # set Pink target in BASE frame
        target_se3 = target_se3_from_pos_quat(target_pos_b, target_quat_b)
        ee_task.set_target(target_se3)

        # pass ONLY the 6 arm joints to Pink
        joint_pos_all = robot.data.joint_pos.clone()

        # Pink.compute expects ALL joints in cfg.all_joint_names order
        curr_joint_pos_full = joint_pos_all[0].detach().cpu().numpy()   # shape (7,)
        curr_arm_q = curr_joint_pos_full[arm_joint_ids]                 # shape (6,), debug only

        q_des_arm = pink.compute(curr_joint_pos_full, dt)               # returns shape (6,)
        q_des_arm_np = q_des_arm.detach().cpu().numpy()

        print("curr full q:", curr_joint_pos_full)
        print("curr arm q :", curr_arm_q)
        print("q_des_arm  :", q_des_arm_np)
        print("delta arm  :", q_des_arm_np - curr_arm_q)

        # if it still returns exactly current q, Pink likely failed
        if np.allclose(q_des_arm_np, curr_arm_q, atol=1e-8):
            print("[WARN] Pink returned current joint positions unchanged. Likely IK failure.")

        q_des_full = joint_pos_all.clone()
        q_des_full[0, arm_joint_ids] = q_des_arm.to(q_des_full.device, dtype=q_des_full.dtype)

        robot.set_joint_position_target(q_des_full)
        robot.write_data_to_sim()

        sim.step()
        scene.update(dt)
        
    simulation_app.close()


if __name__ == "__main__":
    main()