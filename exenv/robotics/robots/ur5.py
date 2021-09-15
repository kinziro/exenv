from gym.utils.seeding import _int_list_from_bigint
from ..urdf import get_urdf_dir_path
from .common import RobotBase
import pybullet_data
import math
import copy
import numpy as np
# from attrdict import AttrDict
from collections import namedtuple
import pybullet as p
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


# def setup_sisbot(p, uid):
#     # controlJoints = ["shoulder_pan_joint","shoulder_lift_joint",
#     #                  "elbow_joint", "wrist_1_joint",
#     #                  "wrist_2_joint", "wrist_3_joint",
#     #                  "robotiq_85_left_knuckle_joint"]
#     controlJoints = ["shoulder_pan_joint", "shoulder_lift_joint",
#                      "elbow_joint", "wrist_1_joint",
#                      "wrist_2_joint", "wrist_3_joint", 'left_gripper_motor', 'right_gripper_motor']
#
#     jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
#     numJoints = p.getNumJoints(uid)
#     jointInfo = namedtuple("jointInfo",
#                            ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity", "controllable"])
#     joints = AttrDict()
#     for i in range(numJoints):
#         info = p.getJointInfo(uid, i)
#         jointID = info[0]
#         jointName = info[1].decode("utf-8")
#         jointType = jointTypeList[info[2]]
#         jointLowerLimit = info[8]
#         jointUpperLimit = info[9]
#         jointMaxForce = info[10]
#         jointMaxVelocity = info[11]
#         controllable = True if jointName in controlJoints else False
#         info = jointInfo(jointID, jointName, jointType, jointLowerLimit,
#                          jointUpperLimit, jointMaxForce, jointMaxVelocity,
#                          controllable)
#         if info.type == "REVOLUTE":  # set revolute joint to static
#             p.setJointMotorControl2(
#                 uid, info.id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
#         joints[info.name] = info
#     controlRobotiqC2 = False
#     mimicParentName = False
#     return joints, controlRobotiqC2, controlJoints, mimicParentName


class Ur5Base(RobotBase):

    def __init__(self, urdf_root_path=pybullet_data.getDataPath(),
                 time_step=0.01,
                 action_dim=None):
        #body_path = "../urdf/ur5_arm.urdf"
        urdf_path = get_urdf_dir_path()
        body_path = os.path.join(urdf_path, "ur5_arm.urdf")
        # self.robotStartPos = [0.0, 0.0, 0.0]
        # self.robotStartOrn = p.getQuaternionFromEuler([1.885, 1.786, 0.132])

        self.lastJointAngle = None
        self.active = False

        self.gripper_length = 0.2
        n_arm_joints = 6

        self.robot_base_pos = (0, 0, -0.1)
        self.robot_base_orn = p.getQuaternionFromEuler([0, 0, 3.1])
        #self.robot_base_orn = p.getQuaternionFromEuler([1.885, 1.786, 0.132])

        body_id, self.motor_names, motor_indices \
            = self._make_robot(body_path, base_pos=self.robot_base_pos,
                               base_orn=self.robot_base_orn)
        endeffector_index = motor_indices[5]
        self.gripper_index = motor_indices[6]
        # endeffector_index = motor_indices[6]
        # self.gripper_index = motor_indices[6]

        # if real:
        #     self.s = init_socket()

        #     if True:
        #         self.grip=RobotiqGripper("COM8")
        #         #grip.resetActivate()
        #         self.grip.reset()
        #         #grip.printInfo()
        #         self.grip.activate()
        #         #grip.printInfo()
        #         #grip.calibrate()

        # self.reset()
        # self.timeout = 0

        # self.fingerAForce = 2
        # self.fingerBForce = 2.5
        # self.fingerTipForce = 2
        # self.useSimulation = True
        # self.useNullSpace = 21
        # self.useOrientation = True
        # self.EndEffectorIndex = 6
        # self.GripperIndex = 7
        # lower limits for null space
        self.ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        # upper limits for null space
        self.ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        # joint ranges for null space
        self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        # restposes for null space
        self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
        # joint damping coefficents
        self.jd = [
            0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
            0.00001, 0.00001, 0.00001,
            0.00001, 0.00001, 0.00001, 0.00001
        ]

        # init_joint_positions = [
        #     0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684,
        #     -0.006539, 0.000048,
        #     -0.299912, 0.000000, -0.000043, 0.299960, 0.000000, -0.000200
        # ]
        # body_path = os.path.join(urdf_root_path,
        #                          "kuka_iiwa/kuka_with_gripper2.sdf")

        # オートで計算したいが変な体勢から始まるため、無理やり、ロボットの初期の姿勢を作る
        # init_joint_pos = [
        #     0, 0, -1.6, -0.75, 0, 0,
        # ]
        self.init_finger_angle = 0.012
        #self.init_finger_angle = 0.1
        # init_joint_pos = [0, np.pi, -0.5*np.pi,
        #                  -0.5*np.pi, -0.5*np.pi,
        #                  0.5 * np.pi, -self.init_finger_angle,
        #                  self.init_finger_angle]
        init_joint_pos = [np.pi, -0.5*np.pi,
                          -0.5*np.pi, -0.5*np.pi,
                          0.5 * np.pi, 0, -self.init_finger_angle,
                          self.init_finger_angle]
        #self.init_finger_angle = 0.012000000476837159
        # init_joint_pos = [0, np.pi, -1.9,
        #                   -1.4, -0.4*np.pi,
        #                   0.5*np.pi, 0.012000000476837159,
        #                   -0.012000000476837159]
        # init_joint_pos = [0, np.pi, -1.5820032364177563,
        #                   -1.2879050862601897, -0.5*np.pi,
        #                   0.5*np.pi, 0.012000000476837159,
        #                   -0.012000000476837159]
        # init_joint_pos = [0.15328961509984124, -1.8, -1.5820032364177563,
        #                   -1.2879050862601897, 1.5824233979484994,
        #                   0.19581299859677043, 0.012000000476837159,
        #                   -0.012000000476837159]
        # init_joint_pos = [
        #     0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684,
        # ]

        # # cal joint
        # joint_pos = p.calculateInverseKinematics(
        #     body_id, self.endeffector_index, self.init_endeffector_pos,
        #     self.init_endeffector_orn,
        #     jointDamping=self.jd)
        # ロボットの初期情報
        state = p.getLinkState(body_id,
                               endeffector_index)
        self.init_endeffector_pos = tuple(state[4])
        #self.init_endeffector_pos = [0.537, 0.0, 0.5]
        self.init_endeffector_orn = p.getQuaternionFromEuler([0, -math.pi, 0])
        self.init_endeffector_angle = 0

        super().__init__(body_id, n_arm_joints, motor_indices,
                         init_joint_pos,
                         endeffector_index,
                         time_step=time_step,
                         action_dim=action_dim)
        # super().__init__(body_id, n_robot_joints,
        #                  init_joint_pos[:n_robot_joints],
        #                  endeffector_index,
        #                  self.init_endeffector_pos,
        #                  self.init_endeffector_orn,
        #                  time_step, action_dim)

        # self._reset_hand_joint_pos(self.init_endeffector_angle,
        #                            self.init_finger_angle)

        state = p.getLinkState(body_id, self.endeffector_index)
        self.init_endeffector_pos = state[4]
        self.init_endeffector_orn = state[5]

    def _reset_hand_joint_pos(self, endeffector_angle, finger_angle):
        # fingers
        p.resetJointState(self.body_id, self.motor_indices[6],
                          -finger_angle)
        p.resetJointState(self.body_id, self.motor_indices[7],
                          finger_angle)

    def _set_hand_joint_pos(self, endeffector_angle, finger_angle):
        # fingers
        p.setJointMotorControl2(self.body_id,
                                self.motor_indices[6],
                                p.POSITION_CONTROL,
                                targetPosition=-finger_angle,
                                force=self.max_force)

        p.setJointMotorControl2(self.body_id,
                                self.motor_indices[7],
                                p.POSITION_CONTROL,
                                targetPosition=finger_angle,
                                force=self.max_force)

    def reset(self):
        super().reset()
        self._reset_hand_joint_pos(self.init_endeffector_angle,
                                   self.init_finger_angle)

        # 初期情報
        state = p.getLinkState(self.body_id,
                               self.endeffector_index)
        self.init_endeffector_pos = tuple(state[4])
        self.init_endeffector_orn = p.getQuaternionFromEuler([0, -math.pi, 0])
        self.init_endeffector_angle = 0

        self.endeffector_pos = self.init_endeffector_pos
        self.endeffector_angle = self.init_endeffector_angle

        p.stepSimulation()

    def apply_action(self, action):
        raise NotImplementedError

    def _grasping_action(self, value, threthold,
                         check_method=lambda v, t: v <= t):
        ret = False
        if check_method(value, threthold):
            finger_angle = self.init_finger_angle
            for _ in range(500):
                grasp_action = [0, 0, 0, 0, finger_angle]
                self.apply_action(grasp_action)
                finger_angle -= self.init_finger_angle / 100.
                # if finger_angle < 0:
                #     finger_angle = 0
                if finger_angle < -0.1:
                    finger_angle = -0.1
                finger_angle = -0.1

            for _ in range(250):
                grasp_action = [0, 0, 0.002, 0, finger_angle]
                self.apply_action(grasp_action)

            ret = True

        return ret


class Ur5InverseKinematics(Ur5Base):
    def __init__(self, urdf_root_path=pybullet_data.getDataPath(),
                 time_step=0.01, action_dim=5):
        super().__init__(urdf_root_path, time_step, action_dim)

    def get_state(self):
        info = {}
        state = p.getLinkState(self.body_id, self.gripper_index)
        # pos = state[0]
        # orn = state[1]
        pos = state[4]
        orn = state[5]
        euler = p.getEulerFromQuaternion(orn)

        info['endeffector_pos'] = list(pos)
        info['endeffector_euler'] = list(euler)

        return info

    def get_observation_dim(self):
        state = self.get_state()
        return len(state)

    def apply_action(self, action):

        dx = action[0]
        dy = action[1]
        dz = action[2]
        da = action[3]
        fingerAngle = action[4]
        #fingerAngle = self.init_finger_angle

        state = p.getLinkState(self.body_id, self.endeffector_index)

        self.endeffector_pos = list(state[4])

        self.endeffector_pos[0] = self.endeffector_pos[0] + dx
        self.endeffector_pos[1] = self.endeffector_pos[1] + dy
        self.endeffector_pos[2] = self.endeffector_pos[2] + dz

        self.endeffector_angle = self.endeffector_angle + da
        pos = self.endeffector_pos
        # orn = p.getQuaternionFromEuler([0, -math.pi, 0])  # -math.pi,yaw])
        orn = state[5]  # -math.pi,yaw])
        jointPoses = p.calculateInverseKinematics(self.body_id,
                                                  self.endeffector_index,
                                                  pos,
                                                  orn, self.ll,
                                                  self.ul, self.jr,
                                                  self.rp)
        # ハンドを開いておく
        jointPoses = list(jointPoses)
        # jointPoses[6] = -self.init_finger_angle
        # jointPoses[7] = self.init_finger_angle

        self._set_robot_joint_pos(self.motor_indices[:self.n_arm_joints],
                                  jointPoses[:self.n_arm_joints])
        self._set_hand_joint_pos(self.init_endeffector_angle,
                                 fingerAngle)
        p.stepSimulation()
