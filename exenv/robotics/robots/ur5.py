from ..urdf.common import get_urdf_dir_path
from .common import RobotBase
import pybullet_data
import math
import copy
import numpy as np
from attrdict import AttrDict
from collections import namedtuple
import pybullet as p
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


def setup_sisbot(p, uid):
    # controlJoints = ["shoulder_pan_joint","shoulder_lift_joint",
    #                  "elbow_joint", "wrist_1_joint",
    #                  "wrist_2_joint", "wrist_3_joint",
    #                  "robotiq_85_left_knuckle_joint"]
    controlJoints = ["shoulder_pan_joint", "shoulder_lift_joint",
                     "elbow_joint", "wrist_1_joint",
                     "wrist_2_joint", "wrist_3_joint", 'left_gripper_motor', 'right_gripper_motor']

    jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
    numJoints = p.getNumJoints(uid)
    jointInfo = namedtuple("jointInfo",
                           ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity", "controllable"])
    joints = AttrDict()
    for i in range(numJoints):
        info = p.getJointInfo(uid, i)
        jointID = info[0]
        jointName = info[1].decode("utf-8")
        jointType = jointTypeList[info[2]]
        jointLowerLimit = info[8]
        jointUpperLimit = info[9]
        jointMaxForce = info[10]
        jointMaxVelocity = info[11]
        controllable = True if jointName in controlJoints else False
        info = jointInfo(jointID, jointName, jointType, jointLowerLimit,
                         jointUpperLimit, jointMaxForce, jointMaxVelocity,
                         controllable)
        if info.type == "REVOLUTE":  # set revolute joint to static
            p.setJointMotorControl2(
                uid, info.id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        joints[info.name] = info
    controlRobotiqC2 = False
    mimicParentName = False
    return joints, controlRobotiqC2, controlJoints, mimicParentName


class Ur5Base(RobotBase):

    def __init__(self, urdf_root_path=pybullet_data.getDataPath(),
                 time_step=0.01,
                 action_dim=None):
        #body_path = "../urdf/ur5_arm.urdf"
        urdf_path = get_urdf_dir_path()
        body_path = os.path.join(urdf_path, "ur5_arm.urdf")
        self.robotStartPos = [0.0, 0.0, 0.0]
        self.robotStartOrn = p.getQuaternionFromEuler([1.885, 1.786, 0.132])

        self.lastJointAngle = None
        self.active = False

        n_robot_joints = 6

        body_id, self.motor_names, self.motor_indices \
            = self._make_robot(body_path)
        a = 1

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

        # self.maxVelocity = .35
        # self.maxForce = 200.
        # self.fingerAForce = 2
        # self.fingerBForce = 2.5
        # self.fingerTipForce = 2
        # self.useSimulation = True
        # self.useNullSpace = 21
        # self.useOrientation = True
        # self.EndEffectorIndex = 6
        # self.GripperIndex = 7
        # # lower limits for null space
        # self.ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        # # upper limits for null space
        # self.ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        # # joint ranges for null space
        # self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        # # restposes for null space
        # self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
        # # joint damping coefficents
        # self.jd = [
        #     0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
        #     0.00001, 0.00001, 0.00001,
        #     0.00001, 0.00001, 0.00001, 0.00001
        # ]

        # init_joint_positions = [
        #     0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684,
        #     -0.006539, 0.000048,
        #     -0.299912, 0.000000, -0.000043, 0.299960, 0.000000, -0.000200
        # ]
        # body_path = os.path.join(urdf_root_path,
        #                          "kuka_iiwa/kuka_with_gripper2.sdf")
        super().__init__(body_path, init_joint_positions, time_step,
                         action_dim)

    def reset(self):
        self.endEffectorPos = [0.537, 0.0, 0.5]
        self.endEffectorAngle = 0

        super().reset()

    def apply_action(self, action):
        raise NotImplementedError


# class KukaInverseKinematics(KukaBase):
#     def __init__(self, urdf_root_path=pybullet_data.getDataPath(),
#                  time_step=0.01, action_dim=5):
#         super().__init__(urdf_root_path, time_step, action_dim)
#
#     def get_state(self):
#         info = {}
#         state = p.getLinkState(self.body_id, self.GripperIndex)
#         #pos = state[0]
#         #orn = state[1]
#         pos = state[4]
#         orn = state[5]
#         euler = p.getEulerFromQuaternion(orn)
#
#         info['endeffector_pos'] = list(pos)
#         info['endeffector_euler'] = list(euler)
#
#         return info
#
#     def get_observation_dim(self):
#         state = self.get_state()
#         return len(state)
#
#     def apply_action(self, action):
#
#         dx = action[0]
#         dy = action[1]
#         dz = action[2]
#         da = action[3]
#         fingerAngle = action[4]
#
#         state = p.getLinkState(self.body_id, self.EndEffectorIndex)
#         #actualEndEffectorPos = state[0]
#         #print("pos[2] (getLinkState(EndEffectorIndex)")
#         # print(actualEndEffectorPos[2])
#
#         self.endEffectorPos = list(state[4])
#
#         self.endEffectorPos[0] = self.endEffectorPos[0] + dx
#         # if (self.endEffectorPos[0] > 0.65):
#         #  self.endEffectorPos[0] = 0.65
#         # if (self.endEffectorPos[0] < 0.50):
#         #  self.endEffectorPos[0] = 0.50
#         self.endEffectorPos[1] = self.endEffectorPos[1] + dy
#         # if (self.endEffectorPos[1] < -0.17):
#         #  self.endEffectorPos[1] = -0.17
#         # if (self.endEffectorPos[1] > 0.22):
#         #  self.endEffectorPos[1] = 0.22
#
#         #print ("self.endEffectorPos[2]")
#         #print (self.endEffectorPos[2])
#         # print("actualEndEffectorPos[2]")
#         # print(actualEndEffectorPos[2])
#         # if (dz<0 or actualEndEffectorPos[2]<0.5):
#         self.endEffectorPos[2] = self.endEffectorPos[2] + dz
#
#         self.endEffectorAngle = self.endEffectorAngle + da
#         pos = self.endEffectorPos
#         orn = p.getQuaternionFromEuler([0, -math.pi, 0])  # -math.pi,yaw])
#         if (self.useNullSpace == 1):
#             if (self.useOrientation == 1):
#                 jointPoses = p.calculateInverseKinematics(self.body_id,
#                                                           self.EndEffectorIndex,
#                                                           pos,
#                                                           orn, self.ll,
#                                                           self.ul, self.jr,
#                                                           self.rp)
#             else:
#                 jointPoses = p.calculateInverseKinematics(self.body_id,
#                                                           self.EndEffectorIndex,
#                                                           pos,
#                                                           lowerLimits=self.ll,
#                                                           upperLimits=self.ul,
#                                                           jointRanges=self.jr,
#                                                           restPoses=self.rp)
#         else:
#             if (self.useOrientation == 1):
#                 jointPoses = p.calculateInverseKinematics(self.body_id,
#                                                           self.EndEffectorIndex,
#                                                           pos,
#                                                           orn,
#                                                           jointDamping=self.jd)
#             else:
#                 jointPoses = p.calculateInverseKinematics(
#                     self.body_id, self.EndEffectorIndex, pos)
#
#         # Eprint("jointPoses")
#         # print(jointPoses)
#         # print("self.EndEffectorIndex")
#         # print(self.EndEffectorIndex)
#         if (self.useSimulation):
#             for i in range(self.EndEffectorIndex + 1):
#                 # print(i)
#                 p.setJointMotorControl2(bodyUniqueId=self.body_id,
#                                         jointIndex=i,
#                                         controlMode=p.POSITION_CONTROL,
#                                         targetPosition=jointPoses[i],
#                                         targetVelocity=0,
#                                         force=self.maxForce,
#                                         maxVelocity=self.maxVelocity,
#                                         positionGain=0.3,
#                                         velocityGain=1)
#         else:
#             # reset the joint state (ignoring all dynamics, not recommended to use during simulation)
#             for i in range(self.numJoints):
#                 p.resetJointState(self.body_id, i, jointPoses[i])
#         # fingers
#         p.setJointMotorControl2(self.body_id,
#                                 7,
#                                 p.POSITION_CONTROL,
#                                 targetPosition=self.endEffectorAngle,
#                                 force=self.maxForce)
#         p.setJointMotorControl2(self.body_id,
#                                 8,
#                                 p.POSITION_CONTROL,
#                                 targetPosition=-fingerAngle,
#                                 force=self.fingerAForce)
#         p.setJointMotorControl2(self.body_id,
#                                 11,
#                                 p.POSITION_CONTROL,
#                                 targetPosition=fingerAngle,
#                                 force=self.fingerBForce)
#
#         p.setJointMotorControl2(self.body_id,
#                                 10,
#                                 p.POSITION_CONTROL,
#                                 targetPosition=0,
#                                 force=self.fingerTipForce)
#         p.setJointMotorControl2(self.body_id,
#                                 13,
#                                 p.POSITION_CONTROL,
#                                 targetPosition=0,
#                                 force=self.fingerTipForce)
#
#         p.stepSimulation()
#
#     def test(self):
#         state = p.getLinkState(self.body_id, self.GripperIndex)
#         pos = state[0]
#         orn = state[1]
#
#         jointPoses = p.calculateInverseKinematics(self.body_id,
#                                                   self.EndEffectorIndex,
#                                                   pos,
#                                                   orn,
#                                                   jointDamping=self.jd)
#         pos = state[4]
#         orn = state[5]
#
#         jointPoses_2 = p.calculateInverseKinematics(self.body_id,
#                                                     self.EndEffectorIndex,
#                                                     pos,
#                                                     orn,
#                                                     jointDamping=self.jd)
#
#         joint_list = []
#         for i in self.motor_indices:
#             state_1 = p.getJointInfo(self.body_id, i)
#             state_2 = p.getJointState(self.body_id, i)
#             joint_list.append(state_2[0])
#         a = 1
#
#
# class KukaDirect(KukaBase):
#     def __init__(self, urdf_root_path=pybullet_data.getDataPath(),
#                  time_step=0.01):
#         super().__init__(urdf_root_path, time_step)
#
#     def get_state(self):
#         info = {}
#         observation = []
#         for j_i in self.motor_indices:
#             state = p.getJointState(self.BodyId, j_i)
#             observation.append(state[0])
#
#         info['motor_indeices'] = self.motor_indices
#         info['joint_poses'] = observation
#
#         return info
#
#     def apply_action(self, action):
#         jointPoses = self.get_state()
#         for c_i, cmd in enumerate(action):
#             jointPoses[c_i] += cmd
#             motor = self.motorIndices[c_i]
#             p.setJointMotorControl2(bodyUniqueId=self.BodyId,
#                                     jointIndex=motor,
#                                     controlMode=p.POSITION_CONTROL,
#                                     targetPosition=jointPoses[c_i],
#                                     targetVelocity=0,
#                                     force=self.maxForce,
#                                     maxVelocity=self.maxVelocity,
#                                     positionGain=0.3,
#                                     velocityGain=1)
#
#         p.stepSimulation()
