from .common import RobotBase
import pybullet_data
import math
import copy
import numpy as np
import pybullet as p
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


class KukaBase(RobotBase):

    def __init__(self, urdf_root_path=pybullet_data.getDataPath(),
                 time_step=0.01,
                 action_dim=None):

        self.fingerAForce = 2
        self.fingerBForce = 2.5
        self.fingerTipForce = 2
        self.useSimulation = True
        self.useNullSpace = 21
        #self.useOrientation = True
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

        n_robot_joints = 7
        self.robot_base_pos = (-0.1, 0, 0.07)

        self.init_endeffector_pos = [0.537, 0.0, 0.5]
        self.init_endeffector_orn = p.getQuaternionFromEuler([0, -math.pi, 0])
        self.init_endeffector_angle = 0
        self.init_finger_angle = 0.3

        # init_joint_positions = [
        #     0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684,
        #     -0.006539, 0.000048,
        #     -0.299912, 0.000000, -0.000043, 0.299960, 0.000000, -0.000200
        # ]
        body_path = os.path.join(urdf_root_path,
                                 "kuka_iiwa/kuka_with_gripper2.sdf")
        body_id, self.motor_names, motor_indices \
            = self._make_robot(body_path, self.robot_base_pos)
        endeffector_index = motor_indices[6]
        self.gripper_index = motor_indices[7]

        # オートで計算したいが変な体勢から始まるため、無理やり、ロボットの初期の姿勢を作る
        init_joint_pos = [
            0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684,
            -0.006539, 0.000048,
            -0.299912, 0.000000, -0.000043, 0.299960, 0.000000, -0.000200
        ]
        # cal joint
        # joint_pos = p.calculateInverseKinematics(
        #     body_id, self.endeffector_index, self.init_endeffector_pos,
        #     self.init_endeffector_orn,
        #     jointDamping=self.jd)
        super().__init__(body_id, motor_indices[:n_robot_joints],
                         init_joint_pos[:n_robot_joints],
                         endeffector_index,
                         self.init_endeffector_pos,
                         self.init_endeffector_orn,
                         time_step, action_dim)

        self._reset_hand_joint_pos(self.init_endeffector_angle,
                                   self.init_finger_angle)

    def _reset_hand_joint_pos(self, endeffector_angle, finger_angle):
        # fingers

        p.resetJointState(self.body_id, 7,
                          endeffector_angle)
        p.resetJointState(self.body_id, 8,
                          -finger_angle)
        p.resetJointState(self.body_id, 11,
                          finger_angle)
        p.resetJointState(self.body_id, 10,
                          0)
        p.resetJointState(self.body_id, 13,
                          0)

    def _set_hand_joint_pos(self, endeffector_angle, finger_angle):
        # fingers
        p.setJointMotorControl2(self.body_id,
                                7,
                                p.POSITION_CONTROL,
                                targetPosition=endeffector_angle,
                                force=self.max_force)
        p.setJointMotorControl2(self.body_id,
                                8,
                                p.POSITION_CONTROL,
                                targetPosition=-finger_angle,
                                force=self.fingerAForce)
        p.setJointMotorControl2(self.body_id,
                                11,
                                p.POSITION_CONTROL,
                                targetPosition=finger_angle,
                                force=self.fingerBForce)

        p.setJointMotorControl2(self.body_id,
                                10,
                                p.POSITION_CONTROL,
                                targetPosition=0,
                                force=self.fingerTipForce)
        p.setJointMotorControl2(self.body_id,
                                13,
                                p.POSITION_CONTROL,
                                targetPosition=0,
                                force=self.fingerTipForce)

    def reset(self):
        # self.endeffector_pos = [0.537, 0.0, 0.5]
        # self.endeffector_angle = 0

        super().reset()

        self._reset_hand_joint_pos(self.init_endeffector_angle,
                                   self.init_finger_angle)
        self.endeffector_angle = self.init_endeffector_angle

    def apply_action(self, action):
        raise NotImplementedError


class KukaInverseKinematics(KukaBase):
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

        state = p.getLinkState(self.body_id, self.endeffector_index)

        self.endeffector_pos = list(state[4])

        self.endeffector_pos[0] = self.endeffector_pos[0] + dx
        self.endeffector_pos[1] = self.endeffector_pos[1] + dy
        self.endeffector_pos[2] = self.endeffector_pos[2] + dz

        self.endeffector_angle = self.endeffector_angle + da
        pos = self.endeffector_pos
        orn = p.getQuaternionFromEuler([0, -math.pi, 0])  # -math.pi,yaw])
        jointPoses = p.calculateInverseKinematics(self.body_id,
                                                  self.endeffector_index,
                                                  pos,
                                                  orn, self.ll,
                                                  self.ul, self.jr,
                                                  self.rp)

        self._set_robot_joint_pos(self.motor_indices[:self.n_robot_joints],
                                  jointPoses[:self.n_robot_joints])
        self._set_hand_joint_pos(self.init_endeffector_angle,
                                 fingerAngle)
        p.stepSimulation()


class KukaDirect(KukaBase):
    def __init__(self, urdf_root_path=pybullet_data.getDataPath(),
                 time_step=0.01):
        super().__init__(urdf_root_path, time_step)

    def get_state(self):
        info = {}
        observation = []
        for j_i in self.motor_indices:
            state = p.getJointState(self.BodyId, j_i)
            observation.append(state[0])

        info['motor_indeices'] = self.motor_indices
        info['joint_poses'] = observation

        return info

    def apply_action(self, action):
        jointPoses = self.get_state()
        for c_i, cmd in enumerate(action):
            jointPoses[c_i] += cmd
            motor = self.motorIndices[c_i]
            p.setJointMotorControl2(bodyUniqueId=self.BodyId,
                                    jointIndex=motor,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[c_i],
                                    targetVelocity=0,
                                    force=self.max_force,
                                    maxVelocity=self.max_velocity,
                                    positionGain=0.3,
                                    velocityGain=1)

        p.stepSimulation()
