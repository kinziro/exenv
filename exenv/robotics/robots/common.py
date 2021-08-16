import pybullet as p
import os


class RobotBase:

    def __init__(self, body_id,
                 n_robot_joints,
                 init_robot_joint_positions,
                 time_step=0.01,
                 action_dim=None):
        self.body_id = body_id
        self.n_robot_joints = n_robot_joints
        self.init_robot_joint_positions = init_robot_joint_positions
        self.time_step = time_step

        # action and observation dimension
        self._action_dim = len(
            self.motor_indices) if action_dim is None else action_dim

        # set init pos
        self._reset_robot_joint_pos(self.n_robot_joints,
                                    self.init_robot_joint_positions)

    def _load_body(self, full_path):
        filepath = os.path.basename(full_path)
        if "sdf" in filepath:
            objects = p.loadSDF(full_path)
        elif "urdf" in filepath:
            objects = p.loadURDF(full_path)
        return objects[0]

    def _make_robot(self, body_path):
        # load body
        body_id = self._load_body(body_path)
        n_joints = p.getNumJoints(body_id)

        # get motor names and indices
        motor_names = []
        motor_indices = []
        for i in range(n_joints):
            joint_info = p.getJointInfo(body_id, i)
            qIndex = joint_info[3]
            if qIndex > -1:
                motor_names.append(str(joint_info[1]))
                motor_indices.append(i)

        return body_id, motor_names, motor_indices

    def _get_init_positions(self):
        raise NotImplementedError

    def _reset_robot_joint_pos(self, n_joints, joint_positions):
        for joint_index in range(n_joints):
            p.resetJointState(self.body_id, joint_index,
                              joint_positions[joint_index])
            # p.setJointMotorControl2(self.body_id,
            #                         joint_index,
            #                         p.POSITION_CONTROL,
            #                         targetPosition=joint_positions[joint_index],
            #                         force=self.maxForce)

    def _set_robot_joint_pos(self, n_joints, joint_positions):
        # set init position
        for i in range(n_joints):
            # print(i)
            p.setJointMotorControl2(bodyUniqueId=self.body_id,
                                    jointIndex=i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=joint_positions[i],
                                    targetVelocity=0,
                                    force=self.maxForce,
                                    maxVelocity=self.maxVelocity,
                                    positionGain=0.3,
                                    velocityGain=1)

    def reset(self):
        self._reset_robot_joint_pos(self.n_robot_joints,
                                    self.init_robot_joint_positions)

    def get_action_dim(self):
        return self._action_dim

    # def get_observation_dim(self):
    #     return len(self.get_bservation())

    def get_observation_dim(self):
        raise NotImplementedError

    def apply_action(self, action):
        raise NotImplementedError
