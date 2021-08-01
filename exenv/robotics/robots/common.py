import pybullet as p
import os


class RobotBase:

    def __init__(self, body_path,
                 init_joint_positions,
                 time_step=0.01,
                 action_dim=None):
        self.body_path = body_path
        self.time_step = time_step
        self.init_joint_positions = init_joint_positions

        # load body
        self.body_id = self._load_body(self.body_path)
        self.num_joints = p.getNumJoints(self.body_id)

        # get motor names and indices
        self.motor_names = []
        self.motor_indices = []
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.body_id, i)
            qIndex = joint_info[3]
            if qIndex > -1:
                self.motor_names.append(str(joint_info[1]))
                self.motor_indices.append(i)

        # action and observation dimension
        self._action_dim = len(
            self.motor_indices) if action_dim is None else action_dim

        # set init pos
        self._set_joint_pos(self.num_joints, self.init_joint_positions)

    def _load_body(self, full_path):
        filepath = os.path.basename(full_path)
        if "sdf" in filepath:
            objects = p.loadSDF(full_path)
        elif "urdf" in filepath:
            objects = p.loadURDF(full_path)
        return objects[0]

    def _get_init_positions(self):
        raise NotImplementedError

    def _set_joint_pos(self, num_joints, joint_positions):
        # set init position
        for joint_index in range(num_joints):
            p.resetJointState(self.body_id, joint_index,
                              joint_positions[joint_index])
            p.setJointMotorControl2(self.body_id,
                                    joint_index,
                                    p.POSITION_CONTROL,
                                    targetPosition=joint_positions[joint_index],
                                    force=self.maxForce)

    def reset(self):
        self._set_joint_pos(self.num_joints, self.init_joint_positions)

    def get_action_dim(self):
        return self._action_dim

    # def get_observation_dim(self):
    #     return len(self.get_bservation())

    def get_observation_dim(self):
        raise NotImplementedError

    def apply_action(self, action):
        raise NotImplementedError
