import pybullet as p
import os


class RobotBase:

    def __init__(self, body_id,
                 motor_indices,
                 init_robot_joint_pos,
                 endeffector_index,
                 time_step=0.01,
                 action_dim=None):
        self.body_id = body_id
        self.endeffector_index = endeffector_index
        self.motor_indices = motor_indices
        self.n_robot_joints = len(motor_indices)
        self.init_robot_joint_pos = init_robot_joint_pos
        self.time_step = time_step

        # simulation config
        self.max_velocity = 0.35
        self.max_force = 200.0

        # action and observation dimension
        self._action_dim = len(
            self.motor_indices) if action_dim is None else action_dim

        # set init pos
        # self._reset_robot_joint_pos(self.n_robot_joints,
        #                             self.init_robot_joint_pos)
        self._set_robot_joint_pos(self.motor_indices,
                                  self.init_robot_joint_pos)

        # if init_endeffector_pos is not None and \
        #         init_endeffector_orn is not None:
        #     self.init_robot_joint_pos = p.calculateInverseKinematics(
        #         body_id, self.endeffector_index, init_endeffector_pos,
        #         init_endeffector_orn)

        #     self._reset_robot_joint_pos(self.n_robot_joints,
        #                                 self.init_robot_joint_pos)

        # a = p.getLinkState(body_id, self.endeffector_index)

    def _load_body(self, full_path, base_pos=(0, 0, 0), base_ori=(0, 0, 0, 1)):
        filepath = os.path.basename(full_path)
        if "sdf" in filepath:
            id = p.loadSDF(full_path)[0]
        elif "urdf" in filepath:
            # objects = p.loadURDF(full_path)
            # objects = p.loadURDF(os.path.join(os.getcwd(),self.robotUrdfPath), self.robotStartPos, self.robotStartOrn,
            #                     flags=p.URDF_USE_INERTIA_FROM_FILE)
            id = p.loadURDF(full_path,
                            flags=p.URDF_USE_INERTIA_FROM_FILE)

        p.resetBasePositionAndOrientation(id, base_pos, base_ori)

        return id

    def _make_robot(self, body_path, base_pos=(0, 0, 0),
                    base_orn=(0, 0, 0, 1)):
        # load body
        body_id = self._load_body(body_path, base_pos, base_orn)
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

    def _set_robot_joint_pos(self, motor_indices, joint_positions):
        # set init position
        # for i in range(n_joints):
        #     # print(i)
        #     p.setJointMotorControl2(bodyUniqueId=self.body_id,
        #                             jointIndex=i,
        #                             controlMode=p.POSITION_CONTROL,
        #                             targetPosition=joint_positions[i],
        #                             targetVelocity=0,
        #                             force=self.max_force,
        #                             maxVelocity=self.max_velocity,
        #                             positionGain=0.3,
        #                             velocityGain=1)
        l = len(motor_indices)
        p.setJointMotorControlArray(self.body_id,
                                    jointIndices=motor_indices,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=joint_positions,
                                    targetVelocities=[0]*l,
                                    forces=[self.max_force]*l,
                                    positionGains=[0.3]*l,
                                    velocityGains=[1]*l)

    def reset(self):
        self._reset_robot_joint_pos(self.n_robot_joints,
                                    self.init_robot_joint_pos)

    def get_action_dim(self):
        return self._action_dim

    # def get_observation_dim(self):
    #     return len(self.get_bservation())

    def get_observation_dim(self):
        raise NotImplementedError

    def apply_action(self, action):
        raise NotImplementedError
