from numpy.core.shape_base import block
from .action_filters import FormatInverseKinematics
from .observation_filters import (GetTargetPosAndOrn,
                                  CalEndEffectorRelativePosition,
                                  CalEndEffectorRelativeEulerAngle,
                                  NormalizeRelativeValue,
                                  FormatKukaInverseKinematics,
                                  LiftBlockReward)
from ..meshes import get_mesh_dir_path
from pkg_resources import parse_version
import pybullet_data
import random
import copy
import pybullet as p
import time
import numpy as np
from gym.utils import seeding
from gym import spaces
import gym
import math
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)


# from pybullet_envs.bullet import kuka

# largeValObservation = 100
largeValObservation = 200

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

X_COEFF = 100
Y_COEFF = 100
A_COEFF = 10


class BaseTask(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': 50}

    def __init__(self, robot, action_filters, observation_filters,
                 action_repeat=1, time_step=1./240., max_steps=5000,
                 render=False,):
        self._time_step = time_step
        self._robot = robot
        self._action_repeat = action_repeat
        self._max_steps = max_steps
        self._render = render
        self._action_filters = action_filters
        self._observation_filters = observation_filters
        self._env_step_counter = 0

        self._robot.reset()

        self.action_dim = self._robot.get_action_dim()
        obs, _, _, _ = self._get_observation()
        self.observation_dim = len(obs)

    def _connect_with_pybullet(self, time_step, render):
        if render:
            cid = p.connect(p.SHARED_MEMORY)
            if (cid < 0):
                cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
        else:
            p.connect(p.DIRECT)
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(time_step)
        p.stepSimulation()

    def reset(self):
        # reset pybullet
        # p.resetSimulation()

        # reset variable
        print("env_step_counter", self._env_step_counter)
        self._env_step_counter = 0

        # reset robot
        self._robot.reset()
        obs, _, _, _ = self._get_observation()

        return np.array(obs)

    def step(self, action):
        self._env_step_counter += 1

        for action_filter in self._action_filters:
            action = action_filter(action)

        for _ in range(self._action_repeat):
            self._robot.apply_action(action)
            if self._termination():
                break
            self._env_step_counter += 1

        obs, reward, done, info = self._get_observation(self._specific_action)

        return obs, reward, done, info

    def _specific_action(self, info):
        pass

    def _get_observation(self, specific_action=None):
        info = self._robot.get_state()
        obs, reward, done = [], 0, False

        # task-specific actions
        if specific_action is not None:
            self._specific_action(info)

        for observation_filter in self._observation_filters:
            obs, reward, done, info \
                = observation_filter(obs, reward, done, info)
        done = self._termination() or done

        return obs, reward, done, info


class LiftBlock(BaseTask):
    def __init__(self,
                 robot,
                 block_pos_offsets=(0.15, 0.15),
                 urdf_root=pybullet_data.getDataPath(),
                 action_repeat=1,
                 time_step=1./240.,
                 max_steps=5000,
                 ):
        self._urdf_root = urdf_root
        # self._cam_dist = 1.3
        self._cam_dist = 1.5
        # self._cam_yaw = 180
        self._cam_yaw = 120
        # self._cam_pitch = -40
        self._cam_pitch = -20
        self._lim_z = robot.gripper_length + 0.02
        #self._lim_z = 0.28
        #self._lim_z = 0.1
        self.block_pos_offsets = block_pos_offsets

        self.seed()

        # load table
        p.loadURDF(os.path.join(self._urdf_root, "table/table.urdf"),
                   [0.5, 0.0, -0.62],
                   [0.000000, 0.000000, 0.0, 1.0])

        # set block
        block_base_pos \
            = np.array(robot.init_endeffector_pos)[:2] \
            + np.array(self.block_pos_offsets)
        self.block_id, self.init_block_pos, self.init_block_orn \
            = self.load_block(block_base_pos)
        self._attempted_grasp = False

        # set filters
        action_filters = (FormatInverseKinematics(robot.get_action_dim()),)
        observation_filters = (GetTargetPosAndOrn(self.block_id),
                               CalEndEffectorRelativePosition(),
                               CalEndEffectorRelativeEulerAngle(),
                               NormalizeRelativeValue(),
                               FormatKukaInverseKinematics(),
                               LiftBlockReward(),
                               )

        super().__init__(robot, action_filters, observation_filters,
                         action_repeat, time_step, max_steps)

        observation_high \
            = np.array([100000, 100000, 100000])
        action_high = np.array([1] * self.action_dim)
        self.action_space = spaces.Box(-action_high, action_high,
                                       shape=(self.action_dim,))
        self.observation_space = spaces.Box(-observation_high,
                                            observation_high,
                                            shape=(self.observation_dim,))

    def reset(self):
        # reset variable
        self._attempted_grasp = False

        # reset block
        block_base_pos \
            = np.array(self._robot.init_endeffector_pos)[:2] \
            + np.array(self.block_pos_offsets)
        self.init_block_pos, self.init_block_orn \
            = self.reset_block(block_base_pos)

        # reset super class
        obs = super().reset()

        return obs

    def cal_block_pos_and_orn(self, base_pos):
        # if self.action_dim == 1:
        #     xpos = 0.55 + 0.12 * (2 * random.random() - 1)
        #     ypos = 0
        #     ang = -np.pi * 0.5
        # elif self.action_dim == 2:
        #     xpos = 0.55 + 0.12 * (2 * random.random() - 1)
        #     ypos = 0 + 0.2 * (2 * random.random() - 1)
        #     ang = -np.pi * 0.5
        # else:
        #     xpos = 0.55 + 0.12 * (2 * random.random() - 1)
        #     ypos = 0 + 0.2 * (2 * random.random() - 1)
        #     ang = -np.pi * 0.5 + np.pi * random.random()

        pos = list(base_pos)
        #pos[2] = 0.02
        pos += [0.1]

        ang = -np.pi * 0.5

        orn = p.getQuaternionFromEuler([0, 0, ang])

        return tuple(pos), orn

    def load_block(self, base_pos):
        pos, orn = self.cal_block_pos_and_orn(base_pos)

        block_id = p.loadURDF(os.path.join(get_mesh_dir_path(),
                                           "objects/cube_small.urdf"),
                              pos[0], pos[1], pos[2],
                              orn[0], orn[1], orn[2], orn[3])
        # block_id = p.loadURDF(os.path.join(self._urdf_root, "block.urdf"),
        #                       pos[0], pos[1], pos[2],
        #                       orn[0], orn[1], orn[2], orn[3])

        return block_id, pos, orn

    def reset_block(self, base_pos):
        pos, orn = self.cal_block_pos_and_orn(base_pos)

        p.resetBasePositionAndOrientation(self.block_id, pos, orn)

        return pos, orn

    def __del__(self):
        p.disconnect()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _termination(self):
        termination = self._attempted_grasp or \
            self._env_step_counter >= self._max_steps
        return termination

    def _specific_action(self, info):
        # If we are close to the bin, attempt grasp.
        end_effector_pos = info["endeffector_pos"]

        self._attempted_grasp \
            = self._robot._grasping_action(end_effector_pos[2], self._lim_z)

    def render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])

        base_pos, orn = p.getBasePositionAndOrientation(
            self._robot.body_id)
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                          distance=self._cam_dist,
                                                          yaw=self._cam_yaw,
                                                          pitch=self._cam_pitch,
                                                          roll=0,
                                                          upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                   aspect=float(
                                                       RENDER_WIDTH) / RENDER_HEIGHT,
                                                   nearVal=0.1,
                                                   farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=RENDER_WIDTH,
                                            height=RENDER_HEIGHT,
                                            viewMatrix=view_matrix,
                                            projectionMatrix=proj_matrix,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)
        # renderer=self._p.ER_TINY_RENDERER)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array
