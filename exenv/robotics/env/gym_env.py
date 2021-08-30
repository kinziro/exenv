from ..robots.kuka import KukaInverseKinematics
from ..robots.ur5 import Ur5InverseKinematics
from ..tasks.Tasks import LiftBlock
import pybullet_data

from ray.rllib.env.env_context import EnvContext
import gym
from gym.spaces import Discrete, Box
import numpy as np
import random


# class LiftBlockWithKukaInverseKinematics(LiftBlock):
#
#     def __init__(self, config):
#         '''
#         Environment in whick a robot lifts a block.
#
#         Args:
#             urdf_root (str): data path in pybullet repository
#             action_repeat (int, optional): number of repeats of the action
#             time_step (float, optional): time step of the simulation
#             max_steps (int, optional): max step of the simulation
#             render (bool, optional): whether show GUI
#         '''
#         block_pos_offsets = config.get("block_pos_offsets", (0.15, 0.15))
#         urdf_root = config.get("urdf_root", pybullet_data.getDataPath())
#         action_repeat = config.get("action_repeat", 1)
#         time_step = config.get("time_step", 1./240.)
#         max_steps = config.get("max_steps", 5000)
#         render = config.get("render", False)
#
#         self._connect_with_pybullet(time_step, render)
#         robot = KukaInverseKinematics(action_dim=2)
#         super().__init__(robot, block_pos_offsets, urdf_root,
#                          action_repeat, time_step, max_steps)


class LiftBlockWithKukaInverseKinematics(LiftBlock):

    def __init__(self,
                 block_pos_offsets=(0.15, 0.15),
                 urdf_root=pybullet_data.getDataPath(),
                 action_repeat=1,
                 time_step=1./240.,
                 max_steps=5000,
                 render=False,
                 ):
        '''
        Environment in whick a robot lifts a block.

        Args:
            urdf_root (str): data path in pybullet repository
            action_repeat (int, optional): number of repeats of the action
            time_step (float, optional): time step of the simulation
            max_steps (int, optional): max step of the simulation
            render (bool, optional): whether show GUI
        '''
        self._connect_with_pybullet(time_step, render)
        robot = KukaInverseKinematics(action_dim=2)
        super().__init__(robot, block_pos_offsets, urdf_root,
                         action_repeat, time_step, max_steps)


class LiftBlockWithUr5InverseKinematics(LiftBlock):

    def __init__(self,
                 block_pos_offsets=(0.15, 0.15),
                 urdf_root=pybullet_data.getDataPath(),
                 action_repeat=1,
                 time_step=1./240.,
                 max_steps=5000,
                 render=False,
                 ):
        '''
        Environment in whick a robot lifts a block.

        Args:
            urdf_root (str): data path in pybullet repository
            action_repeat (int, optional): number of repeats of the action
            time_step (float, optional): time step of the simulation
            max_steps (int, optional): max step of the simulation
            render (bool, optional): whether show GUI
        '''
        self._connect_with_pybullet(time_step, render)
        robot = Ur5InverseKinematics(action_dim=2)
        super().__init__(robot, block_pos_offsets, urdf_root,
                         action_repeat, time_step, max_steps)
