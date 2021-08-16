from ..robots.kuka import KukaInverseKinematics
from ..tasks.Tasks import LiftBlock
import pybullet_data


class LiftBlockWithKukaInverseKinematics(LiftBlock):

    def __init__(self,
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
        super().__init__(robot, urdf_root, action_repeat, time_step, max_steps)
