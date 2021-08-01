from ..robots.kuka import KukaInverseKinematics
from ..tasks.Tasks import LiftBlock
import pybullet_data


class LiftBlockWithKukaInverseKinematics(LiftBlock):

    def __init__(self,
                 urdf_root=pybullet_data.getDataPath(),
                 action_repeat=1,
                 time_step=1./240.,
                 max_steps=5000,
                 ):
        self._connect_with_pybullet(time_step)
        robot = KukaInverseKinematics()
        super().__init__(robot, urdf_root, action_repeat, time_step, max_steps)
