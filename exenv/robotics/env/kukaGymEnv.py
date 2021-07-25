from ..robots.kuka import KukaInverseKinematics
from ..tasks.Tasks import LiftBlock
import pybullet_data


class LiftBlockWithKukaInverseKinematics(LiftBlock):

    def __init__(self,
                 urdf_root=pybullet_data.getDataPath(),
                 action_repeat=1,
                 max_steps=5000):
        robot = KukaInverseKinematics()
        super().__init__(robot, urdf_root, action_repeat, max_steps)
