import numpy as np
import pybullet as p


class GetTargetPosAndOrn:
    def __init__(self, target_id):
        self._target_id = target_id

    def __call__(self, obs, reward, done, info):
        pos, orn = p.getBasePositionAndOrientation(self._target_id)
        euler = p.getEulerFromQuaternion(orn)

        info["target_pos"] = pos
        info["target_orn"] = orn
        info["target_euler"] = euler

        return obs, reward, done, info


class CalEndEffectorRelativePosition:
    def __call__(self, obs, reward, done, info):
        rel_pos \
            = np.array(info["target_pos"]) - np.array(info["endeffector_pos"])

        info["rel_pos"] = rel_pos

        return obs, reward, done, info


class CalEndEffectorRelativeEulerAngle:
    def __call__(self, obs, reward, done, info):
        rel_angle \
            = np.array(info["target_euler"]) \
            - np.array(info["endeffector_euler"])

        rel_angle = rel_angle % (2 * np.pi)
        rel_angle[2] -= 0.5 * np.pi       # 元々のズレを引いて基準角度を0にする

        info["rel_angle"] = rel_angle

        return obs, reward, done, info


class NormalizeRelativeValue:
    def __init__(self, coeff_pos=100, coeff_angle=10):
        self.coeff_pos = coeff_pos
        self.coeff_angle = coeff_angle

    def __call__(self, obs, reward, done, info):
        info["norm_rel_pos"] = info["rel_pos"] * self.coeff_pos
        info["norm_rel_angle"] = info["rel_angle"] * self.coeff_angle

        return obs, reward, done, info


class FormatKukaInverseKinematics:
    def __call__(self, obs, reward, done, info):
        obs = []
        obs.extend(info["norm_rel_pos"].tolist()[:2])        # xy
        obs.extend(info["norm_rel_angle"].tolist()[2:])      # yaw

        return obs, reward, done, info


class LiftBlockReward:
    def __call__(self, obs, reward, done, info):
        # reward = -1000
        block_pos = info['target_pos']

        diff_xy = info['rel_pos']
        # xy_distance = np.linalg.norm(diff_xy) / X_COEFF
        xy_distance = np.linalg.norm(diff_xy)
        # a_diff = obs[2] / A_COEFF
        # a_diff = obs[0] / A_COEFF

        if (block_pos[2] > 0.1):
            reward = reward + 10000
            print("successfully grasped a block!!!")
        else:
            reward1 = 3 - (xy_distance * 10)
            # reward2 = 2 - abs(a_diff)
            # reward = reward1 + reward2
            reward = reward1

        return obs, reward, done, info
