from pkg_resources import parse_version
import pybullet_data
import random
from . import kuka
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

#from pybullet_envs.bullet import kuka

#largeValObservation = 100
largeValObservation = 200

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

X_COEFF = 100
Y_COEFF = 100
A_COEFF = 10


class KukaGymEnv(gym.Env):
    metadata = {'render.modes': [
        'human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 urdfRoot=pybullet_data.getDataPath(),
                 actionRepeat=1,
                 isEnableSelfCollision=True,
                 renders=False,
                 isDiscrete=False,
                 maxSteps=5000):
        #print("KukaGymEnv __init__")
        self._isDiscrete = isDiscrete
        self._timeStep = 1. / 240.
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._maxSteps = maxSteps
        self.terminated = 0
        #self._cam_dist = 1.3
        self._cam_dist = 1.5
        #self._cam_yaw = 180
        self._cam_yaw = 120
        #self._cam_pitch = -40
        self._cam_pitch = -20

        self._p = p
        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if (cid < 0):
                cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
        else:
            p.connect(p.DIRECT)
        #timinglog = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "kukaTimings.json")

        self.action_dim = 1
        #self.action_dim = 2
        #sel.faction_dim = 3

        self.seed()
        self.reset()
        observationDim = len(self.getExtendedObservation())
        # print("observationDim")
        # print(observationDim)

        observation_high = np.array([largeValObservation] * observationDim)
        if (self._isDiscrete):
            self.action_space = spaces.Discrete(7)
        else:
            self._action_bound = 1
            action_high = np.array([self._action_bound] * self.action_dim)
            self.action_space = spaces.Box(-action_high, action_high)
        self.observation_space = spaces.Box(-observation_high,
                                            observation_high)
        self.viewer = None

    def reset(self):
        #print("KukaGymEnv _reset")
        self._attempted_grasp = False
        self.terminated = 0
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        #p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])

        # p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, -.820000,
        #           0.000000, 0.000000, 0.0, 1.0)
        p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), [0.5, 0.0, -0.62],
                   [0.000000, 0.000000, 0.0, 1.0])

        #xpos = 0.55
        #ypos = 0
        #ang = -np.pi * 0.5
        #xpos = 0.55 + 0.06 * (2 * random.random() -1)
        #ypos = 0 + 0.1 * (2 * random.random() -1)
        #xpos = 0.55 + 0.12 * random.random()
        #xpos = 0.55 - 0.12 * random.random()
        #ypos = 0 + 0.2 * random.random()

        if self.action_dim == 1:
            xpos = 0.55 + 0.12 * (2 * random.random() - 1)
            ypos = 0
            ang = -np.pi * 0.5
        elif self.action_dim == 2:
            xpos = 0.55 + 0.12 * (2 * random.random() - 1)
            ypos = 0 + 0.2 * (2 * random.random() - 1)
            ang = -np.pi * 0.5
        else:
            xpos = 0.55 + 0.12 * (2 * random.random() - 1)
            ypos = 0 + 0.2 * (2 * random.random() - 1)
            ang = -np.pi * 0.5 + np.pi * random.random()

        self.init_blockPos = (xpos, ypos, ang)

        # default
        #xpos = 0.55 + 0.12 * random.random()
        #ypos = 0 + 0.2 * random.random()
        #ang = 3.14 * 0.5 + 3.1415925438 * random.random()
        orn = p.getQuaternionFromEuler([0, 0, ang])
        self.blockUid = p.loadURDF(os.path.join(self._urdfRoot, "block.urdf"), xpos, ypos, 0.02,
                                   orn[0], orn[1], orn[2], orn[3])
        # self.blockUid = p.loadURDF(os.path.join(self._urdfRoot, "block.urdf"), xpos, ypos, -0.15,
        #                           orn[0], orn[1], orn[2], orn[3])

        p.setGravity(0, 0, -10)
        self._kuka = kuka.Kuka(
            urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        self._envStepCounter = 0
        p.stepSimulation()
        self._observation = self.getExtendedObservation()
        return np.array(self._observation)

    def __del__(self):
        p.disconnect()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def getExtendedObservation(self):
        kuka_obs = self._kuka.getObservation()
        '''
    gripperState = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaGripperIndex)
    gripperPos = gripperState[0]
    gripperOrn = gripperState[1]
    blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)

    blockEuler = p.getEulerFromQuaternion(blockOrn)

    invGripperPos, invGripperOrn = p.invertTransform(gripperPos, gripperOrn)
    gripperMat = p.getMatrixFromQuaternion(gripperOrn)
    dir0 = [gripperMat[0], gripperMat[3], gripperMat[6]]
    dir1 = [gripperMat[1], gripperMat[4], gripperMat[7]]
    dir2 = [gripperMat[2], gripperMat[5], gripperMat[8]]

    gripperEul = p.getEulerFromQuaternion(gripperOrn)
    #print("gripperEul")
    #print(gripperEul)
    blockPosInGripper, blockOrnInGripper = p.multiplyTransforms(invGripperPos, invGripperOrn,
                                                                blockPos, blockOrn)
    projectedBlockPos2D = [blockPosInGripper[0], blockPosInGripper[1]]
    blockEulerInGripper = p.getEulerFromQuaternion(blockOrnInGripper)
    #print("projectedBlockPos2D")
    #print(projectedBlockPos2D)
    #print("blockEulerInGripper")
    #print(blockEulerInGripper)

    #we return the relative x,y position and euler angle of block in gripper space
    blockInGripperPosXYEulZ = [blockPosInGripper[0], blockPosInGripper[1], blockEulerInGripper[2]]

    #p.addUserDebugLine(gripperPos,[gripperPos[0]+dir0[0],gripperPos[1]+dir0[1],gripperPos[2]+dir0[2]],[1,0,0],lifeTime=1)
    #p.addUserDebugLine(gripperPos,[gripperPos[0]+dir1[0],gripperPos[1]+dir1[1],gripperPos[2]+dir1[2]],[0,1,0],lifeTime=1)
    #p.addUserDebugLine(gripperPos,[gripperPos[0]+dir2[0],gripperPos[1]+dir2[1],gripperPos[2]+dir2[2]],[0,0,1],lifeTime=1)

    #self._observation.extend(list(blockInGripperPosXYEulZ))
    #self._observation = self._observation[:2]
    #self._observation.extend(blockPos[:2])
    '''
        block_pos, block_orn = p.getBasePositionAndOrientation(self.blockUid)
        block_euler = p.getEulerFromQuaternion(block_orn)
        diff_xy = np.array(block_pos[:2]) - np.array(kuka_obs[:2])
        self._observation = (diff_xy*X_COEFF).tolist()
        diff_yaw = block_euler[2] - kuka_obs[5]
        diff_yaw -= 0.5 * np.pi       # 元々のズレを引いて基準角度を0にする
        diff_yaw -= 2*np.pi*math.floor(diff_yaw/(2*np.pi))
        if diff_yaw > np.pi:
            diff_yaw -= 2*np.pi
        self._observation.append(diff_yaw*A_COEFF)
        #self._observation = diff.tolist()

        #diff = np.array(blockPos[0]) - np.array(self._observation[0])
        # self._observation = [diff*100]    # NNの関係で値が小さすぎるとなぜか線形にしかならないため、スケールアップしている
        #self._observation = [diff]

        # self._observation.extend(blockEuler)
        return self._observation

    def step(self, action):
        if (self._isDiscrete):
            dv = 0.005
            dx = [0, -dv, dv, 0, 0, 0, 0][action]
            dy = [0, 0, 0, -dv, dv, 0, 0][action]
            da = [0, 0, 0, 0, 0, -0.05, 0.05][action]
            f = 0.3
            realAction = [dx, dy, -0.002, da, f]
        else:
            #print("action[0]=", str(action[0]))
            dv = 0.005
            if self.action_dim == 1:
                dx = action[0] * dv
                dy = 0
                da = 0
            elif self.action_dim == 2:
                dx = action[0] * dv
                dy = action[1] * dv
                da = 0
            else:
                dx = action[0] * dv
                dy = action[1] * dv
                da = action[2] * 0.05

            f = 0.3
            realAction = [dx, dy, -0.002, da, f]
        return self.step2(realAction)

    def step2(self, action):
        for i in range(self._actionRepeat):
            self._kuka.applyAction(action)
            p.stepSimulation()
            if self._termination():
                break
            self._envStepCounter += 1
        if self._renders:
            time.sleep(self._timeStep)

        # If we are close to the bin, attempt grasp.
        state = p.getLinkState(
            self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
        end_effector_pos = state[4]
        # if end_effector_pos[2] <= 0.0925:
        if end_effector_pos[2] <= 0.28:
            finger_angle = 0.3
            for _ in range(500):
                grasp_action = [0, 0, 0, 0, finger_angle]
                self._kuka.applyAction(grasp_action)
                p.stepSimulation()
                if self._renders:
                    time.sleep(self._timeStep)
                finger_angle -= 0.3 / 100.
                if finger_angle < 0:
                    finger_angle = 0

            for _ in range(250):
                grasp_action = [0, 0, 0.002, 0, finger_angle]
                self._kuka.applyAction(grasp_action)
                p.stepSimulation()
                if self._renders:
                    time.sleep(self._timeStep)
                #finger_angle -= 0.3 / 100.
                # if finger_angle < 0:
                #  finger_angle = 0

            self._attempted_grasp = True

        self._observation = self.getExtendedObservation()

        # print("self._envStepCounter")
        # print(self._envStepCounter)

        done = self._termination()
        npaction = np.array([
            action[3]
        ])  # only penalize rotation until learning works well [action[0],action[1],action[3]])
        actionCost = np.linalg.norm(npaction) * 10.
        # print("actionCost")
        # print(actionCost)
        reward = self._reward() - actionCost
        # print("reward")
        # print(reward)

        #print("len=%r" % len(self._observation))

        return np.array(self._observation), reward, done, {}

    def render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])

        base_pos, orn = self._p.getBasePositionAndOrientation(
            self._kuka.kukaUid)
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                                distance=self._cam_dist,
                                                                yaw=self._cam_yaw,
                                                                pitch=self._cam_pitch,
                                                                roll=0,
                                                                upAxisIndex=2)
        proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                         aspect=float(
                                                             RENDER_WIDTH) / RENDER_HEIGHT,
                                                         nearVal=0.1,
                                                         farVal=100.0)
        (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                                  height=RENDER_HEIGHT,
                                                  viewMatrix=view_matrix,
                                                  projectionMatrix=proj_matrix,
                                                  renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
        # renderer=self._p.ER_TINY_RENDERER)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _termination(self):
        return self._attempted_grasp or self._envStepCounter >= self._maxSteps
        '''
    #print (self._kuka.endEffectorPos[2])
    state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
    actualEndEffectorPos = state[0]

    #print("self._envStepCounter")
    #print(self._envStepCounter)
    if (self.terminated or self._envStepCounter > self._maxSteps):
      self._observation = self.getExtendedObservation()
      return True
    maxDist = 0.005
    closestPoints = p.getClosestPoints(self._kuka.trayUid, self._kuka.kukaUid, maxDist)

    if (len(closestPoints)):  #(actualEndEffectorPos[2] <= -0.43):
      self.terminated = 1

      #print("terminating, closing gripper, attempting grasp")
      #start grasp and terminate
      fingerAngle = 0.3
      for i in range(100):
        graspAction = [0, 0, 0.0001, 0, fingerAngle]
        self._kuka.applyAction(graspAction)
        p.stepSimulation()
        fingerAngle = fingerAngle - (0.3 / 100.)
        if (fingerAngle < 0):
          fingerAngle = 0

      for i in range(1000):
        graspAction = [0, 0, 0.001, 0, fingerAngle]
        self._kuka.applyAction(graspAction)
        p.stepSimulation()
        blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
        if (blockPos[2] > 0.23):
          #print("BLOCKPOS!")
          #print(blockPos[2])
          break
        state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
        actualEndEffectorPos = state[0]
        if (actualEndEffectorPos[2] > 0.5):
          break

      self._observation = self.getExtendedObservation()
      return True
    return False
    '''

    def _reward(self):

        # rewards is height of target object
        blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
        # closestPoints = p.getClosestPoints(self.blockUid, self._kuka.kukaUid, 1000, -1,
        #                                   self._kuka.kukaEndEffectorIndex)

        #reward = -1000
        reward = 0

        '''
    numPt = len(closestPoints)

    # 平面方向の距離のみで評価
    gripper_state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaGripperIndex)
    gripper_xy = np.array(gripper_state[4])[:2]
    block_xy = np.array(blockPos)[:2]
    xy_distance = np.linalg.norm(block_xy - gripper_xy)
    x_distance = abs(block_xy[0] - gripper_xy[0])
    '''

        obs = self.getExtendedObservation()
        xy_distance = np.linalg.norm(obs[:2]) / X_COEFF
        #a_diff = obs[2] / A_COEFF
        a_diff = obs[0] / A_COEFF

        # print(numPt)
        # if (numPt > 0):
        #  #print("reward:")
        #  reward = -closestPoints[0][8] * 10
        # if (blockPos[2] > -0.07):
        if (blockPos[2] > 0.1):
            reward = reward + 10000
            print("successfully grasped a block!!!")
            # print("self._envStepCounter")
            # print(self._envStepCounter)
            # print("self._envStepCounter")
            # print(self._envStepCounter)
            # print("reward")
            # print(reward)
        else:
            #reward1 = 1.5 - (x_distance * 10)
            reward1 = 3 - (xy_distance * 10)
            reward2 = 2 - abs(a_diff)
            reward = reward1 + reward2

        # print("reward")
        # print(reward)
        return reward

    def get_blockPos(self):
        blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
        return blockPos

    def get_endeffectorPos(self):
        state = p.getLinkState(
            self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
        #pos = state[0]
        #orn = state[1]
        pos = state[4]
        orn = state[5]

        return pos

    if parse_version(gym.__version__) < parse_version('0.9.6'):
        _render = render
        _reset = reset
        _seed = seed
        _step = step
