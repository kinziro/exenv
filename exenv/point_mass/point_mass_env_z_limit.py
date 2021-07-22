import math
import gym
from gym import spaces
import numpy as np
import random
import copy

class PointMassEnv(gym.Env):
    def __init__(self, task_id, task_len, goal_range=0.5, reward_range=1, terminal_timestep=28, init_random=False, val=False):
        self.task_id = task_id
        self.task_len = task_len
        self.goal_range = goal_range
        self.line_width = goal_range
        self.reward_range = reward_range
        self.terminal_timestep = terminal_timestep
        self.init_random = init_random
        self.val = val

        obs = self.reset()
        self.action_high = 1
        self.observation_high = np.max(np.abs(self.goal_max)) * 3
        self.action_space = spaces.Box(-self.action_high, self.action_high, shape=(2,))
        self.observation_space = spaces.Box(-self.observation_high, self.observation_high, shape=(len(obs),))

    def task_id_to_int(self, task_id):
        task_int = np.argmax(np.array(task_id[:self.task_len]).reshape(1, -1), axis=1)[0]
        env_int = np.argmax(np.array(task_id[self.task_len:]).reshape(1, -1), axis=1)[0]

        return task_int, env_int
    
    def set_goal(self, task_id):
        self.task_id = task_id
        self.task_int, self.env_int = self.task_id_to_int(task_id)

        import math
        theta = 2 * np.pi * random.random()

        if self.task_int in [0]:
            #self.goal = [2.5, 2.5]
            #self.goal = [4.5, 4.5]
            self.goal = [4 * math.cos(theta), 4 * math.sin(theta)]
            #self.goal = [6, 4]
            #self.goal = [4, 3]
            self.goal_max = [4, 4]

        elif self.task_int in [1]:
            self.goal = [-2.5, 2.5]
        elif self.task_int in [2]:
            self.goal = [-2.5, -2.5]
        elif self.task_int in [3]:
            self.goal = [2.5, -2.5]
        else:
            self.goal = [2.5, 2.5]
        
        if self.env_int in [0]:
            self.a, self.b = 1, 1
        elif self.env_int in [1]:
            self.a, self.b = 1, 0.25
        elif self.env_int in [2]:
            self.a, self.b = 0.25, 1
        elif self.env_int in [3]:
            self.a, self.b = 1, 1
        elif self.env_int in [4]:
            self.a, self.b = 1, 0.25

    def reset(self, task_id=None):
        if task_id is not None:
            self.set_goal(task_id)
        else:
            self.set_goal(self.task_id)

        if self.init_random:
            x = self.goal[0] * random.randint(0, 1) * 2
            y = self.goal[1] * random.randint(0, 1) * 2
        else:
            x = 0
            y = 0

        self.obs_position = np.array([x, y], dtype='float32')
        self.observation = self.get_obs()
        self.init_pos = copy.deepcopy(self.obs_position)
        self.terminal = False
        self.time_step = 0

        return self.observation

    def cal_distance_from_line(self, pos):
        # ラインを計算
        x1, y1 = self.init_pos
        x2, y2 = self.goal
        a = -(y1 - y2)/((x1 - x2))
        b = -y1 + x1 * a

        # 距離の計算
        numer = abs(a*pos[0] + pos[1] + b)
        denom = math.sqrt(pow(a, 2) + pow(1, 2))

        return numer / denom
    
    def cal_reward(self, pos):

        diff = self.goal - pos
        goal_dist = np.linalg.norm(diff)
        line_dist = self.cal_distance_from_line(pos)
        
        #goal_reward = max(self.reward_range - goal_dist, 0)
        #goal_reward = -10
        step_reward = -10
        line_reward = (self.line_width - line_dist) / self.line_width * 30
        if line_reward < 0:
            line_reward *= 2
        #line_reward = max(self.line_width - line_dist, 0) / self.line_width * 30
        #reward = goal_reward + (self.reward_range/self.line_width) * line_reward
        reward = line_reward + step_reward

        self.terminal, info = self.check_terminal(pos, goal_dist, line_dist)

        if self.terminal:
            if goal_dist <= self.goal_range:
                reward = 1000
                #reward = 70
            #elif line_dist > self.line_width:
            #    #reward = -50
            #    reward = 0
            else:
                #reward = -1000
                reward = 0

        return reward, self.terminal, info
    
    def check_terminal(self, pos, goal_dist, line_dist):
        terminal = False
        info = ""
        if self.time_step >= self.terminal_timestep:
            terminal = True
            info = "time_up"
        elif abs(pos[0]) > self.observation_high or abs(pos[1]) > self.observation_high:
            terminal = True
            info = "out_of_range"
        elif line_dist > self.line_width and self.val:
            terminal = True
            info = "out_of_line_width"
        else:
            if goal_dist <= self.goal_range:
                terminal = True
 
        return terminal, info
    
    def get_obs(self):
        #obs = np.hstack([self.obs_position, self.goal])
        obs = np.array(self.goal) - np.array(self.obs_position)
        #obs = self.obs_position
        return obs.tolist()
    
    def transition(self, action):
        coeff_action = [action[0] * self.a, action[1] * self.b]
        self.obs_position = (self.obs_position + coeff_action)
        #self.real_position[0] = self.obs_position[0] * self.obs_x_coeff
        #self.real_position[1] = self.obs_position[1] * self.obs_y_coeff
    
    def transition_filter_act(self, action):
        tra_action = action
 
        return np.array(tra_action)


    def step(self, action):
        self.time_step += 1

        regular_action = np.array(action, dtype='float32')
        regular_action = np.where(regular_action < self.action_high, regular_action, self.action_high)
        regular_action = np.where(regular_action > -self.action_high, regular_action, -self.action_high)
        #regular_action = self.transition_filter_act(regular_action)

        self.transition(regular_action)
        self.observation = self.get_obs()
        self.observation = np.where(self.observation < self.observation_high, self.observation, self.observation_high)
        self.observation = np.where(self.observation > -self.observation_high, self.observation, -self.observation_high)

        reward, done, term_info = self.cal_reward(self.obs_position)

        info = {}
        #info['real_position'] = self.real_position
        #info['obs_position'] = self.obs_position
        #info['goal'] = self.goal
        #info['terminal'] = term_info
        info['position'] = self.obs_position
        info['goal'] = self.goal
        info['terminal'] = term_info

        return self.observation, reward, done, info
    
    def get_info(self):
        info = {}
        #info['real_position'] = self.real_position
        info['position'] = self.obs_position
        info['goal'] = self.goal

        return info
    

if __name__=="__main__":
    env = PointMassEnv(task_id=[1, 0, 0, 0, 1, 0], task_len=4)
    obs = env.reset()
    dis = env.cal_distance_from_line((10, -10))
    print(dis)
    #print(obs)

    #done = False
    #for _ in range(20):
    #    obs, reward, done , _ = env.step([0.9, 0])
    #    print(obs, reward, done)
    #    if done:
    #        break
    
    #print('end')