import math
import gym
from gym import spaces
import numpy as np
import random
import copy

class PointMassEnv(gym.Env):
    def __init__(self, task_id, task_len, goal_range=0.5, reward_range=1, terminal_timestep=28, init_random=False):
        self.task_id = task_id
        self.task_len = task_len
        self.goal_range = goal_range
        self.reward_range = reward_range
        self.terminal_timestep = terminal_timestep
        self.init_random = init_random

        obs = self.reset()
        self.action_high = 1
        self.observation_high = np.max(np.abs(self.goal)) * 3
        self.action_space = spaces.Box(-self.action_high, self.action_high, shape=(2,))
        self.observation_space = spaces.Box(-self.observation_high, self.observation_high, shape=(len(obs),))
    
    def task_id_to_int(self, task_id):
        task_int = np.argmax(np.array(task_id[:self.task_len]).reshape(1, -1), axis=1)[0]
        env_int = np.argmax(np.array(task_id[self.task_len:]).reshape(1, -1), axis=1)[0]

        return task_int, env_int
    
    def set_init(self, task_id):
        self.task_id = task_id
        self.task_int, self.env_int = self.task_id_to_int(task_id)

        if self.task_int in [0]:
            self.goal = [3.5, 3.5]
            self.wall = Line((1.5, 2), (10, 2))
        elif self.task_int in [1]:
            self.goal = [-3.5, 3.5]
            self.wall = Line((1.5, 2), (10, 2))
        elif self.task_int in [2]:
            self.goal = [3.5, 3.5]
            self.wall = Line((-10, 2), (2.5, 2))
        elif self.task_int in [3]:
            self.goal = [-3.5, 3.5]
            self.wall = Line((-10, 2), (2.5, 2))
        else:
            self.goal = [3.5, 3.5]
            self.wall = None
        
        if self.env_int in [0]:
            self.mean = np.array([0, 0])
            self.std = np.array([[0.5, 0], [0, 0.5]])
            #self.std = np.array([[0.001, 0], [0, 0.001]])
        elif self.env_int in [1]:
            self.mean = np.array([1, 0])
            self.std = np.array([[0.5, 0], [0, 0.5]])
        elif self.env_int in [2]:
            self.mean = np.array([0, 1])
            self.std = np.array([[0.5, 0], [0, 0.5]])
        elif self.env_int in [3]:
            self.mean = np.array([1, 1])
            self.std = np.array([[0.5, 0], [0, 0.5]])
        elif self.env_int in [4]:
            self.mean = np.array([0.75, 0.75])
            self.std = np.array([[0.5, 0], [0, 0.5]])

    def reset(self, task_id=None):
        if task_id is not None:
            self.set_init(task_id)
        else:
            self.set_init(self.task_id)

        if self.init_random:
            x = self.goal[0] * random.randint(0, 1) * 2
            y = self.goal[1] * random.randint(0, 1) * 2
        else:
            x = 0
            y = 0

        #x = 7 * random.random() + 2
        #x = 5.5
        #y = 6 * random.random() - 3
        #x = 16 * random.random() - 8
        #y = 16 * random.random() - 8
        #if x >= self.goal[0]-self.reward_range and x <= self.goal[0]+self.reward_range:
        #    x = self.goal[0] - self.reward_range - 0.1 - random.random()
        #if y >= self.goal[1]-self.reward_range and y <= self.goal[1]+self.reward_range:
        #    y = self.goal[1] + self.reward_range + 0.1 + random.random()
        self.real_position = np.array([x, y], dtype='float32')
        self.obs_position = np.array([x, y], dtype='float32')
        self.init_pos = copy.deepcopy(self.real_position)
        self.observation = self.get_obs()
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
        goal_reward = 0.05 * max(self.reward_range - goal_dist, 0) / self.reward_range
        self.line_width = 0.5
        line_reward = max(self.line_width - line_dist, 0)
        #reward = goal_reward + (self.reward_range/self.line_width) * line_reward
        reward = goal_reward

        self.terminal, info = self.check_terminal(pos, goal_dist, line_dist)

        if self.terminal:
            if goal_dist <= self.goal_range:
                reward = 1
                #reward = 1000
            else:
                #reward = -1000
                reward = 0
            #elif line_dist > self.line_width:
            #    #reward = -50
            #    reward = 0
        if reward > 1:
            a = 1

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
        else:
            if goal_dist <= self.goal_range:
                terminal = True
        #elif line_dist > self.line_width:
        #    terminal = True
        #    info = "out_of_line_width"
 
        return terminal, info
    
    def get_obs(self):
        #obs = np.hstack([self.position, self.goal])
        #obs = self.goal - self.position
        obs = self.obs_position
        return obs
    
    def transition(self, action):
        p = self.real_position
        n_p = self.real_position + action
        if self.wall is not None:
            intersection, is_crossing = self.wall.crossing(p, n_p)
            if is_crossing:
                if p[1] < n_p[1]:
                    t_y = intersection[1] - 0.1
                    #n_p = (intersection[0], intersection[1] - 0.1)
                else:
                    t_y = intersection[1] - 0.1
                    #n_p = (intersection[0], intersection[1] + 0.1)
                t_x = self.wall.x_on_line(p, n_p, t_y)
                n_p = (t_x, t_y)

        self.real_position = np.array(n_p)
        error = np.random.multivariate_normal(self.mean, self.std, size=1)
        self.obs_position[0] = self.real_position[0] + error[0, 0]
        self.obs_position[1] = self.real_position[1] + error[0, 1]
    
    def transition_filter_act(self, action):
        tra_action = action
 
        return np.array(tra_action)


    def step(self, action):
        self.time_step += 1

        regular_action = np.array(action, dtype='float32')
        regular_action = np.where(regular_action < self.action_high, regular_action, self.action_high)
        regular_action = np.where(regular_action > -self.action_high, regular_action, -self.action_high)
        regular_action = self.transition_filter_act(regular_action)

        self.transition(regular_action)
        self.observation = self.get_obs()
        self.observation = np.where(self.observation < self.observation_high, self.observation, self.observation_high)
        self.observation = np.where(self.observation > -self.observation_high, self.observation, -self.observation_high)

        reward, done, term_info = self.cal_reward(self.real_position)

        info = {}
        info['real_position'] = self.real_position
        info['obs_position'] = self.obs_position
        info['goal'] = self.goal
        info['terminal'] = term_info

        return self.observation, reward, done, info
    
    def get_info(self):
        info = {}
        info['real_position'] = self.real_position
        info['obs_position'] = self.obs_position
        info['goal'] = self.goal

        return info

class Line:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.a, self.b, self.x = self.cal_coeff(p1, p2)
    
    def cal_coeff(self, p1, p2):
        if p1[0] != p2[0]:
            a = (p2[1] - p1[1]) / (p2[0] - p1[0])
            b = -a * p1[0] + p1[1]
            x = None
        else:
            a = None
            b = None
            x = p1[0]
        
        return a, b, x
    
    def crossing(self, p1, p2):
        a, b, x = self.cal_coeff(p1, p2)

        # 交点の計算
        if self.x is not None and x is None:
            itr_y = a * self.x + b
            itr_x = self.x
            intersection = (self.x, itr_y)

            if self.check_crossing(p1, p2, intersection):
                is_crossing = True
            else:
                is_crossing = False

        elif self.x is None and x is not None:
            itr_y = self.a * x + self.b
            itr_x = x
            intersection = (x, itr_y)

            if self.check_crossing(p1, p2, intersection):
                is_crossing = True
            else:
                is_crossing = False

        elif self.x is not None and x is not None:
            intersection = None
            is_crossing = False
        elif self.a == 0 and a == 0:
            intersection = None
            is_crossing = False
        else:
            c = self.a
            d = self.b
            itr_x = (d - b) / (a - c)
            itr_y = (a*d - b*c) / (a - c)

            intersection = (itr_x, itr_y)
        
            if self.check_crossing(p1, p2, intersection):
                is_crossing = True
            else:
                is_crossing = False
        
        
        return intersection, is_crossing

    def check_crossing(self, p1, p2, intersection):
        itr_x = intersection[0]
        itr_y = intersection[1]
        flag = True
        for tp1, tp2 in ([[p1, p2], [self.p1, self.p2]]):
            min_x = min(tp1[0], tp2[0])
            max_x = max(tp1[0], tp2[0])
            min_y = min(tp1[1], tp2[1])
            max_y = max(tp1[1], tp2[1])

            flag = itr_y >= min_y and itr_y <= max_y and \
                itr_x >= min_x and itr_x <= max_x and flag
            
            if not flag:
                break
        
        return flag
    
    def x_on_line(self, p1, p2, t_y):
        a, b, x = self.cal_coeff(p1, p2)

        if x is None:       # x = **でない場合
            t_x = (t_y - b) / a
        else:   # x = ** の場合
            t_x = x
        
        return t_x


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