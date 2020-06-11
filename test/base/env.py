import time
import gym
from gym.spaces.discrete import Discrete


class MyTestEnv(gym.Env):
    def __init__(self, size, sleep=0, dict_state=False):
        self.size = size
        self.sleep = sleep
        self.dict_state = dict_state
        self.action_space = Discrete(2)
        self.reset()

    def reset(self, state=0):
        self.done = False
        self.index = state
        return {'index': self.index} if self.dict_state else self.index

    def step(self, action):
        if self.done:
            raise ValueError('step after done !!!')
        if self.sleep > 0:
            time.sleep(self.sleep)
        if self.index == self.size:
            self.done = True
            if self.dict_state:
                return {'index': self.index}, 0, True, {}
            else:
                return self.index, 0, True, {}
        if action == 0:
            self.index = max(self.index - 1, 0)
            if self.dict_state:
                return {'index': self.index}, 0, False, {'key': 1, 'env': self}
            else:
                return self.index, 0, False, {}
        elif action == 1:
            self.index += 1
            self.done = self.index == self.size
            if self.dict_state:
                return {'index': self.index}, int(self.done), self.done, \
                    {'key': 1, 'env': self}
            else:
                return self.index, int(self.done), self.done, \
                    {'key': 1, 'env': self}
