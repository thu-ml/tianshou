import gym
import time
import random
import numpy as np
from gym.spaces import Discrete, MultiDiscrete, Box, Dict, Tuple


class MyTestEnv(gym.Env):
    """This is a "going right" task. The task is to go right ``size`` steps.
    """

    def __init__(self, size, sleep=0, dict_state=False, recurse_state=False,
                 ma_rew=0, multidiscrete_action=False, random_sleep=False):
        assert not (dict_state and recurse_state), "dict_state and recurse_state cannot both be true"
        self.size = size
        self.sleep = sleep
        self.random_sleep = random_sleep
        self.dict_state = dict_state
        self.recurse_state = recurse_state
        self.ma_rew = ma_rew
        self._md_action = multidiscrete_action
        if  dict_state:
            self.observation_space = Dict(
                {"index": Box(shape=(1, ), low=0, high=size - 1),
                 "rand": Box(shape=(1,), low=0, high=1)})
        elif recurse_state:
            self.observation_space = Dict(
                {"index": Box(shape=(1, ), low=0, high=size - 1),
                 "dict": Dict({
                     "tuple": Tuple((Discrete(2), Box(shape=(2,), low=0, high=1))),
                     "rand": Box(shape=(1,2), low=0, high=1)})
                })
        else:
            self.observation_space = Box(shape=(1, ), low=0, high=size - 1)
        if multidiscrete_action:
            self.action_space = MultiDiscrete([2, 2])
        else:
            self.action_space = Discrete(2)
        self.reset()

    def seed(self, seed=0):
        np.random.seed(seed)

    def reset(self, state=0):
        self.done = False
        self.index = state
        return self._get_dict_state()

    def _get_reward(self):
        """Generate a non-scalar reward if ma_rew is True."""
        x = int(self.done)
        if self.ma_rew > 0:
            return [x] * self.ma_rew
        return x

    def _get_dict_state(self):
        """Generate a dict_state if dict_state is True."""
        if self.dict_state:
            return {'index': np.array([self.index]), 'rand': np.random.rand()}
        elif self.recurse_state:
            return {'index': np.array([self.index]),
                    'dict': {"tuple": (1, np.random.rand(2)),
                             "rand": np.random.rand(1, 2)}}
        else:
            return np.array([self.index])

    def step(self, action):
        if self._md_action:
            action = action[0]
        if self.done:
            raise ValueError('step after done !!!')
        if self.sleep > 0:
            sleep_time = random.random() if self.random_sleep else 1
            sleep_time *= self.sleep
            time.sleep(sleep_time)
        if self.index == self.size:
            self.done = True
            return self._get_dict_state(), self._get_reward(), self.done, {}
        if action == 0:
            self.index = max(self.index - 1, 0)
            return self._get_dict_state(), self._get_reward(), self.done, \
                {'key': 1, 'env': self} if self.dict_state else {}
        elif action == 1:
            self.index += 1
            self.done = self.index == self.size
            return self._get_dict_state(), self._get_reward(), \
                self.done, {'key': 1, 'env': self}
