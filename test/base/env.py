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
        assert not (
            dict_state and recurse_state), \
            "dict_state and recurse_state cannot both be true"
        self.size = size
        self.sleep = sleep
        self.random_sleep = random_sleep
        self.dict_state = dict_state
        self.recurse_state = recurse_state
        self.ma_rew = ma_rew
        self._md_action = multidiscrete_action
        # how many steps this env has stepped
        self.steps = 0
        if dict_state:
            self.observation_space = Dict(
                {"index": Box(shape=(1, ), low=0, high=size - 1),
                 "rand": Box(shape=(1,), low=0, high=1, dtype=np.float64)})
        elif recurse_state:
            self.observation_space = Dict(
                {"index": Box(shape=(1, ), low=0, high=size - 1),
                 "dict": Dict({
                     "tuple": Tuple((Discrete(2), Box(shape=(2,),
                                     low=0, high=1, dtype=np.float64))),
                     "rand": Box(shape=(1, 2), low=0, high=1,
                                 dtype=np.float64)})
                 })
        else:
            self.observation_space = Box(shape=(1, ), low=0, high=size - 1)
        if multidiscrete_action:
            self.action_space = MultiDiscrete([2, 2])
        else:
            self.action_space = Discrete(2)
        self.done = False
        self.index = 0
        self.seed()

    def seed(self, seed=0):
        self.rng = np.random.RandomState(seed)
        return [seed]

    def reset(self, state=0):
        self.done = False
        self.index = state
        return self._get_state()

    def _get_reward(self):
        """Generate a non-scalar reward if ma_rew is True."""
        x = int(self.done)
        if self.ma_rew > 0:
            return [x] * self.ma_rew
        return x

    def _get_state(self):
        """Generate state(observation) of MyTestEnv"""
        if self.dict_state:
            return {'index': np.array([self.index], dtype=np.float32),
                    'rand': self.rng.rand(1)}
        elif self.recurse_state:
            return {'index': np.array([self.index], dtype=np.float32),
                    'dict': {"tuple": (np.array([1],
                                       dtype=np.int64), self.rng.rand(2)),
                             "rand": self.rng.rand(1, 2)}}
        else:
            return np.array([self.index], dtype=np.float32)

    def step(self, action):
        self.steps += 1
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
            return self._get_state(), self._get_reward(), self.done, {}
        if action == 0:
            self.index = max(self.index - 1, 0)
            return self._get_state(), self._get_reward(), self.done, \
                {'key': 1, 'env': self} if self.dict_state else {}
        elif action == 1:
            self.index += 1
            self.done = self.index == self.size
            return self._get_state(), self._get_reward(), \
                self.done, {'key': 1, 'env': self}
