import gym
import time
import numpy as np
from tianshou.env import FrameStack, VectorEnv, SubprocVectorEnv, RayVectorEnv


class MyTestEnv(gym.Env):
    def __init__(self, size, sleep=0):
        self.size = size
        self.sleep = sleep
        self.index = 0

    def reset(self):
        self.index = 0
        return self.index

    def step(self, action):
        if self.sleep > 0:
            time.sleep(self.sleep)
        if self.index == self.size:
            return self.index, 0, True, {}
        if action == 0:
            self.index = max(self.index - 1, 0)
            return self.index, 0, False, {}
        elif action == 1:
            self.index += 1
            finished = self.index == self.size
            return self.index, int(finished), finished, {}

def test_framestack():
    k = 4
    size = 10
    env = MyTestEnv(size=size)
    fsenv = FrameStack(env, k)
    fsenv.seed()
    obs = fsenv.reset()
    assert abs(obs - np.array([0, 0, 0, 0])).sum() == 0
    for i in range(5):
        obs, rew, done, info = fsenv.step(1)
    assert abs(obs - np.array([2, 3, 4, 5])).sum() == 0
    for i in range(10):
        obs, rew, done, info = fsenv.step(0)
    assert abs(obs - np.array([0, 0, 0, 0])).sum() == 0
    for i in range(9):
        obs, rew, done, info = fsenv.step(1)
    assert abs(obs - np.array([6, 7, 8, 9])).sum() == 0
    assert (rew, done) == (0, False)
    obs, rew, done, info = fsenv.step(1)
    assert abs(obs - np.array([7, 8, 9, 10])).sum() == 0
    assert (rew, done) == (1, True)
    obs, rew, done, info = fsenv.step(0)
    assert abs(obs - np.array([8, 9, 10, 10])).sum() == 0
    assert (rew, done) == (0, True)
    fsenv.close()

if __name__ == '__main__':
    test_framestack()