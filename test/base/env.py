import gym
import time


class MyTestEnv(gym.Env):
    def __init__(self, size, sleep=0):
        self.size = size
        self.sleep = sleep
        self.reset()

    def reset(self, state=0):
        self.done = False
        self.index = state
        return self.index

    def step(self, action):
        if self.done:
            raise ValueError('step after done !!!')
        if self.sleep > 0:
            time.sleep(self.sleep)
        if self.index == self.size:
            self.done = True
            return self.index, 0, True, {}
        if action == 0:
            self.index = max(self.index - 1, 0)
            return self.index, 0, False, {}
        elif action == 1:
            self.index += 1
            self.done = self.index == self.size
            return self.index, int(self.done), self.done, {'key': 1}
