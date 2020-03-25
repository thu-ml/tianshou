import numpy as np
from collections import deque


class EnvWrapper(object):
    def __init__(self, env):
        self.env = env

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def seed(self, seed=None):
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)

    def render(self, **kwargs):
        if hasattr(self.env, 'render'):
            return self.env.render(**kwargs)

    def close(self):
        self.env.close()


class FrameStack(EnvWrapper):
    def __init__(self, env, stack_num):
        """Stack last k frames."""
        super().__init__(env)
        self.stack_num = stack_num
        self._frames = deque([], maxlen=stack_num)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.stack_num):
            self._frames.append(obs)
        return self._get_obs()

    def _get_obs(self):
        try:
            return np.concatenate(self._frames, axis=-1)
        except ValueError:
            return np.stack(self._frames, axis=-1)
