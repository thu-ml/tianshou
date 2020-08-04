from typing import List, Callable, Tuple, Optional, Any

import gym
import numpy as np

from tianshou.env.worker.base import EnvWorker


class SequentialEnvWorker(EnvWorker):
    """
    Dummy worker used in sequential vector environments
    """

    @staticmethod
    def wait(workers: List['SequentialEnvWorker']
             ) -> List['SequentialEnvWorker']:
        # SequentialEnvWorker objects are always ready
        return workers

    def __init__(self, env_fn: Callable[[], gym.Env]) -> None:
        super(SequentialEnvWorker, self).__init__(env_fn)
        self.env = env_fn()

    def __getattr__(self, key: str):
        if hasattr(self.env, key):
            return getattr(self.env, key)
        return None

    def reset(self):
        return self.env.reset()

    def step(self, action: np.ndarray
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.env.step(action)

    def seed(self, seed: Optional[int] = None):
        return self.env.seed(seed) if hasattr(self.env, 'seed') else None

    def render(self, **kwargs) -> None:
        return self.env.render(**kwargs) if \
            hasattr(self.env, 'render') else None

    def close(self) -> Any:
        return self.env.close()
