import gym
import numpy as np
from typing import List, Callable, Optional, Any

from tianshou.env.worker import EnvWorker


class DummyEnvWorker(EnvWorker):
    """Dummy worker used in sequential vector environments."""

    def __init__(self, env_fn: Callable[[], gym.Env]) -> None:
        super().__init__(env_fn)
        self.env = env_fn()

    def __getattr__(self, key: str):
        if hasattr(self.env, key):
            return getattr(self.env, key)
        return None

    @staticmethod
    def wait(workers: List['DummyEnvWorker']
             ) -> List['DummyEnvWorker']:
        # SequentialEnvWorker objects are always ready
        return workers

    def reset(self):
        return self.env.reset()

    def send_action(self, action: np.ndarray):
        self.result = self.env.step(action)

    def seed(self, seed: Optional[int] = None):
        return self.env.seed(seed) if hasattr(self.env, 'seed') else None

    def render(self, **kwargs) -> None:
        return self.env.render(**kwargs) if \
            hasattr(self.env, 'render') else None

    def close(self) -> Any:
        return self.env.close()
