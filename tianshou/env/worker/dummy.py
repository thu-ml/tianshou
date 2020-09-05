import gym
import numpy as np
from typing import List, Callable, Optional, Any

from tianshou.env.worker import EnvWorker


class DummyEnvWorker(EnvWorker):
    """Dummy worker used in sequential vector environments."""

    def __init__(self, env_fn: Callable[[], gym.Env]) -> None:
        super().__init__(env_fn)
        self.env = env_fn()

    def __getattr__(self, key: str) -> Any:
        return getattr(self.env, key)

    def reset(self) -> Any:
        return self.env.reset()

    @staticmethod
    def wait(workers: List['DummyEnvWorker'],
             wait_num: int,
             timeout: Optional[float] = None) -> List['DummyEnvWorker']:
        # SequentialEnvWorker objects are always ready
        return workers

    def send_action(self, action: np.ndarray) -> None:
        self.result = self.env.step(action)

    def seed(self, seed: Optional[int] = None) -> List[int]:
        return self.env.seed(seed) if hasattr(self.env, 'seed') else None

    def render(self, **kwargs) -> Any:
        return self.env.render(**kwargs) \
            if hasattr(self.env, 'render') else None

    def close_env(self) -> None:
        self.env.close()
