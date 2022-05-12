from typing import Any, Callable, List, Optional, Tuple, Union

import gym
import numpy as np

from tianshou.env.worker import EnvWorker


class DummyEnvWorker(EnvWorker):
    """Dummy worker used in sequential vector environments."""

    def __init__(self, env_fn: Callable[[], gym.Env]) -> None:
        self.env = env_fn()
        super().__init__(env_fn)

    def get_env_attr(self, key: str) -> Any:
        return getattr(self.env, key)

    def set_env_attr(self, key: str, value: Any) -> None:
        setattr(self.env, key, value)

    def reset(self, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        return self.env.reset(**kwargs)

    @staticmethod
    def wait(  # type: ignore
        workers: List["DummyEnvWorker"], wait_num: int, timeout: Optional[float] = None
    ) -> List["DummyEnvWorker"]:
        # Sequential EnvWorker objects are always ready
        return workers

    def send(self, action: Optional[np.ndarray]) -> None:
        if action is None:
            self.result = self.env.reset()  # type: ignore
        else:
            self.result = self.env.step(action)  # type: ignore

    def seed(self, seed: Optional[int] = None) -> Optional[List[int]]:
        super().seed(seed)
        try:
            return self.env.seed(seed)
        except NotImplementedError:
            self.env.reset(seed=seed)

    def render(self, **kwargs: Any) -> Any:
        return self.env.render(**kwargs)

    def close_env(self) -> None:
        self.env.close()
