from collections.abc import Callable
from typing import Any

import gymnasium as gym
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
        setattr(self.env.unwrapped, key, value)

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict]:
        if "seed" in kwargs:
            super().seed(kwargs["seed"])
        return self.env.reset(**kwargs)

    @staticmethod
    def wait(  # type: ignore
        workers: list["DummyEnvWorker"],
        wait_num: int,
        timeout: float | None = None,
    ) -> list["DummyEnvWorker"]:
        # Sequential EnvWorker objects are always ready
        return workers

    def send(self, action: np.ndarray | None, **kwargs: Any) -> None:
        if action is None:
            self.result = self.env.reset(**kwargs)
        else:
            self.result = self.env.step(action)  # type: ignore

    def seed(self, seed: int | None = None) -> list[int] | None:
        super().seed(seed)
        try:
            return self.env.seed(seed)  # type: ignore
        except (AttributeError, NotImplementedError):
            self.env.reset(seed=seed)
            return [seed]  # type: ignore

    def render(self, **kwargs: Any) -> Any:
        return self.env.render(**kwargs)

    def close_env(self) -> None:
        self.env.close()
