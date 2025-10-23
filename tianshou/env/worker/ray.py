# mypy: disable-error-code=unused-ignore
import contextlib
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np

from tianshou.env.utils import ENV_TYPE, gym_new_venv_step_type
from tianshou.env.worker import EnvWorker

with contextlib.suppress(ImportError):
    import ray


class _SetAttrWrapper(gym.Wrapper):
    def set_env_attr(self, key: str, value: Any) -> None:
        setattr(self.env.unwrapped, key, value)

    def get_env_attr(self, key: str) -> Any:
        return getattr(self.env, key)


class RayEnvWorker(EnvWorker):
    """Ray worker used in RayVectorEnv."""

    def __init__(
        self,
        env_fn: Callable[[], ENV_TYPE],
    ) -> None:  # TODO: is ENV_TYPE actually correct?
        self.env = ray.remote(_SetAttrWrapper).options(num_cpus=0).remote(env_fn())  # type: ignore
        super().__init__(env_fn)

    def get_env_attr(self, key: str) -> Any:
        return ray.get(self.env.get_env_attr.remote(key))  # type: ignore

    def set_env_attr(self, key: str, value: Any) -> None:
        ray.get(self.env.set_env_attr.remote(key, value))  # type: ignore

    def reset(self, **kwargs: Any) -> Any:
        if "seed" in kwargs:
            super().seed(kwargs["seed"])
        return ray.get(self.env.reset.remote(**kwargs))  # type: ignore

    @staticmethod
    def wait(  # type: ignore
        workers: list["RayEnvWorker"],
        wait_num: int,
        timeout: float | None = None,
    ) -> list["RayEnvWorker"]:
        results = [x.result for x in workers]
        ready_results, _ = ray.wait(results, num_returns=wait_num, timeout=timeout)  # type: ignore
        return [workers[results.index(result)] for result in ready_results]

    def send(self, action: np.ndarray | None, **kwargs: Any) -> None:
        # self.result is actually a handle
        if action is None:
            self.result = self.env.reset.remote(**kwargs)  # type: ignore
        else:
            self.result = self.env.step.remote(action)  # type: ignore

    def recv(self) -> gym_new_venv_step_type:
        return ray.get(self.result)  # type: ignore

    def seed(self, seed: int | None = None) -> list[int] | None:
        super().seed(seed)
        try:
            return ray.get(self.env.seed.remote(seed))  # type: ignore
        except (AttributeError, NotImplementedError):
            self.env.reset.remote(seed=seed)  # type: ignore
            return None

    def render(self, **kwargs: Any) -> Any:
        return ray.get(self.env.render.remote(**kwargs))  # type: ignore

    def close_env(self) -> None:
        ray.get(self.env.close.remote())  # type: ignore
