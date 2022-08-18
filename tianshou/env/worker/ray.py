from typing import Any, Callable, List, Optional, Tuple, Union

import gym
import numpy as np

from tianshou.env.worker import EnvWorker

try:
    import ray
except ImportError:
    pass


class _SetAttrWrapper(gym.Wrapper):

    def set_env_attr(self, key: str, value: Any) -> None:
        setattr(self.env.unwrapped, key, value)

    def get_env_attr(self, key: str) -> Any:
        return getattr(self.env, key)


class RayEnvWorker(EnvWorker):
    """Ray worker used in RayVectorEnv."""

    def __init__(self, env_fn: Callable[[], gym.Env]) -> None:
        self.env = ray.remote(_SetAttrWrapper).options(  # type: ignore
            num_cpus=0
        ).remote(env_fn())
        super().__init__(env_fn)

    def get_env_attr(self, key: str) -> Any:
        return ray.get(self.env.get_env_attr.remote(key))

    def set_env_attr(self, key: str, value: Any) -> None:
        ray.get(self.env.set_env_attr.remote(key, value))

    def reset(self, **kwargs: Any) -> Any:
        if "seed" in kwargs:
            super().seed(kwargs["seed"])
        return ray.get(self.env.reset.remote(**kwargs))

    @staticmethod
    def wait(  # type: ignore
        workers: List["RayEnvWorker"], wait_num: int, timeout: Optional[float] = None
    ) -> List["RayEnvWorker"]:
        results = [x.result for x in workers]
        ready_results, _ = ray.wait(results, num_returns=wait_num, timeout=timeout)
        return [workers[results.index(result)] for result in ready_results]

    def send(self, action: Optional[np.ndarray], **kwargs: Any) -> None:
        # self.result is actually a handle
        if action is None:
            self.result = self.env.reset.remote(**kwargs)
        else:
            self.result = self.env.step.remote(action)

    def recv(
        self
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        return ray.get(self.result)  # type: ignore

    def seed(self, seed: Optional[int] = None) -> Optional[List[int]]:
        super().seed(seed)
        try:
            return ray.get(self.env.seed.remote(seed))
        except NotImplementedError:
            self.env.reset.remote(seed=seed)
            return None

    def render(self, **kwargs: Any) -> Any:
        return ray.get(self.env.render.remote(**kwargs))

    def close_env(self) -> None:
        ray.get(self.env.close.remote())
