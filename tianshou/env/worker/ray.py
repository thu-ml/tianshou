from typing import List, Callable, Tuple, Optional, Any

import gym
import numpy as np
import ray

from tianshou.env.worker.base import EnvWorker


class RayEnvWorker(EnvWorker):
    @staticmethod
    def wait(workers: List['RayEnvWorker']) -> List['RayEnvWorker']:
        ready_envs, _ = ray.wait(
            [x.env for x in workers],
            num_returns=len(workers),
            timeout=0)
        return [workers[ready_envs.index(env)] for env in ready_envs]

    def __init__(self, env_fn: Callable[[], gym.Env]) -> None:
        super(RayEnvWorker, self).__init__(env_fn)
        self.env = ray.remote(gym.Wrapper).options(num_cpus=0).remote(env_fn())

    def __getattr__(self, key: str):
        return ray.get(self.env.__getattr__.remote(key))

    def reset(self):
        return ray.get(self.env.reset.remote())

    def get_result(self
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return ray.get(self._result)

    def send_action(self, action: np.ndarray):
        # self.action is actually a handle
        self._result = self.env.step.remote(action)

    def seed(self, seed: Optional[int] = None):
        if hasattr(self.env, 'seed'):
            return ray.get(self.env.seed.remote(seed))
        return None

    def render(self, **kwargs) -> None:
        if hasattr(self.env, 'render'):
            return ray.get(self.env.render.remote(**kwargs))
        return None

    def close(self) -> Any:
        return ray.get(self.env.close.remote())
