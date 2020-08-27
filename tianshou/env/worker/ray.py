import gym
import numpy as np
from typing import List, Callable, Tuple, Optional, Any

from tianshou.env.worker import EnvWorker

try:
    import ray
except ImportError:
    pass


class RayEnvWorker(EnvWorker):
    """Ray worker used in RayVectorEnv."""

    def __init__(self, env_fn: Callable[[], gym.Env]) -> None:
        super().__init__(env_fn)
        self.env = ray.remote(gym.Wrapper).options(num_cpus=0).remote(env_fn())

    def __getattr__(self, key: str) -> Any:
        return ray.get(self.env.__getattr__.remote(key))

    def reset(self) -> Any:
        return ray.get(self.env.reset.remote())

    @staticmethod
    def wait(workers: List['RayEnvWorker'],
             wait_num: int,
             timeout: Optional[float] = None) -> List['RayEnvWorker']:
        results = [x.result for x in workers]
        ready_results, _ = ray.wait(results,
                                    num_returns=wait_num, timeout=timeout)
        return [workers[results.index(result)] for result in ready_results]

    def send_action(self, action: np.ndarray) -> None:
        # self.action is actually a handle
        self.result = self.env.step.remote(action)

    def get_result(self) -> Tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return ray.get(self.result)

    def seed(self, seed: Optional[int] = None) -> List[int]:
        if hasattr(self.env, 'seed'):
            return ray.get(self.env.seed.remote(seed))
        return None

    def render(self, **kwargs) -> Any:
        if hasattr(self.env, 'render'):
            return ray.get(self.env.render.remote(**kwargs))
        return None

    def close_env(self) -> None:
        ray.get(self.env.close.remote())
