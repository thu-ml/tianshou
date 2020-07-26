import gym
import numpy as np
from typing import List, Tuple, Union, Optional, Callable, Any

try:
    import ray
except ImportError:
    pass

from tianshou.env import BaseVectorEnv


class RayVectorEnv(BaseVectorEnv):
    """Vectorized environment wrapper based on
    `ray <https://github.com/ray-project/ray>`_. This is a choice to run
    distributed environments in a cluster.

    .. seealso::

        Please refer to :class:`~tianshou.env.BaseVectorEnv` for more detailed
        explanation.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]]) -> None:
        super().__init__(env_fns)
        try:
            if not ray.is_initialized():
                ray.init()
        except NameError:
            raise ImportError(
                'Please install ray to support RayVectorEnv: pip install ray')
        self.envs = [
            ray.remote(gym.Wrapper).options(num_cpus=0).remote(e())
            for e in env_fns]

    def __getattr__(self, key):
        return ray.get([e.__getattr__.remote(key) for e in self.envs])

    def step(self,
             action: np.ndarray,
             id: Optional[Union[int, List[int]]] = None
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if id is None:
            id = range(self.env_num)
        elif np.isscalar(id):
            id = [id]
        assert len(action) == len(id)
        result = ray.get([self.envs[j].step.remote(action[i])
                          for i, j in enumerate(id)])
        obs, rew, done, info = map(np.stack, zip(*result))
        return obs, rew, done, info

    def reset(self, id: Optional[Union[int, List[int]]] = None) -> np.ndarray:
        if id is None:
            id = range(self.env_num)
        elif np.isscalar(id):
            id = [id]
        obs = np.stack(ray.get([self.envs[i].reset.remote() for i in id]))
        return obs

    def seed(self, seed: Optional[Union[int, List[int]]] = None) -> List[int]:
        if not hasattr(self.envs[0], 'seed'):
            return []
        if np.isscalar(seed):
            seed = [seed + _ for _ in range(self.env_num)]
        elif seed is None:
            seed = [seed] * self.env_num
        return ray.get([e.seed.remote(s) for e, s in zip(self.envs, seed)])

    def render(self, **kwargs) -> List[Any]:
        if not hasattr(self.envs[0], 'render'):
            return [None for e in self.envs]
        return ray.get([e.render.remote(**kwargs) for e in self.envs])

    def close(self) -> List[Any]:
        return ray.get([e.close.remote() for e in self.envs])
