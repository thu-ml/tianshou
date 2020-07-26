import gym
import numpy as np
from typing import List, Tuple, Union, Optional, Callable, Any

from tianshou.env import BaseVectorEnv


class VectorEnv(BaseVectorEnv):
    """Dummy vectorized environment wrapper, implemented in for-loop.

    .. seealso::

        Please refer to :class:`~tianshou.env.BaseVectorEnv` for more detailed
        explanation.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]]) -> None:
        super().__init__(env_fns)
        self.envs = [_() for _ in env_fns]

    def __getattr__(self, key):
        return [getattr(env, key) if hasattr(env, key) else None
                for env in self.envs]

    def reset(self, id: Optional[Union[int, List[int]]] = None) -> np.ndarray:
        if id is None:
            id = range(self.env_num)
        elif np.isscalar(id):
            id = [id]
        obs = np.stack([self.envs[i].reset() for i in id])
        return obs

    def step(self,
             action: np.ndarray,
             id: Optional[Union[int, List[int]]] = None
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if id is None:
            id = range(self.env_num)
        elif np.isscalar(id):
            id = [id]
        assert len(action) == len(id)
        result = [self.envs[i].step(action[i]) for i in id]
        obs, rew, done, info = map(np.stack, zip(*result))
        return obs, rew, done, info

    def seed(self, seed: Optional[Union[int, List[int]]] = None) -> List[int]:
        if np.isscalar(seed):
            seed = [seed + _ for _ in range(self.env_num)]
        elif seed is None:
            seed = [seed] * self.env_num
        result = []
        for e, s in zip(self.envs, seed):
            if hasattr(e, 'seed'):
                result.append(e.seed(s))
        return result

    def render(self, **kwargs) -> List[Any]:
        result = []
        for e in self.envs:
            if hasattr(e, 'render'):
                result.append(e.render(**kwargs))
        return result

    def close(self) -> List[Any]:
        return [e.close() for e in self.envs]
