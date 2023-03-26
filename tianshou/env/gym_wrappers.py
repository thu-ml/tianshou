from typing import Any, Dict, List, SupportsFloat, Tuple, Union

import gymnasium as gym
import numpy as np
from packaging import version


class ContinuousToDiscrete(gym.ActionWrapper):
    """Gym environment wrapper to take discrete action in a continuous environment.

    :param gym.Env env: gym environment with continuous action space.
    :param int action_per_dim: number of discrete actions in each dimension
        of the action space.
    """

    def __init__(self, env: gym.Env, action_per_dim: Union[int, List[int]]) -> None:
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Box)
        low, high = env.action_space.low, env.action_space.high
        if isinstance(action_per_dim, int):
            action_per_dim = [action_per_dim] * env.action_space.shape[0]
        assert len(action_per_dim) == env.action_space.shape[0]
        self.action_space = gym.spaces.MultiDiscrete(action_per_dim)
        self.mesh = np.array(
            [np.linspace(lo, hi, a) for lo, hi, a in zip(low, high, action_per_dim)],
            dtype=object
        )

    def action(self, act: np.ndarray) -> np.ndarray:
        # modify act
        assert len(act.shape) <= 2, f"Unknown action format with shape {act.shape}."
        if len(act.shape) == 1:
            return np.array([self.mesh[i][a] for i, a in enumerate(act)])
        return np.array([[self.mesh[i][a] for i, a in enumerate(a_)] for a_ in act])


class MultiDiscreteToDiscrete(gym.ActionWrapper):
    """Gym environment wrapper to take discrete action in multidiscrete environment.

    :param gym.Env env: gym environment with multidiscrete action space.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
        nvec = env.action_space.nvec
        assert nvec.ndim == 1
        self.bases = np.ones_like(nvec)
        for i in range(1, len(self.bases)):
            self.bases[i] = self.bases[i - 1] * nvec[-i]
        self.action_space = gym.spaces.Discrete(np.prod(nvec))

    def action(self, act: np.ndarray) -> np.ndarray:
        converted_act = []
        for b in np.flip(self.bases):
            converted_act.append(act // b)
            act = act % b
        return np.array(converted_act).transpose()


class TruncatedAsTerminated(gym.Wrapper):
    """A wrapper that set ``terminated = terminated or truncated`` for ``step()``.

    It's intended to use with ``gym.wrappers.TimeLimit``.

    :param gym.Env env: gym environment.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if not version.parse(gym.__version__) >= version.parse('0.26.0'):
            raise EnvironmentError(
                f"TruncatedAsTerminated is not applicable with gym version \
                {gym.__version__}"
            )

    def step(self,
             act: np.ndarray) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        observation, reward, terminated, truncated, info = super().step(act)
        terminated = (terminated or truncated)
        return observation, reward, terminated, truncated, info
