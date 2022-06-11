import gym
import numpy as np


class ContinuousToDiscrete(gym.ActionWrapper):
    """Gym environment wrapper to take discrete action in a continuous environment.

    :param gym.Env env: gym environment with continuous action space.
    :param int action_per_branch: number of discrete actions in each dimension
        of the action space.
    """

    def __init__(self, env: gym.Env, action_per_branch: int) -> None:
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Box)
        low, high = env.action_space.low, env.action_space.high
        num_branches = env.action_space.shape[0]
        self.action_space = gym.spaces.MultiDiscrete(
            [action_per_branch] * num_branches
        )
        mesh = []
        for lo, hi in zip(low, high):
            mesh.append(np.linspace(lo, hi, action_per_branch))
        self.mesh = np.array(mesh)

    def action(self, act: np.ndarray) -> np.ndarray:
        # modify act
        return np.array([self.mesh[i][a] for i, a in enumerate(act)])


class MultiDiscreteToDiscrete(gym.ActionWrapper):
    """Gym environment wrapper to discrete action in multidiscrete environment.

    :param gym.Env env: gym environment with multidiscrete action space.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        nvec = env.action_space.nvec
        assert nvec.ndim == 1
        self.bases = np.ones_like(nvec)
        for i in range(1, len(self.bases)):
            self.bases[i] = self.bases[i-1]*nvec[-i]
        self.action_space = gym.spaces.Discrete(np.prod(nvec))

    def action(self, act: int) -> np.ndarray:
        converted_act = []
        for b in np.flip(self.bases):
            converted_act.append(act // b)
            act = act % b
        return np.array(converted_act).transpose()
