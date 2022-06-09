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

    :param gym.Env env: gym environment with continuous action space.
    """
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
        self.num_dim = env.action_space.shape[0]
        assert len(set(env.action_space.nvec)) == 1 # TODO support for different num of actions per dim
        self.action_per_dim = env.action_space.nvec[0]
        self.action_space = gym.spaces.Discrete(
            self.action_per_dim ** self.num_dim
        )

    def action(self, act: np.ndarray) -> np.ndarray:
        # modify act
        converted_act = []
        for i in range(self.num_dim):
            converted_act.append(act // self.action_per_dim ** (self.num_dim-i))
            act = act % self.action_per_dim ** (self.num_dim-i)
        return np.array(converted_act)