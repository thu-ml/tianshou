from typing import List, Union, Callable

import gym
import numpy as np

from tianshou.data import Batch


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


class GoalEnv(gym.Env):
    observation_space: gym.spaces.Space[gym.spaces.Dict]


class GoalEnvWrapper(gym.ObservationWrapper):

    def __init__(
        self,
        env: GoalEnv,
        compute_reward: Callable[[np.ndarray, np.ndarray, dict], np.ndarray],
        obs_space_keys: List[str] = ['observation', 'achieved_goal', 'desired_goal']
        # obs_space_keys must be in o, ag, g order.
    ) -> None:
        super().__init__(env)
        self.env = env
        self.compute_reward = compute_reward
        self.obs_space_keys = obs_space_keys
        self.original_space: gym.spaces.Space[gym.spaces.Dict] \
            = self.env.observation_space

        self.observation_space = self.calculate_obs_space()

    def calculate_obs_space(self) -> gym.Space:
        for k in self.obs_space_keys:
            assert isinstance(self.original_space[k], gym.spaces.Box)
        new_low = np.concatenate(
            [self.original_space[k].low.flatten() for k in self.obs_space_keys],
            axis=0
        )

        new_high = np.concatenate(
            [self.original_space[k].high.flatten() for k in self.obs_space_keys],
            axis=0
        )
        new_shape = np.concatenate(
            [self.original_space[k].sample().flatten() for k in self.obs_space_keys],
            axis=0
        ).shape
        samples = [
            self.original_space[k].sample().flatten() for k in self.obs_space_keys
        ]
        self.partitions = np.cumsum([0] + [len(s) for s in samples], dtype=int)
        return gym.spaces.Box(new_low, new_high, new_shape)

    def deconstruct_obs_fn(self, obs: np.ndarray) -> Batch:
        """Deconstruct observation into observation, acheived_goal, goal. The first
        dimension (bsz) is optional.
        obs: shape(bsz, *observation_shape)
        return: Batch(
            o=shape(bsz, *o.shape),
            ag=shape(bsz, *ag.shape),
            g=shape(bsz, *g.shape)
        ) or Batch without the first dim (bsz) according to the input.
        """
        new_shapes = [
            [*self.original_space[self.obs_space_keys[0]].shape],
            [*self.original_space[self.obs_space_keys[1]].shape],
            [*self.original_space[self.obs_space_keys[2]].shape],
        ]
        if len(obs.shape) == 2:
            new_shapes = [[-1] + s for s in new_shapes]
        batch = Batch(
            o=obs[..., self.partitions[0]:self.partitions[1]].reshape(*new_shapes[0]),
            ag=obs[..., self.partitions[1]:self.partitions[2]].reshape(*new_shapes[1]),
            g=obs[..., self.partitions[2]:self.partitions[3]].reshape(*new_shapes[2]),
        )
        return batch

    def flatten_obs_fn(self, obs: Batch) -> np.ndarray:
        """Reconstruct observation. The first dim (bsz) is optional
        obs: Batch(
            o=shape(bsz, *o.shape),
            ag=shape(bsz, *ag.shape),
            g=shape(bsz, *g.shape)
        )
        return: shape(bsz, *observation_shape)
        """
        new_shape = [-1]
        if len(obs.o.shape) > len(self.original_space[self.obs_space_keys[0]].shape):
            bsz = obs.shape[0]
            new_shape = [bsz, -1]
        return np.concatenate(
            [
                obs.o.reshape(*new_shape),
                obs.ag.reshape(*new_shape),
                obs.g.reshape(*new_shape)
            ],
            axis=-1
        )

    def compute_reward_fn(self, obs: Batch) -> np.ndarray:
        """Compute rewards from deconstructed obs. The first dim (bsz) is optional
        obs: Batch(
            o=shape(bsz, *o.shape),
            ag=shape(bsz, *ag.shape),
            g=shape(bsz, *g.shape)
        )
        return: shape(bsz,)
        """
        ag = obs.ag
        g = obs.g
        return self.compute_reward(ag, g, {})

    def observation(self, observation: dict) -> np.ndarray:
        o = observation[self.obs_space_keys[0]].flatten()
        ag = observation[self.obs_space_keys[1]].flatten()
        g = observation[self.obs_space_keys[2]].flatten()
        return np.concatenate([o, ag, g])
