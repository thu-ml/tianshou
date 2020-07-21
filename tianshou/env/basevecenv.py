import gym
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Optional, Callable


class BaseVectorEnv(ABC, gym.Env):
    """Base class for vectorized environments wrapper. Usage:
    ::

        env_num = 8
        envs = VectorEnv([lambda: gym.make(task) for _ in range(env_num)])
        assert len(envs) == env_num

    It accepts a list of environment generators. In other words, an environment
    generator ``efn`` of a specific task means that ``efn()`` returns the
    environment of the given task, for example, ``gym.make(task)``.

    All of the VectorEnv must inherit :class:`~tianshou.env.BaseVectorEnv`.
    Here are some other usages:
    ::

        envs.seed(2)  # which is equal to the next line
        envs.seed([2, 3, 4, 5, 6, 7, 8, 9])  # set specific seed for each env
        obs = envs.reset()  # reset all environments
        obs = envs.reset([0, 5, 7])  # reset 3 specific environments
        obs, rew, done, info = envs.step([1] * 8)  # step synchronously
        envs.render()  # render all environments
        envs.close()  # close all environments
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]]) -> None:
        self._env_fns = env_fns
        self.env_num = len(env_fns)

    def __len__(self) -> int:
        """Return len(self), which is the number of environments."""
        return self.env_num

    def __getattribute__(self, key: str):
        """Switch between the default attribute getter or one
           looking at wrapped environment level depending on the key."""
        if key not in ('observation_space', 'action_space'):
            return super().__getattribute__(key)
        else:
            return self.__getattr__(key)

    @abstractmethod
    def __getattr__(self, key: str):
        """Try to retrieve an attribute from each individual wrapped
           environment, if it does not belong to the wrapping vector
           environment class."""
        pass

    @abstractmethod
    def reset(self, id: Optional[Union[int, List[int]]] = None):
        """Reset the state of all the environments and return initial
        observations if id is ``None``, otherwise reset the specific
        environments with given id, either an int or a list.
        """
        pass

    @abstractmethod
    def step(self,
             action: np.ndarray,
             id: Optional[Union[int, List[int]]] = None
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run one timestep of all the environments’ dynamics if id is
        ``None``, otherwise run one timestep for some environments
        with given id,  either an int or a list. When the end of
        episode is reached, you are responsible for calling reset(id)
        to reset this environment’s state.

        Accept a batch of action and return a tuple (obs, rew, done, info).

        :param numpy.ndarray action: a batch of action provided by the agent.

        :return: A tuple including four items:

            * ``obs`` a numpy.ndarray, the agent's observation of current \
                environments
            * ``rew`` a numpy.ndarray, the amount of rewards returned after \
                previous actions
            * ``done`` a numpy.ndarray, whether these episodes have ended, in \
                which case further step() calls will return undefined results
            * ``info`` a numpy.ndarray, contains auxiliary diagnostic \
                information (helpful for debugging, and sometimes learning)
        """
        pass

    @abstractmethod
    def seed(self, seed: Optional[Union[int, List[int]]] = None) -> List[int]:
        """Set the seed for all environments.

        Accept ``None``, an int (which will extend ``i`` to
        ``[i, i + 1, i + 2, ...]``) or a list.

        :return: The list of seeds used in this env's random number \
        generators. The first value in the list should be the "main" seed, or \
        the value which a reproducer pass to "seed".
        """
        pass

    @abstractmethod
    def render(self, **kwargs) -> None:
        """Render all of the environments."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close all of the environments.

        Environments will automatically close() themselves when garbage
        collected or when the program exits.
        """
        pass
