import gym
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class MultiAgentEnv(ABC, gym.Env):
    """The interface for multi-agent environments.
    Multi-agent environments must be wrapped as
    :class:`~tianshou.env.MultiAgentEnv`. Here are some usages:
    ::

        env = MultiAgentEnv(...)
        # obs is a dict containing obs, agent_id, and mask
        obs = env.reset()
        action = policy(obs)
        obs, rew, done, info = env.step(action)
        env.close()

    Further usage can be found at :ref:`marl_example`.
    """

    def __init__(self, env) -> None:
        self.env = env
        self._obs = None
        self._rew = None
        self._done = None
        self._info = None

    @abstractmethod
    def reset(self) -> dict:
        """Reset the state, return the initial state, first agent_id,
        and the initial action set.
        """
        pass

    @abstractmethod
    def step(self, action: np.ndarray
             ) -> Tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
        """Run one timestep of the environment’s dynamics. When the end of
        episode is reached, you are responsible for calling reset() to reset
        the environment’s state.

        Accept action and return a tuple (obs, rew, done, info).

        :param numpy.ndarray action: action provided by a agent.

        :return: A tuple including four items:

            * ``obs`` a dict containing obs, agent_id, and mask, \
                which means that it is the ``agent_id`` player's turn to \
                play with ``obs`` observation and ``mask``.
            * ``rew`` a numpy.ndarray, the amount of rewards returned after \
                previous actions. Depending on the specific environment, \
                this can be either a scalar reward for current agent or a \
                vector reward for all the agents.
            * ``done`` a numpy.ndarray, whether the episode has ended, in \
                which case further step() calls will return undefined results
            * ``info`` a numpy.ndarray, contains auxiliary diagnostic \
                information (helpful for debugging, and sometimes learning)
        """
        pass

    @abstractmethod
    def seed(self, seed: Optional[int] = None) -> int:
        """Set the seed for the environment.

        :return: The seed used in this env's random number \
        generators.
        """
        pass

    @abstractmethod
    def render(self, **kwargs) -> None:
        """Render the environment."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the environment.

        Environments will automatically close() themselves when garbage
        collected or when the program exits.
        """
        pass
