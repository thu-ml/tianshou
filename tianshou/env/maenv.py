import gym
import numpy as np
from typing import Any, Dict, Tuple
from abc import ABC, abstractmethod


class MultiAgentEnv(ABC, gym.Env):
    """The interface for multi-agent environments.

    Multi-agent environments must be wrapped as
    :class:`~tianshou.env.MultiAgentEnv`. Here is the usage:
    ::

        env = MultiAgentEnv(...)
        # obs is a dict containing obs, agent_id, and mask
        obs = env.reset()
        action = policy(obs)
        obs, rew, done, info = env.step(action)
        env.close()

    The available action's mask is set to 1, otherwise it is set to 0. Further
    usage can be found at :ref:`marl_example`.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def reset(self) -> dict:
        """Reset the state.

        Return the initial state, first agent_id, and the initial action set,
        for example, ``{'obs': obs, 'agent_id': agent_id, 'mask': mask}``.
        """
        pass

    @abstractmethod
    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
        """Run one timestep of the environment’s dynamics.

        When the end of episode is reached, you are responsible for calling
        reset() to reset the environment’s state.

        Accept action and return a tuple (obs, rew, done, info).

        :param numpy.ndarray action: action provided by a agent.

        :return: A tuple including four items:

            * ``obs`` a dict containing obs, agent_id, and mask, which means \
                that it is the ``agent_id`` player's turn to play with ``obs``\
                observation and ``mask``.
            * ``rew`` a numpy.ndarray, the amount of rewards returned after \
                previous actions. Depending on the specific environment, this \
                can be either a scalar reward for current agent or a vector \
                reward for all the agents.
            * ``done`` a numpy.ndarray, whether the episode has ended, in \
                which case further step() calls will return undefined results
            * ``info`` a numpy.ndarray, contains auxiliary diagnostic \
                information (helpful for debugging, and sometimes learning)
        """
        pass
