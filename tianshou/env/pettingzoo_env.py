from abc import ABC
from typing import Any, Dict, List, Tuple, Union

import gym.spaces
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils.wrappers import BaseWrapper


class PettingZooEnv(AECEnv, ABC):
    """The interface for petting zoo environments.

    Multi-agent environments must be wrapped as
    :class:`~tianshou.env.PettingZooEnv`. Here is the usage:
    ::

        env = PettingZooEnv(...)
        # obs is a dict containing obs, agent_id, and mask
        obs = env.reset()
        action = policy(obs)
        obs, rew, done, info = env.step(action)
        env.close()

    The available action's mask is set to True, otherwise it is set to False.
    Further usage can be found at :ref:`marl_example`.
    """

    def __init__(self, env: BaseWrapper):
        super().__init__()
        self.env = env
        # agent idx list
        self.agents = self.env.possible_agents
        self.agent_idx = {}
        for i, agent_id in enumerate(self.agents):
            self.agent_idx[agent_id] = i

        self.rewards = [0] * len(self.agents)

        # Get first observation space, assuming all agents have equal space
        self.observation_space: Any = self.env.observation_space(self.agents[0])

        # Get first action space, assuming all agents have equal space
        self.action_space: Any = self.env.action_space(self.agents[0])

        assert all(self.env.observation_space(agent) == self.observation_space
                   for agent in self.agents), \
            "Observation spaces for all agents must be identical. Perhaps " \
            "SuperSuit's pad_observations wrapper can help (useage: " \
            "`supersuit.aec_wrappers.pad_observations(env)`"

        assert all(self.env.action_space(agent) == self.action_space
                   for agent in self.agents), \
            "Action spaces for all agents must be identical. Perhaps " \
            "SuperSuit's pad_action_space wrapper can help (useage: " \
            "`supersuit.aec_wrappers.pad_action_space(env)`"

        self.reset()

    def reset(self, *args: Any, **kwargs: Any) -> Union[dict, Tuple[dict, dict]]:
        self.env.reset(*args, **kwargs)
        observation, _, _, info = self.env.last(self)
        if isinstance(observation, dict) and 'action_mask' in observation:
            observation_dict = {
                'agent_id': self.env.agent_selection,
                'obs': observation['observation'],
                'mask':
                [True if obm == 1 else False for obm in observation['action_mask']]
            }
        else:
            if isinstance(self.action_space, gym.spaces.Discrete):
                observation_dict = {
                    'agent_id': self.env.agent_selection,
                    'obs': observation,
                    'mask': [True] * self.env.action_space(self.env.agent_selection).n
                }
            else:
                observation_dict = {
                    'agent_id': self.env.agent_selection,
                    'obs': observation,
                }

        if "return_info" in kwargs and kwargs["return_info"]:
            return observation_dict, info
        else:
            return observation_dict

    def step(self, action: Any) -> Tuple[Dict, List[int], bool, Dict]:
        self.env.step(action)
        observation, rew, done, info = self.env.last()
        if isinstance(observation, dict) and 'action_mask' in observation:
            obs = {
                'agent_id': self.env.agent_selection,
                'obs': observation['observation'],
                'mask':
                [True if obm == 1 else False for obm in observation['action_mask']]
            }
        else:
            if isinstance(self.action_space, gym.spaces.Discrete):
                obs = {
                    'agent_id': self.env.agent_selection,
                    'obs': observation,
                    'mask': [True] * self.env.action_space(self.env.agent_selection).n
                }
            else:
                obs = {'agent_id': self.env.agent_selection, 'obs': observation}

        for agent_id, reward in self.env.rewards.items():
            self.rewards[self.agent_idx[agent_id]] = reward
        return obs, self.rewards, done, info

    def close(self) -> None:
        self.env.close()

    def seed(self, seed: Any = None) -> None:
        try:
            self.env.seed(seed)
        except NotImplementedError:
            self.env.reset(seed=seed)

    def render(self, mode: str = "human") -> Any:
        return self.env.render(mode)
