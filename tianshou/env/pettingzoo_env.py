from pettingzoo.utils.env import AECEnv


class PettingZooEnv(AECEnv):

    def __init__(self, env):
        self.env = env
        # agent idx list
        self.agents = self.env.possible_agents
        self.agent_idx = {}
        for i, agent_id in enumerate(self.agents):
            self.agent_idx[agent_id] = i
        # Get dictionaries of obs_spaces and act_spaces
        self.observation_spaces = self.env.observation_spaces
        self.action_spaces = self.env.action_spaces

        self.rewards = [0] * len(self.agents)

        # Get first observation space, assuming all agents have equal space
        self.observation_space = self.observation_spaces[self.agents[0]]

        # Get first action space, assuming all agents have equal space
        self.action_space = self.action_spaces[self.agents[0]]

        assert all(obs_space == self.observation_space
                   for obs_space
                   in self.env.observation_spaces.values()), \
            "Observation spaces for all agents must be identical. Perhaps " \
            "SuperSuit's pad_observations wrapper can help (useage: " \
            "`supersuit.aec_wrappers.pad_observations(env)`"

        assert all(act_space == self.action_space
                   for act_space in self.env.action_spaces.values()), \
            "Action spaces for all agents must be identical. Perhaps " \
            "SuperSuit's pad_action_space wrapper can help (useage: " \
            "`supersuit.aec_wrappers.pad_action_space(env)`"

        self.reset()

    def reset(self):
        self.env.reset()
        observation = self.env.observe(self.env.agent_selection)
        if isinstance(observation, dict) and 'action_mask' in observation:
            return {
                'agent_id': self.env.agent_selection,
                'obs': observation['observation'],
                'mask': observation['action_mask']
            }
        else:
            return {
                'agent_id': self.env.agent_selection,
                'obs': observation,
                'mask': [True] * self.action_spaces[self.env.agent_selection].n
            }

    def step(self, action):
        self.env.step(action)
        observation, rew, done, info = self.env.last()
        if isinstance(observation, dict) and 'action_mask' in observation:
            obs = {
                'agent_id': self.env.agent_selection,
                'obs': observation['observation'],
                'mask': observation['action_mask']
            }
        else:
            obs = {
                'agent_id': self.env.agent_selection,
                'obs': observation,
                'mask': [True] * self.action_spaces[self.env.agent_selection].n
            }
        
        for agent_id, reward in self.env.rewards.items():
            self.rewards[self.agent_idx[agent_id]] = reward
        return obs, self.rewards, done, info

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed)

    def render(self, mode="human"):
        return self.env.render(mode)
