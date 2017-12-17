import numpy as np


def full_return(raw_data):
    """
    naively compute full return
    :param raw_data: dict of specified keys and values.
    """
    observations = raw_data['observations']
    actions = raw_data['actions']
    rewards = raw_data['rewards']
    episode_start_flags = raw_data['episode_start_flags']
    num_timesteps = rewards.shape[0]

    data = {}
    data['observations'] = observations
    data['actions'] = actions

    returns = rewards.copy()
    episode_start_idx = 0
    for i in range(1, num_timesteps):
        if episode_start_flags[i] or (
                i == num_timesteps - 1):  # found the start of next episode or the end of all episodes
            if i < rewards.shape[0] - 1:
                t = i - 1
            else:
                t = i
            Gt = 0
            while t >= episode_start_idx:
                Gt += rewards[t]
                returns[t] = Gt
                t -= 1

            episode_start_idx = i

    data['returns'] = returns

    return data


class QLearningTarget:
    def __init__(self, policy, gamma):
        self._policy = policy
        self._gamma = gamma

    def __call__(self, raw_data):
        data = dict()
        observations = list()
        actions = list()
        rewards = list()
        wi = list()
        all_data, data_wi, data_index = raw_data

        for i in range(0, all_data.shape[0]):
            current_data = all_data[i]
            current_wi = data_wi[i]
            current_index = data_index[i]
            observations.append(current_data['observation'])
            actions.append(current_data['action'])
            next_max_qvalue = np.max(self._policy.values(current_data['observation']))
            current_qvalue = self._policy.values(current_data['previous_observation'])[current_data['previous_action']]
            reward = current_data['reward'] + next_max_qvalue - current_qvalue
            rewards.append(reward)
            wi.append(current_wi)

        data['observations'] = np.array(observations)
        data['actions'] = np.array(actions)
        data['rewards'] = np.array(rewards)

        return data
