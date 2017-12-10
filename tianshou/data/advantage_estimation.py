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
        if episode_start_flags[i] or (i == num_timesteps - 1): # found the start of next episode or the end of all episodes
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