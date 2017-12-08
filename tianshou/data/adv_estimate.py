import numpy as np


def full_return(raw_data):
    """
    naively compute full return
    :param raw_data: dict of specified keys and values.
    """
    obs = raw_data['obs']
    acs = raw_data['acs']
    rews = raw_data['rews']
    news = raw_data['news']
    num_timesteps = rews.shape[0]

    data = {}
    data['obs'] = obs
    data['acs'] = acs

    Gts = rews.copy()
    episode_start_idx = 0
    for i in range(1, num_timesteps):
        if news[i] or (i == num_timesteps - 1): # found one full episode
            if i < rews.shape[0] - 1:
                t = i - 1
            else:
                t = i
            Gt = 0
            while t >= episode_start_idx:
                Gt += rews[t]
                Gts[t] = Gt
                t -= 1

            episode_start_idx = i

    data['Gts'] = Gts

    return data