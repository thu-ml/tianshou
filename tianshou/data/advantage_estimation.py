import logging


STATE = 0
ACTION = 1
REWARD = 2
DONE = 3

# modified for new interfaces
def full_return(buffer, index=None):
    """
    naively compute full return
    :param buffer: buffer with property index and data. index determines the current content in `buffer`.
    :param index: (sampled) index to be computed. Defaults to all the data in `buffer`. Not necessarily in order within
                  each episode.
    :return: dict with key 'return' and value the computed returns corresponding to `index`.
    """
    index = index or buffer.index
    raw_data = buffer.data

    returns = []
    for i_episode in range(len(index)):
        index_this = index[i_episode]
        if index_this:
            episode = raw_data[i_episode]
            if not episode[-1][DONE]:
                logging.warning('Computing full return on episode {} which is not terminated.'.format(i_episode))

            episode_length = len(episode)
            returns_episode = [0.] * episode_length
            returns_this = [0.] * len(index_this)
            return_ = 0.
            index_min = min(index_this)
            for i, frame in zip(range(episode_length - 1, index_min - 1, -1), reversed(episode[index_min:])):
                return_ += frame[REWARD]
                returns_episode[i] = return_

            for i in range(len(index_this)):
                returns_this[i] = returns_episode[index_this[i]]

            returns.append(returns_this)
        else:
            returns.append([])

    return {'return': returns}


class gae_lambda:
    """
    Generalized Advantage Estimation (Schulman, 15) to compute advantage
    """
    def __init__(self, T, value_function):
        self.T = T
        self.value_function = value_function

    def __call__(self, buffer, index=None):
        """
        :param buffer: buffer with property index and data. index determines the current content in `buffer`.
        :param index: (sampled) index to be computed. Defaults to all the data in `buffer`. Not necessarily in order within
                      each episode.
        :return: dict with key 'advantage' and value the computed advantages corresponding to `index`.
        """
        pass


class nstep_return:
    """
    compute the n-step return from n-step rewards and bootstrapped value function
    """
    def __init__(self, n, value_function):
        self.n = n
        self.value_function = value_function

    def __call__(self, buffer, index=None):
        """
        :param buffer: buffer with property index and data. index determines the current content in `buffer`.
        :param index: (sampled) index to be computed. Defaults to all the data in `buffer`. Not necessarily in order within
                      each episode.
        :return: dict with key 'return' and value the computed returns corresponding to `index`.
        """
        pass


class ddpg_return:
    """
    compute the return as in DDPG. this seems to have to be special
    """
    def __init__(self, actor, critic, use_target_network=True):
        self.actor = actor
        self.critic = critic
        self.use_target_network = use_target_network

    def __call__(self, buffer, index=None):
        """
        :param buffer: buffer with property index and data. index determines the current content in `buffer`.
        :param index: (sampled) index to be computed. Defaults to all the data in `buffer`. Not necessarily in order within
                      each episode.
        :return: dict with key 'return' and value the computed returns corresponding to `index`.
        """
        pass


class nstep_q_return:
    """
    compute the n-step return for Q-learning targets
    """
    def __init__(self, n, action_value, use_target_network=True):
        self.n = n
        self.action_value = action_value
        self.use_target_network = use_target_network

    def __call__(self, buffer, index=None):
        """
        :param buffer: buffer with property index and data. index determines the current content in `buffer`.
        :param index: (sampled) index to be computed. Defaults to all the data in `buffer`. Not necessarily in order within
                      each episode.
        :return: dict with key 'return' and value the computed returns corresponding to `index`.
        """
        pass
