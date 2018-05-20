import logging

__all__ = [
    'full_return',
    'nstep_return',
    'nstep_q_return',
    'ddpg_return',
]


STATE = 0
ACTION = 1
REWARD = 2
DONE = 3


# TODO: add discount_factor... maybe make it to be a global config?
def full_return(buffer, indexes=None):
    """
    Naively compute full undiscounted return on episodic data, :math:`G_t = \sum_{t=0}^T r_t`.
    This function will print a warning when some of the episodes
    in ``buffer`` has not yet terminated.

    :param buffer: A :class:`tianshou.data.data_buffer`.
    :param indexes: Optional. Indexes of data points on which the full return should be computed.
        If not set, it defaults to all the data points in ``buffer``.
        Note that if it's the index of a sampled minibatch, it doesn't have to be in order within
        each episode.

    :return: A dict with key 'return' and value the computed returns corresponding to ``indexes``.
    """
    indexes = indexes or buffer.index
    raw_data = buffer.data

    returns = []
    for i_episode in range(len(indexes)):
        index_this = indexes[i_episode]
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


class nstep_return:
    """
    Compute the n-step return from n-step rewards and bootstrapped state value function V(s),
    :math:`V(s_t) = r_t + \gamma r_{t+1} + ... + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n})`.

    :param n: An int. The number of steps to lookahead, where :math:`n=1` will directly apply V(s) to
        the next observation, as in the above equation.
    :param value_function: A :class:`tianshou.core.value_function.StateValue`. The V(s) as in the
        above equation
    :param return_advantage: Optional. A bool defaulting to ``False``. If ``True`` than this callable
        also returns the advantage function
        :math:`A(s_t) = r_t + \gamma r_{t+1} + ... + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n}) - V(s_t)` when called.
    :param discount_factor: Optional. A float in range :math:`[0, 1]` defaulting to 0.99. The discount
        factor :math:`\gamma` as in the above equation.
    """
    def __init__(self, n, value_function, return_advantage=False, discount_factor=0.99):
        self.n = n
        self.value_function = value_function
        self.return_advantage = return_advantage
        self.discount_factor = discount_factor

    def __call__(self, buffer, indexes=None):
        """
        :param buffer: A :class:`tianshou.data.data_buffer`.
        :param indexes: Optional. Indexes of data points on which the specified return should be computed.
            If not set, it defaults to all the data points in ``buffer``.
            Note that if it's the index of a sampled minibatch, it doesn't have to be in order within
            each episode.

        :return: A dict with key 'return' and value the computed returns corresponding to ``indexes``.
            If ``return_advantage`` set to ``True`` then also a key 'advantage' and value the corresponding
            advantages.
        """
        indexes = indexes or buffer.index
        episodes = buffer.data
        returns = []
        advantages = []

        for i_episode in range(len(indexes)):
            index_this = indexes[i_episode]
            if index_this:
                episode = episodes[i_episode]
                returns_this = []
                advantages_this = []

                for i in index_this:
                    current_discount_factor = 1.
                    last_frame_index = i
                    return_ = 0.
                    for last_frame_index in range(i, min(len(episode), i + self.n)):
                        return_ += current_discount_factor * episode[last_frame_index][REWARD]
                        current_discount_factor *= self.discount_factor
                        if episode[last_frame_index][DONE]:
                            break
                    if not episode[last_frame_index][DONE]:
                        state = episode[last_frame_index + 1][STATE]
                        v_sT = self.value_function.eval_value(state[None])
                        return_ += current_discount_factor * v_sT
                    returns_this.append(return_)
                    if self.return_advantage:
                        v_s0 = self.value_function.eval_value(episode[i][STATE][None])
                        advantages_this.append(return_ - v_s0)

                returns.append(returns_this)
                advantages.append(advantages_this)
            else:
                returns.append([])
                advantages.append([])

        if self.return_advantage:
            return {'return': returns, 'advantage':advantages}
        else:
            return {'return': returns}


class ddpg_return:
    """
    Compute the return as in `DDPG <https://arxiv.org/pdf/1509.02971.pdf>`_,
    :math:`G_t = r_t + \gamma Q'(s_{t+1}, \mu'(s_{t+1}))`, where :math:`Q'` and :math:`\mu'` are the
    target networks.

    :param actor: A :class:`tianshou.core.policy.Deterministic`. A deterministic policy.
    :param critic: A :class:`tianshou.core.value_function.ActionValue`. An action value function Q(s, a).
    :param use_target_network: Optional. A bool defaulting to ``True``. Whether to use the target networks
        in the above equation.
    :param discount_factor: Optional. A float in range :math:`[0, 1]` defaulting to 0.99. The discount
        factor :math:`\gamma` as in the above equation.
    """
    def __init__(self, actor, critic, use_target_network=True, discount_factor=0.99):
        self.actor = actor
        self.critic = critic
        self.use_target_network = use_target_network
        self.discount_factor = discount_factor

    def __call__(self, buffer, indexes=None):
        """
        :param buffer: A :class:`tianshou.data.data_buffer`.
        :param indexes: Optional. Indexes of data points on which the specified return should be computed.
            If not set, it defaults to all the data points in ``buffer``.
            Note that if it's the index of a sampled minibatch, it doesn't have to be in order within
            each episode.

        :return: A dict with key 'return' and value the computed returns corresponding to ``indexes``.
        """
        indexes = indexes or buffer.index
        episodes = buffer.data
        returns = []

        for i_episode in range(len(indexes)):
            index_this = indexes[i_episode]
            if index_this:
                episode = episodes[i_episode]
                returns_this = []

                for i in index_this:
                    return_ = episode[i][REWARD]
                    if not episode[i][DONE]:
                        if self.use_target_network:
                            state = episode[i + 1][STATE][None]
                            action = self.actor.eval_action_old(state)
                            q_value = self.critic.eval_value_old(state, action)
                            return_ += self.discount_factor * q_value
                        else:
                            state = episode[i + 1][STATE][None]
                            action = self.actor.eval_action(state)
                            q_value = self.critic.eval_value(state, action)
                            return_ += self.discount_factor * q_value

                    returns_this.append(return_)

                returns.append(returns_this)
            else:
                returns.append([])

        return {'return': returns}


class nstep_q_return:
    """
    Compute the n-step return for Q-learning targets,
    :math:`G_t = r_t + \gamma \max_a Q'(s_{t+1}, a)`.

    :param n: An int. The number of steps to lookahead, where :math:`n=1` will directly apply :math:`Q'(s, \*)` to
        the next observation, as in the above equation.
    :param action_value: A :class:`tianshou.core.value_function.DQN`. The :math:`Q'(s, \*)` as in the
        above equation.
    :param use_target_network: Optional. A bool defaulting to ``True``. Whether to use the target networks
        in the above equation.
    :param discount_factor: Optional. A float in range :math:`[0, 1]` defaulting to 0.99. The discount
        factor :math:`\gamma` as in the above equation.
    """
    def __init__(self, n, action_value, use_target_network=True, discount_factor=0.99):
        self.n = n
        self.action_value = action_value
        self.use_target_network = use_target_network
        self.discount_factor = discount_factor

    def __call__(self, buffer, indexes=None):
        """
        :param buffer: A :class:`tianshou.data.data_buffer`.
        :param indexes: Optional. Indexes of data points on which the full return should be computed.
            If not set, it defaults to all the data points in ``buffer``.
            Note that if it's the index of a sampled minibatch, it doesn't have to be in order within
            each episode.

        :return: A dict with key 'return' and value the computed returns corresponding to ``indexes``.
        """
        indexes = indexes or buffer.index
        episodes = buffer.data
        returns = []

        for episode_index in range(len(indexes)):
            index = indexes[episode_index]
            if index:
                episode = episodes[episode_index]
                episode_q = []

                for i in index:
                    current_discount_factor = 1.
                    last_frame_index = i
                    target_q = 0.
                    for last_frame_index in range(i, min(len(episode), i + self.n)):
                        target_q += current_discount_factor * episode[last_frame_index][REWARD]
                        current_discount_factor *= self.discount_factor
                        if episode[last_frame_index][DONE]:
                            break
                    if not episode[last_frame_index][DONE]:  # not done will definitely have one frame later
                        state = episode[last_frame_index + 1][STATE]
                        if self.use_target_network:
                            # [None] adds one dimension to the beginning
                            qpredict = self.action_value.eval_value_all_actions_old(state[None])
                        else:
                            qpredict = self.action_value.eval_value_all_actions(state[None])
                        target_q += current_discount_factor * max(qpredict[0])
                    episode_q.append(target_q)

                returns.append(episode_q)
            else:
                returns.append([])

        return {'return': returns}


class gae_lambda:
    """
    Generalized Advantage Estimation (Schulman, 15) to compute advantage. To be implemented.
    """
    def __init__(self, T, value_function):
        self.T = T
        self.value_function = value_function

        raise NotImplementedError()

    def __call__(self, buffer, indexes=None):
        """
        To be implemented

        :param buffer: A :class:`tianshou.data.data_buffer`.
        :param indexes: Optional. Indexes of data points on which the full return should be computed.
            If not set, it defaults to all the data points in ``buffer``.
            Note that if it's the index of a sampled minibatch, it doesn't have to be in order within
            each episode.

        :return: A dict with key 'advantage' and value the computed advantages corresponding to ``indexes``.
        """
        raise NotImplementedError()
