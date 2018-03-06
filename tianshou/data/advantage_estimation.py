import logging
import tensorflow as tf
import numpy as np

STATE = 0
ACTION = 1
REWARD = 2
DONE = 3

# modified for new interfaces
def full_return(buffer, indexes=None):
    """
    naively compute full return
    :param buffer: buffer with property index and data. index determines the current content in `buffer`.
    :param indexes: (sampled) index to be computed. Defaults to all the data in `buffer`. Not necessarily in order within
                  each episode.
    :return: dict with key 'return' and value the computed returns corresponding to `index`.
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

    # TODO : we should transfer the tf -> numpy/python -> tf into a monolithic compute graph in tf
    def __call__(self, buffer, indexes=None):
        """
        :param buffer: buffer with property index and data. index determines the current content in `buffer`.
        :param index: (sampled) index to be computed. Defaults to all the data in `buffer`. Not necessarily in order within
                      each episode.
        :return: dict with key 'return' and value the computed returns corresponding to `index`.
        """
        qvalue = self.action_value._value_tensor_all_actions
        indexes = indexes or buffer.index
        episodes = buffer.data
        discount_factor = 0.99
        returns = []

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            for episode_index in range(len(indexes)):
                index = indexes[episode_index]
                if index:
                    episode = episodes[episode_index]
                    episode_q = []

                    for i in index:
                        current_discount_factor = 1
                        last_frame_index = i
                        target_q = episode[i][REWARD]
                        for lfi in range(i, min(len(episode), i + self.n + 1)):
                            if episode[lfi][DONE]:
                                break
                            target_q += current_discount_factor * episode[lfi][REWARD]
                            current_discount_factor *= discount_factor
                            last_frame_index = lfi
                        if last_frame_index > i:
                            state = episode[last_frame_index][STATE]
                            # the shape of qpredict is [batch_size, action_dimension]
                            qpredict = sess.run(qvalue, feed_dict={self.action_value.managed_placeholders['observation']:
                                                                        state.reshape(1, state.shape[0])})
                            target_q += current_discount_factor * max(qpredict[0])
                        episode_q.append(target_q)

                    returns.append(episode_q)
                else:
                    returns.append([])
        return {'return': returns}
