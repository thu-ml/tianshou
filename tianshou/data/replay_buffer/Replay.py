import tianshou.data.replay_buffer.naive as naive
import tianshou.data.replay_buffer.rank_based as rank_based
import tianshou.data.replay_buffer.proportional as proportional
import numpy as np
import tensorflow as tf
from tianshou.data import utils
import logging


class Replay(object):
    def __init__(self, replay_memory, env, pi, reward_processors, networks):
        self._replay_memory = replay_memory
        self._env = env
        self._pi = pi
        self._reward_processors = reward_processors
        self._networks = networks

        self._required_placeholders = {}
        for net in self._networks:
            self._required_placeholders.update(net.managed_placeholders)
        self._require_advantage = 'advantage' in self._required_placeholders.keys()
        self._collected_data = list()

        self._is_first_collect = True

    def _begin_act(self, exploration):
        while self._is_first_collect:
            self._observation = self._env.reset()
            self._action = self._pi.act(self._observation, exploration)
            self._observation, reward, done, _ = self._env.step(self._action)
            if not done:
                self._is_first_collect = False

    def collect(self, nums, exploration=None):
        """
        collect data for replay memory and update the priority according to the given data.
        store the previous action, previous observation, reward, action, observation in the replay memory.
        """
        sess = tf.get_default_session()
        self._collected_data = list()

        for _ in range(0, nums):
            if self._is_first_collect:
                self._begin_act(exploration)

            current_data = dict()
            current_data['previous_action'] = self._action
            current_data['previous_observation'] = self._observation
            self._action = self._pi.act(self._observation, exploration)
            self._observation, reward, done, _ = self._env.step(self._action)
            current_data['action'] = self._action
            current_data['observation'] = self._observation
            current_data['reward'] = reward
            current_data['end_flag'] = done
            self._replay_memory.add(current_data)
            self._collected_data.append(current_data)
            if done:
                self._begin_act(exploration)

    # I don't know what statistics should replay memory provide, for replay memory only saves discrete data
    def statistics(self):
        """
        compute the statistics of the current sampled paths
        :return:
        """
        raw_data = dict(zip(self._collected_data[0], zip(*[d.values() for d in self._collected_data])))
        rewards = np.array(raw_data['reward'])
        episode_start_flags = np.array(raw_data['end_flag'])
        num_timesteps = rewards.shape[0]

        returns = []
        episode_lengths = []
        max_return = 0
        num_episodes = 1
        episode_start_idx = 0
        for i in range(1, num_timesteps):
            if episode_start_flags[i] or (
                    i == num_timesteps - 1):  # found the start of next episode or the end of all episodes
                if episode_start_flags[i]:
                    num_episodes += 1
                if i < rewards.shape[0] - 1:
                    t = i - 1
                else:
                    t = i
                Gt = 0
                episode_lengths.append(t - episode_start_idx)
                while t >= episode_start_idx:
                    Gt += rewards[t]
                    t -= 1

                returns.append(Gt)
                if Gt > max_return:
                    max_return = Gt
                episode_start_idx = i

        print('AverageReturn: {}'.format(np.mean(returns)))
        print('StdReturn    : {}'.format(np.std(returns)))
        print('NumEpisodes  : {}'.format(num_episodes))
        print('MinMaxReturns: {}..., {}'.format(np.sort(returns)[:3], np.sort(returns)[-3:]))
        print('AverageLength: {}'.format(np.mean(episode_lengths)))
        print('MinMaxLengths: {}..., {}'.format(np.sort(episode_lengths)[:3], np.sort(episode_lengths)[-3:]))

    def next_batch(self, batch_size, global_step=0, standardize_advantage=True):
        """
        collect a batch of data from replay buffer, update the priority and calculate the necessary statistics for
        updating q value network.
        :param batch_size: int batch size.
        :param global_step: int training global step.
        :return: a batch of data, with target storing the target q value and wi, rewards storing the coefficient
        for gradient of q value network.
        """

        feed_dict = {}
        is_first = True

        for _ in range(0, batch_size):
            current_datas, current_wis, current_indexs = \
                self._replay_memory.sample(
                    {'batch_size': 1, 'global_step': global_step})
            current_data = current_datas[0]
            current_wi = current_wis[0]
            current_index = current_indexs[0]
            current_processed_data = {}
            for processors in self._reward_processors:
                current_processed_data.update(processors(current_data))

            for key, placeholder in self._required_placeholders.items():
                found, data_key = utils.internal_key_match(key, current_data.keys())
                if found:
                    if is_first:
                        feed_dict[placeholder] = np.array([current_data[data_key]])
                    else:
                        feed_dict[placeholder] = np.append(feed_dict[placeholder], np.array([current_data[data_key]]), 0)
                else:
                    found, data_key = utils.internal_key_match(key, current_processed_data.keys())
                    if found:
                        if is_first:
                            feed_dict[placeholder] = np.array(current_processed_data[data_key])
                        else:
                            feed_dict[placeholder] = np.append(feed_dict[placeholder],
                                                               np.array(current_processed_data[data_key]), 0)
                    else:
                        raise TypeError('Placeholder {} has no value to feed!'.format(str(placeholder.name)))
            next_max_qvalue = np.max(self._networks[-1].eval_value_all_actions(
                current_data['observation'].reshape((1,) + current_data['observation'].shape)))
            current_qvalue = self._networks[-1].eval_value_all_actions(
                current_data['previous_observation']
                    .reshape((1,) + current_data['previous_observation'].shape))[0, current_data['previous_action']]
            reward = current_data['reward'] + next_max_qvalue - current_qvalue
            import math
            self._replay_memory.update_priority([current_index], [math.fabs(reward)])
            if is_first:
                is_first = False

        if standardize_advantage:
            if self._require_advantage:
                advantage_value = feed_dict[self._required_placeholders['advantage']]
                advantage_mean = np.mean(advantage_value)
                advantage_std = np.std(advantage_value)
                if advantage_std < 1e-3:
                    logging.warning(
                        'advantage_std too small (< 1e-3) for advantage standardization. may cause numerical issues')
                feed_dict[self._required_placeholders['advantage']] = (advantage_value - advantage_mean) / advantage_std
        return feed_dict