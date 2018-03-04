import numpy as np
import logging
import itertools

from .replay_buffer.base import ReplayBufferBase

class DataCollector(object):
    """
    a utility class to manage the interaction between buffer and advantage_estimation
    """
    def __init__(self, env, policy, data_buffer, process_functions, managed_networks):
        self.env = env
        self.policy = policy
        self.data_buffer = data_buffer
        self.process_functions = process_functions
        self.managed_networks = managed_networks

        self.data = {}
        self.data_batch = {}

        self.required_placeholders = {}
        for net in self.managed_networks:
            self.required_placeholders.update(net.managed_placeholders)
        self.require_advantage = 'advantage' in self.required_placeholders.keys()

        if isinstance(self.data_buffer, ReplayBufferBase):  # process when sampling minibatch
            self.process_mode = 'sample'
        else:
            self.process_mode = 'full'

        self.current_observation = self.env.reset()

    def collect(self, num_timesteps=1, num_episodes=0, my_feed_dict={}):
        assert sum([num_timesteps > 0, num_episodes > 0]) == 1,\
            "One and only one collection number specification permitted!"

        if num_timesteps > 0:
            for _ in range(num_timesteps):
                action = self.policy.act(self.current_observation, my_feed_dict=my_feed_dict)
                next_observation, reward, done, _ = self.env.step(action)
                self.data_buffer.add((self.current_observation, action, reward, done))
                self.current_observation = next_observation

        if num_episodes > 0:
            for _ in range(num_episodes):
                observation = self.env.reset()
                done = False
                while not done:
                    action = self.policy.act(observation, my_feed_dict=my_feed_dict)
                    next_observation, reward, done, _ = self.env.step(action)
                    self.data_buffer.add((observation, action, reward, done))
                    observation = next_observation

        if self.process_mode == 'full':
            for processor in self.process_functions:
                self.data.update(processor(self.data_buffer))

    def next_batch(self, batch_size, standardize_advantage=True):
        sampled_index = self.data_buffer.sample(batch_size)
        if self.process_mode == 'sample':
            for processor in self.process_functions:
                self.data_batch.update(processor(self.data_buffer, index=sampled_index))

        # flatten rank-2 list to numpy array, construct feed_dict
        feed_dict = {}
        frame_key_map = {'observation': 0, 'action': 1, 'reward': 2, 'done_flag': 3}
        for key, placeholder in self.required_placeholders.items():
            if key in frame_key_map.keys():  # access raw_data
                frame_index = frame_key_map[key]
                flattened = []
                for index_episode, data_episode in zip(sampled_index, self.data_buffer.data):
                    for i in index_episode:
                        flattened.append(data_episode[i][frame_index])
                feed_dict[placeholder] = np.array(flattened)
            elif key in self.data_batch.keys():  # access processed minibatch data
                flattened = list(itertools.chain.from_iterable(self.data_batch[key]))
                feed_dict[placeholder] = np.array(flattened)
            elif key in self.data.keys():  # access processed full data
                flattened = [0.] * batch_size  # float
                i_in_batch = 0
                for index_episode, data_episode in zip(sampled_index, self.data[key]):
                    for i in index_episode:
                        flattened[i_in_batch] = data_episode[i]
                        i_in_batch += 1
                feed_dict[placeholder] = np.array(flattened)
            else:
                raise TypeError('Placeholder {} has no value to feed!'.format(str(placeholder.name)))

        if standardize_advantage:
            if self.require_advantage:
                advantage_value = feed_dict[self.required_placeholders['advantage']]
                advantage_mean = np.mean(advantage_value)
                advantage_std = np.std(advantage_value)
                if advantage_std < 1e-3:
                    logging.warning('advantage_std too small (< 1e-3) for advantage standardization. may cause numerical issues')
                feed_dict[self.required_placeholders['advantage']] = (advantage_value - advantage_mean) / advantage_std

        return feed_dict

    def statistics(self):
        pass