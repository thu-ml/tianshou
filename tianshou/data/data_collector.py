import numpy as np
import logging
import itertools

from .data_buffer.replay_buffer_base import ReplayBufferBase
from .data_buffer.batch_set import BatchSet
from .utils import internal_key_match
from ..core.policy.deterministic import Deterministic


class DataCollector(object):
    """
    A utility class to manage the data flow during the interaction between the policy and the environment.
    It stores data into ``data_buffer``, processes the reward signals and returns the feed_dict for
    tf graph running.

    :param env: An environment.
    :param policy: A :class:`tianshou.core.policy`.
    :param data_buffer: A :class:`tianshou.data.data_buffer`.
    :param process_functions: A list of callables in :mod:`tianshou.data.advantage_estimation`
        to process rewards.
    :param managed_networks: A list of networks of :class:`tianshou.core.policy` and/or
        :class:`tianshou.core.value_function`. The networks you want this class to manage. This class
        will automatically generate the feed_dict for all the placeholders in the ``managed_placeholders``
        of all networks in this list.
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
        self.step_count_this_episode = 0

    def collect(self, num_timesteps=0, num_episodes=0, my_feed_dict={}, auto_clear=True, episode_cutoff=None):
        """
        Collect data in the environment using ``self.policy``.

        :param num_timesteps: An int specifying the number of timesteps to act. It defaults to 0 and either
            ``num_timesteps`` or ``num_episodes`` could be set but not both.
        :param num_episodes: An int specifying the number of episodes to act. It defaults to 0 and either
            ``num_timesteps`` or ``num_episodes`` could be set but not both.
        :param my_feed_dict: Optional. A dict defaulting to empty.
            Specifies placeholders such as dropout and batch_norm except observation and action.
        :param auto_clear: Optional. A bool defaulting to ``True``. If ``True`` then this method clears the
            ``self.data_buffer`` if ``self.data_buffer`` is an instance of
            :class:`tianshou.data.data_buffer.BatchSet.` and does nothing if it's not that instance.
            If set to ``False`` then the aforementioned auto clearing behavior is disabled.
        :param episode_cutoff: Optional. An int. The maximum number of timesteps in one episode. This is
            useful when the environment has no terminal states or a single episode could be prohibitively long.
            If set than all episodes are forced to stop beyond this number to timesteps.
        """
        assert sum([num_timesteps > 0, num_episodes > 0]) == 1,\
            "One and only one collection number specification permitted!"

        if isinstance(self.data_buffer, BatchSet) and auto_clear:
            self.data_buffer.clear()

        if num_timesteps > 0:
            num_timesteps_ = int(num_timesteps)
            for _ in range(num_timesteps_):
                action = self.policy.act(self.current_observation, my_feed_dict=my_feed_dict)
                next_observation, reward, done, _ = self.env.step(action)
                self.step_count_this_episode += 1
                if episode_cutoff and self.step_count_this_episode >= episode_cutoff:
                    done = True
                self.data_buffer.add((self.current_observation, action, reward, done))

                if done:
                    self.current_observation = self.env.reset()
                    self.policy.reset()
                    self.step_count_this_episode = 0
                else:
                    self.current_observation = next_observation

        if num_episodes > 0:
            num_episodes_ = int(num_episodes)
            for _ in range(num_episodes_):
                observation = self.env.reset()
                self.policy.reset()
                done = False
                step_count = 0
                while not done:
                    action = self.policy.act(observation, my_feed_dict=my_feed_dict)
                    next_observation, reward, done, _ = self.env.step(action)
                    step_count += 1

                    if episode_cutoff and step_count >= episode_cutoff:
                        done = True

                    self.data_buffer.add((observation, action, reward, done))
                    observation = next_observation

            self.current_observation = self.env.reset()

        if self.process_mode == 'full':
            for processor in self.process_functions:
                self.data.update(processor(self.data_buffer))

        return

    def next_batch(self, batch_size, standardize_advantage=True):
        """
        Constructs and returns the feed_dict of data to be used with ``sess.run``.

        :param batch_size: An int. The size of one minibatch.
        :param standardize_advantage: Optional. A bool but defaulting to ``True``.
            If ``True``, then this method standardize advantages if advantage is required by the networks.
            If ``False`` then this method will never standardize advantage.

        :return: A dict in the format of conventional feed_dict in tf, with keys the placeholders and
            values the numpy arrays.
        """
        sampled_index = self.data_buffer.sample(batch_size)
        if self.process_mode == 'sample':
            for processor in self.process_functions:
                self.data_batch.update(processor(self.data_buffer, indexes=sampled_index))

        # flatten rank-2 list to numpy array, construct feed_dict
        feed_dict = {}
        frame_key_map = {'observation': 0, 'action': 1, 'reward': 2, 'done_flag': 3}
        for key, placeholder in self.required_placeholders.items():
            # check raw_data first
            found, matched_key = internal_key_match(key, frame_key_map.keys())
            if found:
                frame_index = frame_key_map[matched_key]
                flattened = []
                for index_episode, data_episode in zip(sampled_index, self.data_buffer.data):
                    for i in index_episode:
                        flattened.append(data_episode[i][frame_index])
                feed_dict[placeholder] = np.array(flattened)
            else:
                # then check processed minibatch data
                found, matched_key = internal_key_match(key, self.data_batch.keys())
                if found:
                    flattened = list(itertools.chain.from_iterable(self.data_batch[matched_key]))
                    feed_dict[placeholder] = np.array(flattened)
                else:
                    # finally check processed full data
                    found, matched_key = internal_key_match(key, self.data.keys())
                    if found:
                        flattened = [0.] * batch_size  # float
                        i_in_batch = 0
                        for index_episode, data_episode in zip(sampled_index, self.data[matched_key]):
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

    def denoise_action(self, feed_dict, my_feed_dict={}):
        """
        Recompute the actions of deterministic policies without exploration noise, hence denoising.
        It modifies ``feed_dict`` **in place** and has no return value.
        This is useful in, e.g., DDPG since the stored action in ``self.data_buffer`` is the sampled
        action with additional exploration noise.

        :param feed_dict: A dict. It has to be the dict returned by :func:`next_batch` by this class.
        :param my_feed_dict: Optional. A dict defaulting to empty.
            Specifies placeholders such as dropout and batch_norm except observation and action.
        """
        assert isinstance(self.policy, Deterministic), 'denoise_action() could only be called' \
                                                       'with deterministic policies'
        observation = feed_dict[self.required_placeholders['observation']]
        action_mean = self.policy.eval_action(observation, my_feed_dict)
        feed_dict[self.required_placeholders['action']] = action_mean

        return
