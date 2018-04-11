import gc
import numpy as np
import logging

from .base import DataBufferBase

STATE = 0
ACTION = 1
REWARD = 2
DONE = 3

class BatchSet(DataBufferBase):
    """
    class for batched dataset as used in on-policy algos
    """
    def __init__(self, nstep=None):
        self.nstep = nstep or 1  # RL has to look ahead at least one timestep

        self.data = [[]]
        self.index = [[]]
        self.candidate_index = 0

        self.size = 0  # number of valid data points (not frames)

        self.index_lengths = [0]  # for sampling

    def add(self, frame):
        self.data[-1].append(frame)

        has_enough_frames = len(self.data[-1]) > self.nstep
        if frame[DONE]:  # episode terminates, all trailing frames become valid data points
            trailing_index = list(range(self.candidate_index, len(self.data[-1])))
            self.index[-1] += trailing_index
            self.size += len(trailing_index)
            self.index_lengths[-1] += len(trailing_index)

            # prepare for the next episode
            self.data.append([])
            self.index.append([])
            self.candidate_index = 0

            self.index_lengths.append(0)

        elif has_enough_frames:  # add one valid data point
            self.index[-1].append(self.candidate_index)
            self.candidate_index += 1
            self.size += 1
            self.index_lengths[-1] += 1

    def clear(self):
        del self.data
        del self.index
        del self.index_lengths

        gc.collect()

        self.data = [[]]
        self.index = [[]]
        self.candidate_index = 0
        self.size = 0
        self.index_lengths = [0]

    def sample(self, batch_size):
        # TODO: move unified properties and methods to base. but this depends on how to deal with nstep

        prob_episode = np.array(self.index_lengths) * 1. / self.size
        num_episodes = len(self.index)
        sampled_index = [[] for _ in range(num_episodes)]

        for _ in range(batch_size):
            # sample which episode
            sampled_episode_i = int(np.random.choice(num_episodes, p=prob_episode))

            # sample which data point within the sampled episode
            sampled_frame_i = int(np.random.randint(self.index_lengths[sampled_episode_i]))
            sampled_index[sampled_episode_i].append(sampled_frame_i)

        return sampled_index

    def statistics(self, discount_factor=0.99):
        returns = []
        undiscounted_returns = []

        if len(self.data) == 1:
            data = self.data
            logging.warning('The first episode in BatchSet is still not finished. '
                            'Logging its return anyway.')
        else:
            data = self.data[:-1]

        for episode in data:
            current_return = 0.
            current_undiscounted_return = 0.
            current_discount = 1.
            for frame in episode:
                current_return += frame[REWARD] * current_discount
                current_undiscounted_return += frame[REWARD]
                current_discount *= discount_factor
            returns.append(current_return)
            undiscounted_returns.append(current_undiscounted_return)

        mean_return = np.mean(returns)
        mean_undiscounted_return = np.mean(undiscounted_returns)

        print('Mean return: {}'.format(mean_return))
        print('Mean undiscounted return: {}'.format(mean_undiscounted_return))
