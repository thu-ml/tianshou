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
    Class for batched dataset as used in on-policy algorithms, where a batch of data is first collected
    with the current policy, several optimization steps are then conducted on this batch of data and the
    data are then discarded and collected again.

    :param nstep: An int defaulting to 1. The number of timesteps to lookahead for temporal difference computation.
        Only continuous data pieces longer than this number or already terminated ones are
        considered valid data points.
    """
    def __init__(self, nstep=1):
        self.nstep = nstep  # RL has to look ahead at least one timestep

        self.data = [[]]
        self.index = [[]]
        self.candidate_index = 0

        self.size = 0  # number of valid data points (not frames)

        self.index_lengths = [0]  # for sampling

    def add(self, frame):
        """
        Adds one frame of data to the buffer.

        :param frame: A tuple of (observation, action, reward, done_flag).
        """
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
        """
        Empties the data buffer and prepares to collect a new batch of data.
        """
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
        """
        Performs uniform random sampling on ``self.index``. For simplicity, we do random sampling with replacement
        for now with time O(``batch_size``). Fastest sampling without replacement seems to have to be of time
        O(``batch_size`` * log(num_episodes)).

        :param batch_size: An int. The size of the minibatch.

        :return: A list of list of the sampled indexes. Episodes without sampled data points
            correspond to empty sub-lists.
        """
        # TODO: move unified properties and methods to base. but this may depend on how to deal with nstep

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
        """
        Computes and prints out the statistics (e.g., discounted returns, undiscounted returns) in the batch set.
        This is useful when policies are optimized by on-policy algorithms, so the current data in
        the batch set directly reflect the performance of the current policy.

        :param discount_factor: Optional. A float in range :math:`[0, 1]` defaulting to 0.99. The discount
            factor to compute discounted returns.
        """
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
