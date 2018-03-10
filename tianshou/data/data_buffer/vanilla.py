import logging
import numpy as np

from .replay_buffer_base import ReplayBufferBase

STATE = 0
ACTION = 1
REWARD = 2
DONE = 3

# TODO: valid data points could be less than `nstep` timesteps. Check priority replay paper!
class VanillaReplayBuffer(ReplayBufferBase):
    """
    vanilla replay buffer as used in (Mnih, et al., 2015).
    Frames are always continuous in temporal order. They are only removed from the beginning. This continuity
    in `self.data` could be exploited, but only in vanilla replay buffer.
    """
    def __init__(self, capacity, nstep=1):
        """
        :param capacity: int. capacity of the buffer.
        :param nstep: int. number of timesteps to lookahead for temporal difference
        """
        assert capacity > 0
        self.capacity = int(capacity)
        self.nstep = nstep

        self.data = [[]]
        self.index = [[]]
        self.candidate_index = 0

        self.size = 0  # number of valid data points (not frames)

        self.index_lengths = [0]  # for sampling

    def add(self, frame):
        """
        add one frame to the buffer.
        :param frame: tuple, (observation, action, reward, done_flag).
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

        # automated removal to capacity
        if self.size > self.capacity:
            self.remove()

    def remove(self):
        """
        remove data until `self.size` <= `self.capacity`
        """
        if self.size:
            while self.size > self.capacity:
                self.remove_oldest()
        else:
            logging.warning('Attempting to remove from empty buffer!')

    def remove_oldest(self):
        """
        remove the oldest data point, in this case, just the oldest frame. Empty episodes are also removed
        if resulted from removal.
        """
        self.index[0].pop()  # note that all index of frames in the first episode are shifted forward by 1

        if self.index[0]:  # first episode still has data points
            self.data[0].pop(0)
            if len(self.data) == 1:  # otherwise self.candidate index is for another episode
                self.candidate_index -= 1
            self.index_lengths[0] -= 1

        else:  # first episode becomes empty
            self.data.pop(0)
            self.index.pop(0)
            if len(self.data) == 0:  # otherwise self.candidate index is for another episode
                self.candidate_index = 0

            self.index_lengths.pop(0)

        self.size -= 1

    def sample(self, batch_size):
        """
        uniform random sampling on `self.index`. For simplicity, we do random sampling with replacement
        for now with time O(`batch_size`). Fastest sampling without replacement seems to have to be of time
        O(`batch_size` * log(num_episodes)).
        :param batch_size: int.
        :return: sampled index, same structure as `self.index`. Episodes without sampled data points
        correspond to empty sub-lists.
        """
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
