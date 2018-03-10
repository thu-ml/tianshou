

class DataBufferBase(object):
    """
    base class for data buffer, including replay buffer as in DQN and batched dataset as in on-policy algos
    """
    def add(self, frame):
        raise NotImplementedError()

    def clear(self):
        raise NotImplementedError()

    def sample(self, batch_size):
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
