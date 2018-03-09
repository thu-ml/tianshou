

class DataBufferBase(object):
    """
    base class for data buffer, including replay buffer as in DQN and batched dataset as in on-policy algos
    """
    def add(self, frame):
        raise NotImplementedError()

    def clear(self):
        raise NotImplementedError()

    def sample(self, batch_size):
        raise NotImplementedError()