from .base import DataBufferBase


class BatchSet(DataBufferBase):
    """
    class for batched dataset as used in on-policy algos
    """
    def __init__(self):
        self.data = [[]]
        self.index = [[]]
        self.candidate_index = 0

        self.size = 0  # number of valid data points (not frames)

        self.index_lengths = [0]  # for sampling

    def add(self, frame):
        self.data[-1].append(frame)

    def clear(self):
        pass

    def sample(self, batch_size):
        pass
