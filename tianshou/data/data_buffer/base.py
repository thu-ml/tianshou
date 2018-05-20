__all__ = []


class DataBufferBase(object):
    """
    Base class for data buffer, including replay buffer as used by DQN
    and batched dataset as used by on-policy algorithms.
    Our data buffer adopts a memory-efficient implementation where raw data are always stored in a
    sequential manner, and an additional set of index is used to indicate the valid data points
    in the data buffer.

    The raw data and index are both organized in a two-level architecture as lists of lists, where
    the high-level lists correspond to episodes and low-level lists correspond to the data within
    each episode.

    Mandatory methods for a data buffer class are:

    - :func:`add`. It adds one timestep of data to the data buffer.

    - :func:`clear`. It empties the data buffer.

    - :func:`sample`. It samples one minibatch of data and returns the index of the sampled data\
        points, not the raw data.
    """
    def add(self, frame):
        raise NotImplementedError()

    def clear(self):
        """Empties the data buffer, usually used in batch set but not in replay buffer."""
        raise NotImplementedError()

    def sample(self, batch_size):
        raise NotImplementedError()
