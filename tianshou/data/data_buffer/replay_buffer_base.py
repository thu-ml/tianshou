from .base import DataBufferBase

class ReplayBufferBase(DataBufferBase):
    """
    base class for replay buffer.
    """
    def remove(self):
        """
        when size exceeds capacity, removes extra data points
        :return:
        """
        raise NotImplementedError()
