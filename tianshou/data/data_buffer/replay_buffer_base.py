from .base import DataBufferBase

__all__ = []


class ReplayBufferBase(DataBufferBase):
    """
    Base class for replay buffer.
    Compared to :class:`DataBufferBase`, it has an additional method :func:`remove`,
    which removes extra data points when the size of the data buffer exceeds capacity.
    Besides, as the practice of using such replay buffer, it's never :func:`clear` ed.
    """
    def remove(self):
        raise NotImplementedError()
