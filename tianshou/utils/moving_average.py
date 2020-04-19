import torch
import numpy as np


class MovAvg(object):
    """Class for moving average. Usage:
    ::

        >>> stat = MovAvg(size=66)
        >>> stat.add(torch.tensor(5))
        5.0
        >>> stat.add(float('inf'))  # which will not add to stat
        5.0
        >>> stat.add([6, 7, 8])
        6.5
        >>> stat.get()
        6.5
        >>> print(f'{stat.mean():.2f}±{stat.std():.2f}')
        6.50±1.12
    """
    def __init__(self, size=100):
        super().__init__()
        self.size = size
        self.cache = []

    def add(self, x):
        """Add a scalar into :class:`MovAvg`. You can add ``torch.Tensor`` with
        only one element, a python scalar, or a list of python scalar. It will
        automatically exclude the infinity and NaN.
        """
        if isinstance(x, torch.Tensor):
            x = x.item()
        if isinstance(x, list):
            for _ in x:
                if _ not in [np.inf, np.nan, -np.inf]:
                    self.cache.append(_)
        elif x != np.inf:
            self.cache.append(x)
        if self.size > 0 and len(self.cache) > self.size:
            self.cache = self.cache[-self.size:]
        return self.get()

    def get(self):
        """Get the average."""
        if len(self.cache) == 0:
            return 0
        return np.mean(self.cache)

    def mean(self):
        """Get the average. Same as :meth:`get`."""
        return self.get()

    def std(self):
        """Get the standard deviation."""
        if len(self.cache) == 0:
            return 0
        return np.std(self.cache)
