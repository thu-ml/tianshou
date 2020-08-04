import torch
import numpy as np
from typing import Union

from tianshou.data import to_numpy


class MovAvg(object):
    """Class for moving average. It will automatically exclude the infinity and
    NaN. Usage:
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

    def __init__(self, size: int = 100) -> None:
        super().__init__()
        self.size = size
        self.cache = []
        self.banned = [np.inf, np.nan, -np.inf]

    def add(self, x: Union[float, list, np.ndarray, torch.Tensor]) -> float:
        """Add a scalar into :class:`MovAvg`. You can add ``torch.Tensor`` with
        only one element, a python scalar, or a list of python scalar.
        """
        if isinstance(x, torch.Tensor):
            x = to_numpy(x.flatten())
        if isinstance(x, list) or isinstance(x, np.ndarray):
            for _ in x:
                if _ not in self.banned:
                    self.cache.append(_)
        elif x not in self.banned:
            self.cache.append(x)
        if self.size > 0 and len(self.cache) > self.size:
            self.cache = self.cache[-self.size:]
        return self.get()

    def get(self) -> float:
        """Get the average."""
        if len(self.cache) == 0:
            return 0
        return np.mean(self.cache)

    def mean(self) -> float:
        """Get the average. Same as :meth:`get`."""
        return self.get()

    def std(self) -> float:
        """Get the standard deviation."""
        if len(self.cache) == 0:
            return 0
        return np.std(self.cache)
