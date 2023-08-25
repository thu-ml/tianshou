from numbers import Number
from typing import Optional, Union

import numpy as np
import torch


class MovAvg:
    """Class for moving average.

    It will automatically exclude the infinity and NaN. Usage:
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
        self.cache: list[np.number] = []
        self.banned = [np.inf, np.nan, -np.inf]

    def add(self, data_array: Union[Number, np.number, list, np.ndarray, torch.Tensor]) -> float:
        """Add a scalar into :class:`MovAvg`.

        You can add ``torch.Tensor`` with only one element, a python scalar, or
        a list of python scalar.
        """
        if isinstance(data_array, torch.Tensor):
            data_array = data_array.flatten().cpu().numpy()
        if np.isscalar(data_array):
            data_array = [data_array]
        for number in data_array:  # type: ignore
            if number not in self.banned:
                self.cache.append(number)
        if self.size > 0 and len(self.cache) > self.size:
            self.cache = self.cache[-self.size :]
        return self.get()

    def get(self) -> float:
        """Get the average."""
        if len(self.cache) == 0:
            return 0.0
        return float(np.mean(self.cache))  # type: ignore

    def mean(self) -> float:
        """Get the average. Same as :meth:`get`."""
        return self.get()

    def std(self) -> float:
        """Get the standard deviation."""
        if len(self.cache) == 0:
            return 0.0
        return float(np.std(self.cache))  # type: ignore


class RunningMeanStd:
    """Calculates the running mean and std of a data stream.

    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

    :param mean: the initial mean estimation for data array. Default to 0.
    :param std: the initial standard error estimation for data array. Default to 1.
    :param float clip_max: the maximum absolute value for data array. Default to
        10.0.
    :param float epsilon: To avoid division by zero.
    """

    def __init__(
        self,
        mean: Union[float, np.ndarray] = 0.0,
        std: Union[float, np.ndarray] = 1.0,
        clip_max: Optional[float] = 10.0,
        epsilon: float = np.finfo(np.float32).eps.item(),
    ) -> None:
        self.mean, self.var = mean, std
        self.clip_max = clip_max
        self.count = 0
        self.eps = epsilon

    def norm(self, data_array: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        data_array = (data_array - self.mean) / np.sqrt(self.var + self.eps)
        if self.clip_max:
            data_array = np.clip(data_array, -self.clip_max, self.clip_max)
        return data_array

    def update(self, data_array: np.ndarray) -> None:
        """Add a batch of item into RMS with the same shape, modify mean/var/count."""
        batch_mean, batch_var = np.mean(data_array, axis=0), np.var(data_array, axis=0)
        batch_count = len(data_array)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m_2 / total_count

        self.mean, self.var = new_mean, new_var
        self.count = total_count
