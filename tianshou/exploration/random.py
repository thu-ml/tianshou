import numpy as np
from typing import Union, Optional
from abc import ABC, abstractmethod


class BaseNoise(ABC, object):
    """The action noise base class."""

    def __init__(self, **kwargs) -> None:
        super(BaseNoise, self).__init__()

    @abstractmethod
    def __call__(self, **kwargs) -> np.ndarray:
        """Generate new noise."""
        raise NotImplementedError

    def reset(self, **kwargs) -> None:
        """Reset to the initial state."""
        pass


class GaussianNoise(BaseNoise):
    """Class for vanilla gaussian process,
    used for exploration in DDPG by default.
    """

    def __init__(self,
                 mu: float = 0.0,
                 sigma: float = 1.0):
        super().__init__()
        self._mu = mu
        assert 0 <= sigma, 'noise std should not be negative'
        self._sigma = sigma

    def __call__(self, size: tuple) -> np.ndarray:
        return np.random.normal(self._mu, self._sigma, size)


class OUNoise(BaseNoise):
    """Class for Ornstein-Uhlenbeck process, as used for exploration in DDPG.
    Usage:
    ::

        # init
        self.noise = OUNoise()
        # generate noise
        noise = self.noise(logits.shape, eps)

    For required parameters, you can refer to the stackoverflow page. However,
    our experiment result shows that (similar to OpenAI SpinningUp) using
    vanilla gaussian process has little difference from using the
    Ornstein-Uhlenbeck process.
    """

    def __init__(self,
                 mu: float = 0.0,
                 sigma: float = 0.3,
                 theta: float = 0.15,
                 dt: float = 1e-2,
                 x0: Optional[Union[float, np.ndarray]] = None
                 ) -> None:
        super(BaseNoise, self).__init__()
        self._mu = mu
        self._alpha = theta * dt
        self._beta = sigma * np.sqrt(dt)
        self._x0 = x0
        self.reset()

    def __call__(self, size: tuple, mu: Optional[float] = None) -> np.ndarray:
        """Generate new noise. Return a ``numpy.ndarray`` which size is equal
        to ``size``.
        """
        if self._x is None or self._x.shape != size:
            self._x = 0
        if mu is None:
            mu = self._mu
        r = self._beta * np.random.normal(size=size)
        self._x = self._x + self._alpha * (mu - self._x) + r
        return self._x

    def reset(self) -> None:
        """Reset to the initial state."""
        self._x = self._x0
