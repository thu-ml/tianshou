from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class BaseNoise(ABC):
    """The action noise base class."""

    @abstractmethod
    def reset(self) -> None:
        """Reset to the initial state."""

    @abstractmethod
    def __call__(self, size: Sequence[int]) -> np.ndarray:
        """Generate new noise."""
        raise NotImplementedError


class GaussianNoise(BaseNoise):
    """The vanilla Gaussian process, for exploration in DDPG by default."""

    def __init__(self, mu: float = 0.0, sigma: float = 1.0) -> None:
        self._mu = mu
        assert sigma >= 0, "Noise std should not be negative."
        self._sigma = sigma

    def __call__(self, size: Sequence[int]) -> np.ndarray:
        return np.random.normal(self._mu, self._sigma, size)

    def reset(self) -> None:
        pass


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
    vanilla Gaussian process has little difference from using the
    Ornstein-Uhlenbeck process.
    """

    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 0.3,
        theta: float = 0.15,
        dt: float = 1e-2,
        x0: float | np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self._mu = mu
        self._alpha = theta * dt
        self._beta = sigma * np.sqrt(dt)
        self._x0 = x0
        self.reset()

    def reset(self) -> None:
        """Reset to the initial state."""
        self._x = self._x0

    def __call__(self, size: Sequence[int], mu: float | None = None) -> np.ndarray:
        """Generate new noise.

        Return an numpy array which size is equal to ``size``.
        """
        if self._x is None or (isinstance(self._x, np.ndarray) and self._x.shape != size):
            self._x = 0.0
        if mu is None:
            mu = self._mu
        r = self._beta * np.random.normal(size=size)
        self._x = self._x + self._alpha * (mu - self._x) + r
        return self._x  # type: ignore
