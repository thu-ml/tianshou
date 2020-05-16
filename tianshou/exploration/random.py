import numpy as np
from typing import Union, Optional


class OUNoise(object):
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
                 sigma: float = 0.3,
                 theta: float = 0.15,
                 dt: float = 1e-2,
                 x0: Optional[Union[float, np.ndarray]] = None
                 ) -> None:
        self.alpha = theta * dt
        self.beta = sigma * np.sqrt(dt)
        self.x0 = x0
        self.reset()

    def __call__(self, size: tuple, mu: float = .1) -> np.ndarray:
        """Generate new noise. Return a ``numpy.ndarray`` which size is equal
        to ``size``.
        """
        if self.x is None or self.x.shape != size:
            self.x = 0
        r = self.beta * np.random.normal(size=size)
        self.x = self.x + self.alpha * (mu - self.x) + r
        return self.x

    def reset(self) -> None:
        """Reset to the initial state."""
        self.x = None
