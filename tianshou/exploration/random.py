import numpy as np


class OUNoise(object):
    """docstring for OUNoise"""

    def __init__(self, sigma=0.3, theta=0.15, dt=1e-2, x0=None):
        self.alpha = theta * dt
        self.beta = sigma * np.sqrt(dt)
        self.x0 = x0
        self.reset()

    def __call__(self, size, mu=.1):
        if self.x is None or self.x.shape != size:
            self.x = 0
        self.x = self.x + self.alpha * (mu - self.x) + \
            self.beta * np.random.normal(size=size)
        return self.x

    def reset(self):
        self.x = None
