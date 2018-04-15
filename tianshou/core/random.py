"""
adapted from keras-rl
"""

from __future__ import division
import numpy as np


class RandomProcess(object):
    """
    Base class for random process for exploration in the environment.
    """
    def reset_states(self):
        """
        Reset the internal states, if any, of the random process. Does nothing by default.
        """
        pass


class AnnealedGaussianProcess(RandomProcess):
    """
    Class for annealed Gaussian process, annealing the sigma in the Gaussian-like distribution along sampling.
    At each timestep, the class samples from a Gaussian-like distribution.

    :param mu: A float. Specifying the mean of the Gaussian-like distribution.
    :param sigma: A float. Specifying the std of teh Gaussian-like distribution.
    :param sigma_min: A float. Specifying the minimum std until which the annealing stops.
    :param n_steps_annealing: An int. It specifies the total number of steps for which the annealing happens.
    """
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        """The current sigma after potential annealing."""
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma


class GaussianWhiteNoiseProcess(AnnealedGaussianProcess):
    """
    Class for Gaussian white noise. At each timestep, the class samples from an exact Gaussian distribution.
    It allows annealing in the std of the Gaussian, but the distribution is independent at different timesteps.

    :param mu: A float defaulting to 0. Specifying the mean of the Gaussian-like distribution.
    :param sigma: A float defaulting to 1. Specifying the std of the Gaussian-like distribution.
    :param sigma_min: Optional. A float. Specifying the minimum std until which the annealing stops. It defaults to
        ``None`` where no annealing takes place.
    :param n_steps_annealing: Optional. An int. It specifies the total number of steps for which the annealing happens.
        Only effective when ``sigma_mean`` is not ``None``.
    :param size: An int or tuple of ints. It corresponds to the shape of the action of the environment.
    """
    def __init__(self, mu=0., sigma=1., sigma_min=None, n_steps_annealing=1000, size=1):
        super(GaussianWhiteNoiseProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.size = size

    def sample(self):
        """
        Draws one sample from the random process.

        :return: A numpy array. The drawn sample.
        """
        sample = np.random.normal(self.mu, self.current_sigma, self.size)
        self.n_steps += 1
        return sample


class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    """
    Class for Ornstein-Uhlenbeck Process, as used for exploration in DDPG. Implemented based on
    http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab .
    It basically is a temporal-correlated Gaussian process where the distribution at the current timestep depends on
    the samples from the last timestep. It's not exactly Gaussian but still resembles Gaussian.

    :param theta: A float. A special parameter for this process.
    :param mu: A float. Another parameter of this process, but it's not exactly the mean of the distribution.
    :param sigma: A float. Another parameter of this process. It acts like the std of the Gaussian-like distribution
        to some extent.
    :param dt: A float. The time interval to simulate this process discretely, as the process is mathematically defined
        to be a continuous one.
    :param x0: Optional. A float. The initial value of "the samples from the last timestep" so as to draw the first
        sample. It defaults to zero.
    :param size:  An int or tuple of ints. It corresponds to the shape of the action of the environment.
    :param sigma_min: Optional. A float. Specifying the minimum std until which the annealing stops. It defaults to
        ``None`` where no annealing takes place.
    :param n_steps_annealing: An int. It specifies the total number of steps for which the annealing happens.
    """
    def __init__(self, theta, mu=0., sigma=1., dt=1e-2, x0=None, size=1, sigma_min=None, n_steps_annealing=1000):
        super(OrnsteinUhlenbeckProcess, self).__init__(
            mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        """
        Draws one sample from the random process.

        :return: A numpy array. The drawn sample.
        """
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        """
        Reset ``self.x_prev`` to be ``self.x0``.
        """
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)
