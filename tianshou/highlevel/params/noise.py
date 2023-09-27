from abc import ABC, abstractmethod

from tianshou.exploration import BaseNoise, GaussianNoise
from tianshou.highlevel.env import ContinuousEnvironments, Environments


class NoiseFactory(ABC):
    @abstractmethod
    def create_noise(self, envs: Environments) -> BaseNoise:
        pass


class NoiseFactoryMaxActionScaledGaussian(NoiseFactory):
    """Factory for Gaussian noise where the standard deviation is a fraction of the maximum action value.

    This factory can only be applied to continuous action spaces.
    """

    def __init__(self, std_fraction: float):
        self.std_fraction = std_fraction

    def create_noise(self, envs: Environments) -> BaseNoise:
        envs.get_type().assert_continuous(self)
        envs: ContinuousEnvironments
        return GaussianNoise(sigma=envs.max_action * self.std_fraction)


class MaxActionScaledGaussian(NoiseFactoryMaxActionScaledGaussian):
    pass
