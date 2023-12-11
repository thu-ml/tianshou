from abc import ABC, abstractmethod

from tianshou.exploration import BaseNoise, GaussianNoise
from tianshou.highlevel.env import ContinuousEnvironments, Environments
from tianshou.utils.string import ToStringMixin


class NoiseFactory(ToStringMixin, ABC):
    @abstractmethod
    def create_noise(self, envs: Environments) -> BaseNoise:
        pass


class NoiseFactoryMaxActionScaledGaussian(NoiseFactory):
    def __init__(self, std_fraction: float):
        """Factory for Gaussian noise where the standard deviation is a fraction of the maximum action value.

        This factory can only be applied to continuous action spaces.

        :param std_fraction: fraction (between 0 and 1) of the maximum action value that shall
        be used as the standard deviation
        """
        self.std_fraction = std_fraction

    def create_noise(self, envs: Environments) -> GaussianNoise:
        envs.get_type().assert_continuous(self)
        envs: ContinuousEnvironments
        return GaussianNoise(sigma=envs.max_action * self.std_fraction)


class MaxActionScaledGaussian(NoiseFactoryMaxActionScaledGaussian):
    pass
