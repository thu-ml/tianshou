from abc import ABC, abstractmethod

import torch

from tianshou.highlevel.env import Environments, EnvType
from tianshou.policy.modelfree.pg import TDistributionFunction
from tianshou.utils.string import ToStringMixin


class DistributionFunctionFactory(ToStringMixin, ABC):
    @abstractmethod
    def create_dist_fn(self, envs: Environments) -> TDistributionFunction:
        pass


class DistributionFunctionFactoryCategorical(DistributionFunctionFactory):
    def create_dist_fn(self, envs: Environments) -> TDistributionFunction:
        envs.get_type().assert_discrete(self)
        return self._dist_fn

    @staticmethod
    def _dist_fn(p: torch.Tensor) -> torch.distributions.Distribution:
        return torch.distributions.Categorical(logits=p)


class DistributionFunctionFactoryIndependentGaussians(DistributionFunctionFactory):
    def create_dist_fn(self, envs: Environments) -> TDistributionFunction:
        envs.get_type().assert_continuous(self)
        return self._dist_fn

    @staticmethod
    def _dist_fn(*p: torch.Tensor) -> torch.distributions.Distribution:
        return torch.distributions.Independent(torch.distributions.Normal(*p), 1)


class DistributionFunctionFactoryDefault(DistributionFunctionFactory):
    def create_dist_fn(self, envs: Environments) -> TDistributionFunction:
        match envs.get_type():
            case EnvType.DISCRETE:
                return DistributionFunctionFactoryCategorical().create_dist_fn(envs)
            case EnvType.CONTINUOUS:
                return DistributionFunctionFactoryIndependentGaussians().create_dist_fn(envs)
            case _:
                raise ValueError(envs.get_type())
