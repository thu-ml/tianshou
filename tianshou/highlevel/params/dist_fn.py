from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch

from tianshou.highlevel.env import Environments, EnvType
from tianshou.policy.modelfree.pg import TDistFnDiscrete, TDistFnDiscrOrCont
from tianshou.utils.string import ToStringMixin


class DistributionFunctionFactory(ToStringMixin, ABC):
    # True return type defined in subclasses
    @abstractmethod
    def create_dist_fn(
        self,
        envs: Environments,
    ) -> Callable[[Any], torch.distributions.Distribution]:
        pass


class DistributionFunctionFactoryCategorical(DistributionFunctionFactory):
    def create_dist_fn(self, envs: Environments) -> TDistFnDiscrete:
        envs.get_type().assert_discrete(self)
        return self._dist_fn

    @staticmethod
    def _dist_fn(p: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=p)


class DistributionFunctionFactoryIndependentGaussians(DistributionFunctionFactory):
    def create_dist_fn(self, envs: Environments) -> TDistFnDiscrOrCont:
        envs.get_type().assert_continuous(self)
        return self._dist_fn

    @staticmethod
    def _dist_fn(loc_scale: tuple[torch.Tensor, torch.Tensor]) -> torch.distributions.Distribution:
        loc, scale = loc_scale
        return torch.distributions.Independent(torch.distributions.Normal(loc, scale), 1)


class DistributionFunctionFactoryDefault(DistributionFunctionFactory):
    def create_dist_fn(self, envs: Environments) -> TDistFnDiscrOrCont:
        match envs.get_type():
            case EnvType.DISCRETE:
                return DistributionFunctionFactoryCategorical().create_dist_fn(envs)
            case EnvType.CONTINUOUS:
                return DistributionFunctionFactoryIndependentGaussians().create_dist_fn(envs)
            case _:
                raise ValueError(envs.get_type())
