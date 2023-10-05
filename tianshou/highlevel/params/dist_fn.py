from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TypeAlias

import torch

from tianshou.highlevel.env import Environments, EnvType
from tianshou.policy.modelfree.pg import TDistParams
from tianshou.utils.string import ToStringMixin

TDistributionFunction: TypeAlias = Callable[[TDistParams], torch.distributions.Distribution]


class DistributionFunctionFactory(ToStringMixin, ABC):
    @abstractmethod
    def create_dist_fn(self, envs: Environments) -> TDistributionFunction:
        pass


def _dist_fn_categorical(p):
    return torch.distributions.Categorical(logits=p)


def _dist_fn_gaussian(*p):
    return torch.distributions.Independent(torch.distributions.Normal(*p), 1)


class DistributionFunctionFactoryDefault(DistributionFunctionFactory):
    def create_dist_fn(self, envs: Environments) -> TDistributionFunction:
        match envs.get_type():
            case EnvType.DISCRETE:
                return _dist_fn_categorical
            case EnvType.CONTINUOUS:
                return _dist_fn_gaussian
            case _:
                raise ValueError(envs.get_type())
