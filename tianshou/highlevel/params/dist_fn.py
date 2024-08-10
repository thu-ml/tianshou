from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch
from sensai.util.string import ToStringMixin

from tianshou.highlevel.env import Environments
from tianshou.policy.modelfree.pg import TDistFnDiscrete, TDistFnDiscrOrCont


class DistributionFunctionFactory(ToStringMixin, ABC):
    # True return type defined in subclasses
    @abstractmethod
    def create_dist_fn(
        self,
        envs: Environments,
    ) -> Callable[[Any], torch.distributions.Distribution]:
        pass


class DistributionFunctionFactoryCategorical(DistributionFunctionFactory):
    def __init__(self, is_probs_input: bool = True):
        """
        :param is_probs_input: If True, the distribution function shall create a categorical distribution from a
            tensor containing probabilities; otherwise the tensor is assumed to contain logits.
        """
        self.is_probs_input = is_probs_input

    def create_dist_fn(self, envs: Environments) -> TDistFnDiscrete:
        envs.get_type().assert_discrete(self)
        if self.is_probs_input:
            return self._dist_fn_probs
        else:
            return self._dist_fn

    # NOTE: Do not move/rename because a reference to the function can appear in persisted policies
    @staticmethod
    def _dist_fn(logits: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=logits)

    # NOTE: Do not move/rename because a reference to the function can appear in persisted policies
    @staticmethod
    def _dist_fn_probs(probs: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(probs=probs)


class DistributionFunctionFactoryIndependentGaussians(DistributionFunctionFactory):
    def create_dist_fn(self, envs: Environments) -> TDistFnDiscrOrCont:
        envs.get_type().assert_continuous(self)
        return self._dist_fn

    # NOTE: Do not move/rename because a reference to the function can appear in persisted policies
    @staticmethod
    def _dist_fn(loc_scale: tuple[torch.Tensor, torch.Tensor]) -> torch.distributions.Distribution:
        loc, scale = loc_scale
        return torch.distributions.Independent(torch.distributions.Normal(loc, scale), 1)
