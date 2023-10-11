from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import torch

from tianshou.highlevel.env import Environments
from tianshou.utils.net.discrete import ImplicitQuantileNetwork
from tianshou.utils.string import ToStringMixin

TDevice: TypeAlias = str | torch.device


def init_linear_orthogonal(module: torch.nn.Module) -> None:
    """Applies orthogonal initialization to linear layers of the given module and sets bias weights to 0.

    :param module: the module whose submodules are to be processed
    """
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)


class ModuleFactory(ABC):
    @abstractmethod
    def create_module(self, envs: Environments, device: TDevice) -> torch.nn.Module:
        pass


@dataclass
class IntermediateModule:
    module: torch.nn.Module
    output_dim: int


class IntermediateModuleFactory(ToStringMixin, ModuleFactory, ABC):
    @abstractmethod
    def create_intermediate_module(self, envs: Environments, device: TDevice) -> IntermediateModule:
        pass

    def create_module(self, envs: Environments, device: TDevice) -> torch.nn.Module:
        return self.create_intermediate_module(envs, device).module


class ImplicitQuantileNetworkFactory(ModuleFactory, ToStringMixin):
    def __init__(
        self,
        preprocess_net_factory: IntermediateModuleFactory,
        hidden_sizes: Sequence[int] = (),
        num_cosines: int = 64,
    ):
        self.preprocess_net_factory = preprocess_net_factory
        self.hidden_sizes = hidden_sizes
        self.num_cosines = num_cosines

    def create_module(self, envs: Environments, device: TDevice) -> ImplicitQuantileNetwork:
        preprocess_net = self.preprocess_net_factory.create_intermediate_module(envs, device)
        return ImplicitQuantileNetwork(
            preprocess_net=preprocess_net.module,
            action_shape=envs.get_action_shape(),
            hidden_sizes=self.hidden_sizes,
            num_cosines=self.num_cosines,
            preprocess_net_output_dim=preprocess_net.output_dim,
            device=device,
        ).to(device)
