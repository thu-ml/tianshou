from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import torch

from tianshou.highlevel.env import Environments
from tianshou.utils.net.common import Net
from tianshou.utils.string import ToStringMixin

TDevice: TypeAlias = str | int | torch.device


def init_linear_orthogonal(module: torch.nn.Module):
    """Applies orthogonal initialization to linear layers of the given module and sets bias weights to 0.

    :param module: the module whose submodules are to be processed
    """
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)


@dataclass
class Module:
    module: torch.nn.Module
    output_dim: int


class ModuleFactory(ToStringMixin, ABC):
    @abstractmethod
    def create_module(self, envs: Environments, device: TDevice) -> Module:
        pass


class ModuleFactoryNet(ModuleFactory):
    def __init__(self, hidden_sizes: int | Sequence[int]):
        self.hidden_sizes = hidden_sizes

    def create_module(self, envs: Environments, device: TDevice) -> Module:
        module = Net(envs.get_observation_shape())
        return Module(module, module.output_dim)
