from abc import ABC, abstractmethod
from typing import TypeAlias

import numpy as np
import torch

from tianshou.highlevel.env import Environments

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
    """Represents a factory for the creation of a torch module given an environment and target device."""

    @abstractmethod
    def create_module(self, envs: Environments, device: TDevice) -> torch.nn.Module:
        pass
