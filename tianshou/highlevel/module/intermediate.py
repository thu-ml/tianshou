from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from tianshou.highlevel.env import Environments
from tianshou.highlevel.module.core import ModuleFactory, TDevice
from tianshou.utils.string import ToStringMixin


@dataclass
class IntermediateModule:
    """Container for a module which computes an intermediate representation (with a known dimension)."""

    module: torch.nn.Module
    output_dim: int


class IntermediateModuleFactory(ToStringMixin, ModuleFactory, ABC):
    """Factory for the generation of a module which computes an intermediate representation."""

    @abstractmethod
    def create_intermediate_module(self, envs: Environments, device: TDevice) -> IntermediateModule:
        pass

    def create_module(self, envs: Environments, device: TDevice) -> torch.nn.Module:
        return self.create_intermediate_module(envs, device).module
