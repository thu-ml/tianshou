from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from sensai.util.string import ToStringMixin

from tianshou.highlevel.env import Environments
from tianshou.highlevel.module.core import ModuleFactory, TDevice
from tianshou.utils.net.common import ModuleWithVectorOutput


@dataclass
class IntermediateModule:
    """Container for a module which computes an intermediate representation (with a known dimension)."""

    module: torch.nn.Module
    output_dim: int

    def get_module_with_vector_output(self) -> ModuleWithVectorOutput:
        if isinstance(self.module, ModuleWithVectorOutput):
            return self.module
        else:
            return ModuleWithVectorOutput.from_module(self.module, self.output_dim)


class IntermediateModuleFactory(ToStringMixin, ModuleFactory, ABC):
    """Factory for the generation of a module which computes an intermediate representation."""

    @abstractmethod
    def create_intermediate_module(self, envs: Environments, device: TDevice) -> IntermediateModule:
        pass

    def create_module(self, envs: Environments, device: TDevice) -> torch.nn.Module:
        return self.create_intermediate_module(envs, device).module
