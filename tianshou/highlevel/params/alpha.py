from abc import ABC, abstractmethod

import numpy as np
import torch

from tianshou.highlevel.env import Environments
from tianshou.highlevel.module.core import TDevice
from tianshou.highlevel.optim import OptimizerFactory
from tianshou.utils.string import ToStringMixin


class AutoAlphaFactory(ToStringMixin, ABC):
    @abstractmethod
    def create_auto_alpha(
        self,
        envs: Environments,
        optim_factory: OptimizerFactory,
        device: TDevice,
    ) -> tuple[float, torch.Tensor, torch.optim.Optimizer]:
        pass


class AutoAlphaFactoryDefault(AutoAlphaFactory):  # TODO better name?
    def __init__(self, lr: float = 3e-4):
        self.lr = lr

    def create_auto_alpha(
        self,
        envs: Environments,
        optim_factory: OptimizerFactory,
        device: TDevice,
    ) -> tuple[float, torch.Tensor, torch.optim.Optimizer]:
        target_entropy = float(-np.prod(envs.get_action_shape()))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=self.lr)
        return target_entropy, log_alpha, alpha_optim
