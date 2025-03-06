from abc import ABC, abstractmethod

import numpy as np
import torch
from sensai.util.string import ToStringMixin

from tianshou.highlevel.env import Environments
from tianshou.highlevel.module.core import TDevice
from tianshou.highlevel.optim import OptimizerFactory
from tianshou.policy.modelfree.sac import AutoAlpha


class AutoAlphaFactory(ToStringMixin, ABC):
    @abstractmethod
    def create_auto_alpha(
        self,
        envs: Environments,
        optim_factory: OptimizerFactory,
        device: TDevice,
    ) -> AutoAlpha:
        pass


class AutoAlphaFactoryDefault(AutoAlphaFactory):
    def __init__(self, lr: float = 3e-4):
        self.lr = lr

    def create_auto_alpha(
        self,
        envs: Environments,
        optim_factory: OptimizerFactory,
        device: TDevice,
    ) -> AutoAlpha:
        target_entropy = float(-np.prod(envs.get_action_shape()))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optim = optim_factory.create_optimizer_for_params([log_alpha], self.lr)
        return AutoAlpha(target_entropy, log_alpha, alpha_optim)
