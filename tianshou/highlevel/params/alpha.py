from abc import ABC, abstractmethod

import numpy as np
import torch
from sensai.util.string import ToStringMixin
from torch.nn import ParameterList

from tianshou.highlevel.env import Environments
from tianshou.highlevel.module.core import TDevice
from tianshou.highlevel.optim import OptimizerFactoryFactory
from tianshou.policy.modelfree.sac import AutoAlpha


class AutoAlphaFactory(ToStringMixin, ABC):
    @abstractmethod
    def create_auto_alpha(
        self,
        envs: Environments,
        device: TDevice,
    ) -> AutoAlpha:
        pass


class AutoAlphaFactoryDefault(AutoAlphaFactory):
    def __init__(
        self,
        lr: float = 3e-4,
        target_entropy_coefficient: float = -1.0,
        optimizer: OptimizerFactoryFactory | None = None,
    ):
        """
        :param lr: the learning rate for the optimizer of the alpha parameter
        :param target_entropy_coefficient: the coefficient with which to multiply the target entropy;
            The base value being scaled is dim(A) for continuous action spaces and log(|A|) for discrete action spaces,
            i.e. with the default coefficient -1, we obtain -dim(A) and -log(dim(A)) for continuous and discrete action
            spaces respectively, which gives a reasonable trade-off between exploration and exploitation.
            For decidedly stochastic exploration, you can use a positive value closer to 1 (e.g. 0.98);
            1.0 would give full entropy exploration.
        :param optimizer: the optimizer factory to use; if None, use default
        """
        self.lr = lr
        self.target_entropy_coefficient = target_entropy_coefficient
        self.optimizer_factory_factory = optimizer or OptimizerFactoryFactory.default()

    def create_auto_alpha(
        self,
        envs: Environments,
        device: TDevice,
    ) -> AutoAlpha:
        action_dim = np.prod(envs.get_action_shape())
        if envs.get_type().is_continuous():
            target_entropy = self.target_entropy_coefficient * float(action_dim)
        else:
            target_entropy = self.target_entropy_coefficient * np.log(action_dim)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        optim_factory = self.optimizer_factory_factory.create_optimizer_factory(lr=self.lr)
        optim, lr_scheduler = optim_factory.create_instances(ParameterList([log_alpha]))
        if lr_scheduler is not None:
            raise ValueError("Learning rate schedulers are not supported for AutoAlpha")
        return AutoAlpha(target_entropy, log_alpha, optim)
