from abc import ABC, abstractmethod

import numpy as np
from sensai.util.string import ToStringMixin

from tianshou.highlevel.env import Environments
from tianshou.highlevel.module.core import TDevice
from tianshou.highlevel.optim import OptimizerFactoryFactory
from tianshou.algorithm.modelfree.sac import Alpha, AutoAlpha


class AutoAlphaFactory(ToStringMixin, ABC):
    @abstractmethod
    def create_auto_alpha(
        self,
        envs: Environments,
        device: TDevice,
    ) -> Alpha:
        pass


class AutoAlphaFactoryDefault(AutoAlphaFactory):
    def __init__(
        self,
        lr: float = 3e-4,
        target_entropy_coefficient: float = -1.0,
        log_alpha: float = 0.0,
        optim: OptimizerFactoryFactory | None = None,
    ) -> None:
        """
        :param lr: the learning rate for the optimizer of the alpha parameter
        :param target_entropy_coefficient: the coefficient with which to multiply the target entropy;
            The base value being scaled is dim(A) for continuous action spaces and log(|A|) for discrete action spaces,
            i.e. with the default coefficient -1, we obtain -dim(A) and -log(dim(A)) for continuous and discrete action
            spaces respectively, which gives a reasonable trade-off between exploration and exploitation.
            For decidedly stochastic exploration, you can use a positive value closer to 1 (e.g. 0.98);
            1.0 would give full entropy exploration.
        :param log_alpha: the (initial) value of the log of the entropy regularization coefficient alpha.
        :param optim: the optimizer factory to use; if None, use default
        """
        self.lr = lr
        self.target_entropy_coefficient = target_entropy_coefficient
        self.log_alpha = log_alpha
        self.optimizer_factory_factory = optim or OptimizerFactoryFactory.default()

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
        optim_factory = self.optimizer_factory_factory.create_optimizer_factory(lr=self.lr)
        return AutoAlpha(target_entropy, self.log_alpha, optim_factory)
