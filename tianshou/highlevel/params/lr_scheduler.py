from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from tianshou.highlevel.config import SamplingConfig
from tianshou.utils.string import ToStringMixin


class LRSchedulerFactory(ToStringMixin, ABC):
    """Factory for the creation of a learning rate scheduler."""

    @abstractmethod
    def create_scheduler(self, optim: torch.optim.Optimizer) -> LRScheduler:
        pass


class LRSchedulerFactoryLinear(LRSchedulerFactory):
    def __init__(self, sampling_config: SamplingConfig):
        self.sampling_config = sampling_config

    def create_scheduler(self, optim: torch.optim.Optimizer) -> LRScheduler:
        return LambdaLR(optim, lr_lambda=self._LRLambda(self.sampling_config).compute)

    class _LRLambda:
        def __init__(self, sampling_config: SamplingConfig):
            self.max_update_num = (
                np.ceil(sampling_config.step_per_epoch / sampling_config.step_per_collect)
                * sampling_config.num_epochs
            )

        def compute(self, epoch: int) -> float:
            return 1.0 - epoch / self.max_update_num
