from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.optim.lr_scheduler import LRScheduler, LambdaLR

from tianshou.highlevel.config import RLSamplingConfig


class LRSchedulerFactory(ABC):
    @abstractmethod
    def create_scheduler(self, optim: torch.optim.Optimizer) -> LRScheduler:
        pass


class LinearLRSchedulerFactory(LRSchedulerFactory):
    def __init__(self, sampling_config: RLSamplingConfig):
        self.sampling_config = sampling_config

    def create_scheduler(self, optim: torch.optim.Optimizer) -> LRScheduler:
        max_update_num = (
            np.ceil(self.sampling_config.step_per_epoch / self.sampling_config.step_per_collect)
            * self.sampling_config.num_epochs
        )
        return LambdaLR(optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num)
