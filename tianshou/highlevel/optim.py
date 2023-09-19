from abc import ABC, abstractmethod
from typing import Union, Iterable, Dict, Any, Optional

import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import LRScheduler, LambdaLR

from tianshou.config import RLSamplingConfig, NNConfig

TParams = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]


class OptimizerFactory(ABC):
    @abstractmethod
    def create_optimizer(self, module: torch.nn.Module) -> torch.optim.Optimizer:
        pass


class TorchOptimizerFactory(OptimizerFactory):
    def __init__(self, optim_class, **kwargs):
        self.optim_class = optim_class
        self.kwargs = kwargs

    def create_optimizer(self, module: torch.nn.Module) -> torch.optim.Optimizer:
        return self.optim_class(module.parameters(), **self.kwargs)


class AdamOptimizerFactory(OptimizerFactory):
    def __init__(self, lr):
        self.lr = lr

    def create_optimizer(self, module: torch.nn.Module) -> Adam:
        return Adam(module.parameters(), lr=self.lr)


class LRSchedulerFactory(ABC):
    @abstractmethod
    def create_scheduler(self, optim: torch.optim.Optimizer) -> Optional[LRScheduler]:
        pass


class LinearLRSchedulerFactory(LRSchedulerFactory):
    def __init__(self, nn_config: NNConfig, sampling_config: RLSamplingConfig):
        self.nn_config = nn_config
        self.sampling_config = sampling_config

    def create_scheduler(self, optim: torch.optim.Optimizer) -> Optional[LRScheduler]:
        lr_scheduler = None
        if self.nn_config.lr_decay:
            max_update_num = np.ceil(self.sampling_config.step_per_epoch / self.sampling_config.step_per_collect) * self.sampling_config.num_epochs
            lr_scheduler = LambdaLR(optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num)
        return lr_scheduler
