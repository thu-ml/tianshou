from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor
from torch.optim import Adam

TParams = Iterable[Tensor] | Iterable[dict[str, Any]]


class OptimizerFactory(ABC):
    @abstractmethod
    def create_optimizer(self, module: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
        pass


class TorchOptimizerFactory(OptimizerFactory):
    def __init__(self, optim_class: Any, **kwargs):
        self.optim_class = optim_class
        self.kwargs = kwargs

    def create_optimizer(self, module: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
        return self.optim_class(module.parameters(), lr=lr, **self.kwargs)


class AdamOptimizerFactory(OptimizerFactory):
    def __init__(self, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):
        self.weight_decay = weight_decay
        self.eps = eps
        self.betas = betas

    def create_optimizer(self, module: torch.nn.Module, lr: float) -> Adam:
        return Adam(
            module.parameters(),
            lr=lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )


