from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.optim import Adam, RMSprop


class OptimizerFactory(ABC):
    # TODO: Is it OK to assume that all optimizers have a learning rate argument?
    # Right now, the learning rate is typically a configuration parameter.
    # If we drop the assumption, we can't have that and will need to move the parameter
    # to the optimizer factory, which is inconvenient for the user.
    @abstractmethod
    def create_optimizer(self, module: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
        pass


class OptimizerFactoryTorch(OptimizerFactory):
    def __init__(self, optim_class: Any, **kwargs):
        """:param optim_class: the optimizer class (e.g. subclass of `torch.optim.Optimizer`),
            which will be passed the module parameters, the learning rate as `lr` and the
            kwargs provided.
        :param kwargs: keyword arguments to provide at optimizer construction
        """
        self.optim_class = optim_class
        self.kwargs = kwargs

    def create_optimizer(self, module: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
        return self.optim_class(module.parameters(), lr=lr, **self.kwargs)


class OptimizerFactoryAdam(OptimizerFactory):
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


class OptimizerFactoryRMSprop(OptimizerFactory):
    def __init__(self, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False):
        self.alpha = alpha
        self.momentum = momentum
        self.centered = centered
        self.weight_decay = weight_decay
        self.eps = eps

    def create_optimizer(self, module: torch.nn.Module, lr: float) -> RMSprop:
        return RMSprop(
            module.parameters(),
            lr=lr,
            alpha=self.alpha,
            eps=self.eps,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            centered=self.centered,
        )
