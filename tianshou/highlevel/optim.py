from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Protocol, TypeAlias

import torch
from torch.optim import Adam, RMSprop

from tianshou.utils.string import ToStringMixin

TParams: TypeAlias = Iterable[torch.Tensor] | Iterable[dict[str, Any]]


class OptimizerWithLearningRateProtocol(Protocol):
    def __call__(self, parameters: Any, lr: float, **kwargs: Any) -> torch.optim.Optimizer:
        pass


class OptimizerFactory(ABC, ToStringMixin):
    def create_optimizer(
        self,
        module: torch.nn.Module,
        lr: float,
    ) -> torch.optim.Optimizer:
        return self.create_optimizer_for_params(module.parameters(), lr)

    @abstractmethod
    def create_optimizer_for_params(self, params: TParams, lr: float) -> torch.optim.Optimizer:
        pass


class OptimizerFactoryTorch(OptimizerFactory):
    def __init__(self, optim_class: OptimizerWithLearningRateProtocol, **kwargs: Any):
        """Factory for torch optimizers.

        :param optim_class: the optimizer class (e.g. subclass of `torch.optim.Optimizer`),
            which will be passed the module parameters, the learning rate as `lr` and the
            kwargs provided.
        :param kwargs: keyword arguments to provide at optimizer construction
        """
        self.optim_class = optim_class
        self.kwargs = kwargs

    def create_optimizer_for_params(self, params: TParams, lr: float) -> torch.optim.Optimizer:
        return self.optim_class(params, lr=lr, **self.kwargs)


class OptimizerFactoryAdam(OptimizerFactory):
    def __init__(
        self,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0,
    ):
        self.weight_decay = weight_decay
        self.eps = eps
        self.betas = betas

    def create_optimizer_for_params(self, params: TParams, lr: float) -> torch.optim.Optimizer:
        return Adam(
            params,
            lr=lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )


class OptimizerFactoryRMSprop(OptimizerFactory):
    def __init__(
        self,
        alpha: float = 0.99,
        eps: float = 1e-08,
        weight_decay: float = 0,
        momentum: float = 0,
        centered: bool = False,
    ):
        self.alpha = alpha
        self.momentum = momentum
        self.centered = centered
        self.weight_decay = weight_decay
        self.eps = eps

    def create_optimizer_for_params(self, params: TParams, lr: float) -> torch.optim.Optimizer:
        return RMSprop(
            params,
            lr=lr,
            alpha=self.alpha,
            eps=self.eps,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            centered=self.centered,
        )
