from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any, Self, TypeAlias

import numpy as np
import torch
from sensai.util.string import ToStringMixin
from torch.optim import Adam, RMSprop
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

ParamsType: TypeAlias = Iterable[torch.Tensor] | Iterable[dict[str, Any]]


class LRSchedulerFactory(ToStringMixin, ABC):
    """Factory for the creation of a learning rate scheduler."""

    @abstractmethod
    def create_scheduler(self, optim: torch.optim.Optimizer) -> LRScheduler:
        pass


class LRSchedulerFactoryLinear(LRSchedulerFactory):
    """
    Factory for a learning rate scheduler where the learning rate linearly decays towards
    zero for the given trainer parameters.
    """

    def __init__(self, num_epochs: int, step_per_epoch: int, step_per_collect: int):
        self.num_epochs = num_epochs
        self.step_per_epoch = step_per_epoch
        self.step_per_collect = step_per_collect

    def create_scheduler(self, optim: torch.optim.Optimizer) -> LRScheduler:
        return LambdaLR(optim, lr_lambda=self._LRLambda(self).compute)

    class _LRLambda:
        def __init__(self, parent: "LRSchedulerFactoryLinear"):
            self.max_update_num = (
                np.ceil(parent.step_per_epoch / parent.step_per_collect) * parent.num_epochs
            )

        def compute(self, epoch: int) -> float:
            return 1.0 - epoch / self.max_update_num


class OptimizerFactory(ABC, ToStringMixin):
    def __init__(self) -> None:
        self.lr_scheduler_factory: LRSchedulerFactory | None = None

    def with_lr_scheduler_factory(self, lr_scheduler_factory: LRSchedulerFactory) -> Self:
        self.lr_scheduler_factory = lr_scheduler_factory
        return self

    def create_instances(
        self,
        module: torch.nn.Module,
    ) -> tuple[torch.optim.Optimizer, LRScheduler | None]:
        optimizer = self._create_optimizer_for_params(module.parameters())
        lr_scheduler = None
        if self.lr_scheduler_factory is not None:
            lr_scheduler = self.lr_scheduler_factory.create_scheduler(optimizer)
        return optimizer, lr_scheduler

    @abstractmethod
    def _create_optimizer_for_params(self, params: ParamsType) -> torch.optim.Optimizer:
        pass


class TorchOptimizerFactory(OptimizerFactory):
    """General factory for arbitrary torch optimizers."""

    def __init__(self, optim_class: Callable[..., torch.optim.Optimizer], **kwargs: Any):
        """

        :param optim_class: the optimizer class (e.g. subclass of `torch.optim.Optimizer`),
            which will be passed the module parameters, the learning rate as `lr` and the
            kwargs provided.
        :param kwargs: keyword arguments to provide at optimizer construction
        """
        super().__init__()
        self.optim_class = optim_class
        self.kwargs = kwargs

    def _create_optimizer_for_params(self, params: ParamsType) -> torch.optim.Optimizer:
        return self.optim_class(params, **self.kwargs)


class AdamOptimizerFactory(OptimizerFactory):
    def __init__(
        self,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0,
    ):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps
        self.betas = betas

    def _create_optimizer_for_params(self, params: ParamsType) -> torch.optim.Optimizer:
        return Adam(
            params,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )


class RMSpropOptimizerFactory(OptimizerFactory):
    def __init__(
        self,
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-08,
        weight_decay: float = 0,
        momentum: float = 0,
        centered: bool = False,
    ):
        super().__init__()
        self.lr = lr
        self.alpha = alpha
        self.momentum = momentum
        self.centered = centered
        self.weight_decay = weight_decay
        self.eps = eps

    def _create_optimizer_for_params(self, params: ParamsType) -> torch.optim.Optimizer:
        return RMSprop(
            params,
            lr=self.lr,
            alpha=self.alpha,
            eps=self.eps,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            centered=self.centered,
        )
