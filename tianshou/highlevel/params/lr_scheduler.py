from abc import ABC, abstractmethod

from sensai.util.string import ToStringMixin

from tianshou.highlevel.config import TrainingConfig
from tianshou.algorithm.optim import LRSchedulerFactory, LRSchedulerFactoryLinear


class LRSchedulerFactoryFactory(ToStringMixin, ABC):
    """Factory for the creation of a learning rate scheduler factory."""

    @abstractmethod
    def create_lr_scheduler_factory(self) -> LRSchedulerFactory:
        pass


class LRSchedulerFactoryFactoryLinear(LRSchedulerFactoryFactory):
    def __init__(self, training_config: TrainingConfig):
        self.training_config = training_config

    def create_lr_scheduler_factory(self) -> LRSchedulerFactory:
        if (
            self.training_config.step_per_epoch is None
            or self.training_config.step_per_collect is None
        ):
            raise ValueError(
                f"{self.__class__.__name__} requires step_per_epoch and step_per_collect to be set "
                f"in order for the scheduling to be well-defined."
            )
        return LRSchedulerFactoryLinear(
            num_epochs=self.training_config.num_epochs,
            step_per_epoch=self.training_config.step_per_epoch,
            step_per_collect=self.training_config.step_per_collect,
        )
