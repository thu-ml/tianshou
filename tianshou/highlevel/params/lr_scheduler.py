from abc import ABC, abstractmethod

from sensai.util.string import ToStringMixin

from tianshou.highlevel.config import SamplingConfig
from tianshou.policy.optim import LRSchedulerFactory, LRSchedulerFactoryLinear


class LRSchedulerFactoryFactory(ToStringMixin, ABC):
    """Factory for the creation of a learning rate scheduler factory."""

    @abstractmethod
    def create_lr_scheduler_factory(self) -> LRSchedulerFactory:
        pass


class LRSchedulerFactoryFactoryLinear(LRSchedulerFactoryFactory):
    def __init__(self, sampling_config: SamplingConfig):
        self.sampling_config = sampling_config

    def create_lr_scheduler_factory(self) -> LRSchedulerFactory:
        return LRSchedulerFactoryLinear(
            num_epochs=self.sampling_config.num_epochs,
            step_per_epoch=self.sampling_config.step_per_epoch,
            step_per_collect=self.sampling_config.step_per_collect,
        )
