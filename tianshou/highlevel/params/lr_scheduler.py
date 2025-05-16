from abc import ABC, abstractmethod

from sensai.util.string import ToStringMixin

from tianshou.algorithm.optim import LRSchedulerFactory, LRSchedulerFactoryLinear
from tianshou.highlevel.config import TrainingConfig


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
            self.training_config.epoch_num_steps is None
            or self.training_config.collection_step_num_env_steps is None
        ):
            raise ValueError(
                f"{self.__class__.__name__} requires epoch_num_steps and collection_step_num_env_steps to be set "
                f"in order for the scheduling to be well-defined."
            )
        return LRSchedulerFactoryLinear(
            max_epochs=self.training_config.max_epochs,
            epoch_num_steps=self.training_config.epoch_num_steps,
            collection_step_num_env_steps=self.training_config.collection_step_num_env_steps,
        )
