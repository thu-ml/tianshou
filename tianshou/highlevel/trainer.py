from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from tianshou.highlevel.env import Environments
from tianshou.highlevel.logger import Logger
from tianshou.policy import BasePolicy
from tianshou.utils.string import ToStringMixin

TPolicy = TypeVar("TPolicy", bound=BasePolicy)


class TrainingContext:
    def __init__(self, policy: TPolicy, envs: Environments, logger: Logger):
        self.policy = policy
        self.envs = envs
        self.logger = logger


class TrainerEpochCallback(ToStringMixin, ABC):
    """Callback which is called at the beginning of each epoch."""

    @abstractmethod
    def callback(self, epoch: int, env_step: int, context: TrainingContext) -> None:
        pass

    def get_trainer_fn(self, context: TrainingContext) -> Callable[[int, int], None]:
        def fn(epoch, env_step):
            return self.callback(epoch, env_step, context)

        return fn


class TrainerStopCallback(ToStringMixin, ABC):
    """Callback indicating whether training should stop."""

    @abstractmethod
    def should_stop(self, mean_rewards: float, context: TrainingContext) -> bool:
        """:param mean_rewards: the average undiscounted returns of the testing result
        :return: True if the goal has been reached and training should stop, False otherwise
        """

    def get_trainer_fn(self, context: TrainingContext) -> Callable[[float], bool]:
        def fn(mean_rewards: float):
            return self.should_stop(mean_rewards, context)

        return fn


@dataclass
class TrainerCallbacks:
    epoch_callback_train: TrainerEpochCallback | None = None
    epoch_callback_test: TrainerEpochCallback | None = None
    stop_callback: TrainerStopCallback | None = None
