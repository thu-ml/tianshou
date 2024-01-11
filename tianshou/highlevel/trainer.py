from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar, cast

from tianshou.highlevel.env import Environments
from tianshou.highlevel.logger import TLogger
from tianshou.policy import BasePolicy, DQNPolicy
from tianshou.utils.string import ToStringMixin

TPolicy = TypeVar("TPolicy", bound=BasePolicy)


class TrainingContext:
    def __init__(self, policy: TPolicy, envs: Environments, logger: TLogger):
        self.policy = policy
        self.envs = envs
        self.logger = logger


class TrainerEpochCallbackTrain(ToStringMixin, ABC):
    """Callback which is called at the beginning of each epoch."""

    @abstractmethod
    def callback(self, epoch: int, env_step: int, context: TrainingContext) -> None:
        pass

    def get_trainer_fn(self, context: TrainingContext) -> Callable[[int, int], None]:
        def fn(epoch: int, env_step: int) -> None:
            return self.callback(epoch, env_step, context)

        return fn


class TrainerEpochCallbackTest(ToStringMixin, ABC):
    """Callback which is called at the beginning of each epoch."""

    @abstractmethod
    def callback(self, epoch: int, env_step: int | None, context: TrainingContext) -> None:
        pass

    def get_trainer_fn(self, context: TrainingContext) -> Callable[[int, int | None], None]:
        def fn(epoch: int, env_step: int | None) -> None:
            return self.callback(epoch, env_step, context)

        return fn


class TrainerStopCallback(ToStringMixin, ABC):
    """Callback indicating whether training should stop."""

    @abstractmethod
    def should_stop(self, mean_rewards: float, context: TrainingContext) -> bool:
        """Determines whether training should stop.

        :param mean_rewards: the average undiscounted returns of the testing result
        :param context: the training context
        :return: True if the goal has been reached and training should stop, False otherwise
        """

    def get_trainer_fn(self, context: TrainingContext) -> Callable[[float], bool]:
        def fn(mean_rewards: float) -> bool:
            return self.should_stop(mean_rewards, context)

        return fn


@dataclass
class TrainerCallbacks:
    """Container for callbacks used during training."""

    epoch_callback_train: TrainerEpochCallbackTrain | None = None
    epoch_callback_test: TrainerEpochCallbackTest | None = None
    stop_callback: TrainerStopCallback | None = None


class TrainerEpochCallbackTrainDQNSetEps(TrainerEpochCallbackTrain):
    """Sets the epsilon value for DQN-based policies at the beginning of the training
    stage in each epoch.
    """

    def __init__(self, eps_test: float):
        self.eps_test = eps_test

    def callback(self, epoch: int, env_step: int, context: TrainingContext) -> None:
        policy = cast(DQNPolicy, context.policy)
        policy.set_eps(self.eps_test)


class TrainerEpochCallbackTrainDQNEpsLinearDecay(TrainerEpochCallbackTrain):
    """Sets the epsilon value for DQN-based policies at the beginning of the training
    stage in each epoch, using a linear decay in the first `decay_steps` steps.
    """

    def __init__(self, eps_train: float, eps_train_final: float, decay_steps: int = 1000000):
        self.eps_train = eps_train
        self.eps_train_final = eps_train_final
        self.decay_steps = decay_steps

    def callback(self, epoch: int, env_step: int, context: TrainingContext) -> None:
        policy = cast(DQNPolicy, context.policy)
        logger = context.logger
        if env_step <= self.decay_steps:
            eps = self.eps_train - env_step / self.decay_steps * (
                self.eps_train - self.eps_train_final
            )
        else:
            eps = self.eps_train_final
        policy.set_eps(eps)
        logger.write("train/env_step", env_step, {"train/eps": eps})


class TrainerEpochCallbackTestDQNSetEps(TrainerEpochCallbackTest):
    """Sets the epsilon value for DQN-based policies at the beginning of the test
    stage in each epoch.
    """

    def __init__(self, eps_test: float):
        self.eps_test = eps_test

    def callback(self, epoch: int, env_step: int | None, context: TrainingContext) -> None:
        policy = cast(DQNPolicy, context.policy)
        policy.set_eps(self.eps_test)
