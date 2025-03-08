"""
This module contains Tianshou's trainer classes, which orchestrate the training and call upon an RL algorithm's
specific network updating logic to perform the actual gradient updates.

Training is structured as follows (hierarchical glossary):
- **epoch**: The outermost iteration level of the training loop. Each epoch consists of a number of training steps
  and one test step (see :attr:`TrainingConfig.max_epoch` for a detailed explanation):
    - **training step**: A training step performs the steps necessary in order to apply a single update of the neural
      network components as defined by the underlying RL algorithm (:class:`Algorithm`). This involves the following sub-steps:
        - for online learning algorithms:
            - **collection step**: collecting environment steps/transitions to be used for training.
            - (potentially) a test step (see below) if the early stopping criterion is satisfied based on
              the data collected (see :attr:`OnlineTrainingConfig.test_in_train`).
        - **update step**: applying the actual gradient updates using the RL algorithm.
          The update is based on either ...
            - data from only the preceding collection step (on-policy learning),
            - data from the collection step and previously collected data (off-policy learning), or
            - data from the user-provided replay buffer (offline learning).
      For offline learning algorithms, a training step is thus equivalent to an update step.
    - **test step**: Collects test episodes from dedicated test environments which are used to evaluate the performance
      of the policy. Optionally, the performance result can be used to determine whether training shall stop early
      (see :attr:`TrainingConfig.stop_fn`).
"""
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
import tqdm
from sensai.util.helper import count_none
from sensai.util.string import ToStringMixin

from tianshou.data import (
    AsyncCollector,
    CollectStats,
    EpochStats,
    InfoStats,
    ReplayBuffer,
    SequenceSummaryStats,
)
from tianshou.data.buffer.base import MalformedBufferError
from tianshou.data.collector import BaseCollector, CollectStatsBase
from tianshou.policy.base import TrainingStats
from tianshou.trainer.utils import gather_info, test_episode
from tianshou.utils import (
    BaseLogger,
    LazyLogger,
    MovAvg,
)
from tianshou.utils.logging import set_numerical_fields_to_precision
from tianshou.utils.torch_utils import policy_within_training_step

if TYPE_CHECKING:
    from tianshou.policy import Algorithm

log = logging.getLogger(__name__)


@dataclass(kw_only=True)
class TrainingConfig(ToStringMixin):
    max_epoch: int = 100
    """
    the (maximum) number of epochs to run training for. An **epoch** is the outermost iteration level and each
    epoch consists of a number of training steps and one test step, where each training step

      * [for the online case] collects environment steps/transitions (**collection step**),
        adding them to the (replay) buffer (see :attr:`step_per_collect` and :attr:`episode_per_collect`)
      * performs an **update step** via the RL algorithm being used, which can involve
        one or more actual gradient updates, depending on the algorithm

    and the test step collects :attr:`num_episodes_per_test` test episodes in order to evaluate
    agent performance.

    Training may be stopped early if the stop criterion is met (see :attr:`stop_fn`).

    For online training, the number of training steps in each epoch is indirectly determined by
    :attr:`step_per_epoch`: As many training steps will be performed as are required in
    order to reach :attr:`step_per_epoch` total steps in the training environments.
    Specifically, if the number of transitions collected per step is `c` (see
    :attr:`step_per_collect`) and :attr:`step_per_epoch` is set to `s`, then the number
    of training steps per epoch is `ceil(s / c)`.
    Therefore, if `num_epochs = e`, the total number of environment steps taken during training
    can be computed as `e * ceil(s / c) * c`.

    For offline training, the number of training steps per epoch is equal to :attr:`step_per_epoch`.
    """

    step_per_epoch: int = 30000
    """
    for an online algorithm, this is the total number of environment steps to be collected per epoch, and,
    for an offline algorithm, it is the total number of training steps to take per epoch.
    See :attr:`num_epochs` for an explanation of epoch semantics.
    """

    test_collector: BaseCollector | None = None
    """
    the collector to use for test episode collection (test steps); if None, perform no test steps.
    """

    episode_per_test: int = 1
    """the number of episodes to collect in each test step.
    """

    train_fn: Callable[[int, int], None] | None = None
    """
    a callback function which is called at the beginning of each training step.
    It can be used to perform custom additional operations, with the
    signature ``f(num_epoch: int, step_idx: int) -> None``.
    """

    test_fn: Callable[[int, int | None], None] | None = None
    """
    a callback function to be called at the beginning of each test step.
    It can be used to perform custom additional operations, with the
    signature ``f(num_epoch: int, step_idx: int) -> None``.
    """

    stop_fn: Callable[[float], bool] | None = None
    """
    a callback function with signature ``f(score: float) -> bool``, which
    is used to decide whether training shall be stopped early based on the score
    achieved in a test step.
    The score it receives is computed by the :attr:`compute_score_fn` callback
    (which defaults to the mean reward if the function is not provided).

    Requires test steps to be activated and thus :attr:`test_collector` to be set.

    Note: The function is also used when :attr:`test_in_train` is activated (see docstring).
    """

    compute_score_fn: Callable[[CollectStats], float] | None = None
    """
    the callback function to use in order to compute the test batch performance score, which is used to
    determine what the best model is (score is maximized); if None, use the mean reward.
    """

    save_best_fn: Callable[["Algorithm"], None] | None = None
    """
    the callback function to call in order to save the best model whenever a new best score (see :attr:`compute_score_fn`)
    is achieved in a test step. It should have the signature ``f(policy: BasePolicy) -> None``.
    """

    save_checkpoint_fn: Callable[[int, int, int], str] | None = None
    """
    the callback function with which to save checkpoint data after each training step,
    which can save whatever data is desired to a file and returns the path of the file.
    Signature: ``f(epoch: int, env_step: int, gradient_step: int) -> str``.
    """

    resume_from_log: bool = False
    """
    whether to load env_step/gradient_step and other metadata from the existing log,
    which is given in :attr:`logger`.
    """

    reward_metric: Callable[[np.ndarray], np.ndarray] | None = None
    """
    a function with signature
    ``f(rewards: np.ndarray with shape (num_episode, agent_num)) -> np.ndarray with shape (num_episode,)``,
    which is used in multi-agent RL. We need to return a single scalar for each episode's result
    to monitor training in the multi-agent RL setting. This function specifies what is the desired metric,
    e.g., the reward of agent 1 or the average reward over all agents.
    """

    logger: BaseLogger | None = None
    """
    the logger with which to log statistics during training/testing/updating. To not log anything, use None.
    """

    verbose: bool = True
    """
    whether to print status information to stdout.
    If set to False, status information will still be logged (provided that logging is enabled via the
    `logging` Python module).
    """

    show_progress: bool = True
    """
    whether to display a progress bars during training.
    """

    def __post_init__(self):
        if self.resume_from_log and self.logger is None:
            raise ValueError("Cannot resume from log without a logger being provided")
        if self.test_collector is None:
            if self.stop_fn is not None:
                raise ValueError(
                    "stop_fn cannot be activated without test steps being enabled (test_collector being set)"
                )
            if self.test_fn is not None:
                raise ValueError(
                    "test_fn is set while test steps are disabled (test_collector is None)"
                )
            if self.save_best_fn is not None:
                raise ValueError(
                    "save_best_fn is set while test steps are disabled (test_collector is None)"
                )


@dataclass(kw_only=True)
class OnlineTrainingConfig(TrainingConfig):
    train_collector: BaseCollector
    """
    the collector with which to gather new data for training in each training step
    """

    step_per_collect: int | None = 2048
    """
    the number of environment steps/transitions to collect in each collection step before the
    network update within each training step.

    This is mutually exclusive with :attr:`episode_per_collect`, and one of the two must be set.

    Note that the exact number can be reached only if this is a multiple of the number of
    training environments being used, as each training environment will produce the same
    (non-zero) number of transitions.
    Specifically, if this is set to `n` and `m` training environments are used, then the total
    number of transitions collected per collection step is `ceil(n / m) * m =: c`.

    See :attr:`num_epochs` for information on the total number of environment steps being
    collected during training.
    """

    episode_per_collect: int | None = None
    """
    the number of episodes to collect in each collection step before the network update within
    each training step. If this is set, the number of environment steps collected in each
    collection step is the sum of the lengths of the episodes collected.

    This is mutually exclusive with :attr:`step_per_collect`, and one of the two must be set.
    """

    test_in_train: bool = True
    """
    Whether to apply an effective test step triggered by the early stopping criterion (given by :attr:`stop_fn`)
    being satisfied in the data collected in the collect step within a training step:
    If the stop criterion is satisfied, it collects `episode_per_test` test episodes (as in a test step)
    and determines whether the stop criterion is also satisfied by the episodes thus collected,
    and if so, training stops early.
    """

    def __post_init__(self):
        super().__post_init__()
        if count_none(self.step_per_collect, self.episode_per_collect) != 1:
            raise ValueError("Exactly one of {step_per_collect, episode_per_collect} must be set")
        if self.test_in_train and (self.test_collector is None or self.stop_fn is None):
            raise ValueError("test_in_train requires test_collector and stop_fn to be set")


@dataclass(kw_only=True)
class OnPolicyTrainingConfig(OnlineTrainingConfig):
    batch_size: int | None = 64
    """
    Use mini-batches of this size for gradient updates (causing the gradient to be less accurate,
    a form of regularization).
    Set ``batch_size=None`` for the full buffer that was collected within the training step to be
    used for the gradient update (no mini-batching).
    """

    repeat_per_collect: int = 1
    """
    controls, within one update step of an on-policy algorithm, the number of times
    the full collected data is applied for gradient updates, i.e. if the parameter is
    5, then the collected data shall be used five times to update the policy within the same
    update step.
    """


@dataclass(kw_only=True)
class OffPolicyTrainingConfig(OnlineTrainingConfig):
    batch_size: int = 64
    """
    the the number of environment steps/transitions to sample from the buffer for a gradient update.
    """

    # TODO: Given our glossary, this is confusingly named. Should definitely contain the word "gradient"
    update_per_step: float = 1.0
    """
    the number of gradient steps to perform per sample collected (see :attr:`step_per_collect`).
    Specifically, if this is set to `u` and the number of samples collected in the preceding
    collection step is `n`, then `round(u * n)` gradient steps will be performed.
    """


@dataclass(kw_only=True)
class OfflineTrainingConfig(TrainingConfig):
    buffer: ReplayBuffer
    """
    the replay buffer with environment steps to use as training data for offline learning.
    This buffer will be pre-processed using the RL algorithm's pre-processing
    function (if any) before training.
    """

    batch_size: int = 64
    """
    the the number of environment steps/transitions to sample from the buffer for a gradient update.
    """


TTrainingConfig = TypeVar("TTrainingConfig", bound=TrainingConfig)
TOnlineTrainingConfig = TypeVar("TOnlineTrainingConfig", bound=OnlineTrainingConfig)


class Trainer(Generic[TTrainingConfig], ABC):
    """
    Base class for trainers in Tianshou, which orchestrate the training process and call upon an RL algorithm's
    specific network updating logic to perform the actual gradient updates.

    The base class already implements the fundamental epoch logic and fully implements the test step
    logic, which is common to all trainers. The training step logic is left to be implemented by subclasses.
    """

    def __init__(
        self,
        policy: "Algorithm",
        config: TTrainingConfig,
    ):
        self.algorithm = policy
        self.config = config

        self._logger = config.logger or LazyLogger()

        self._start_time = time.time()
        self._stat: defaultdict[str, MovAvg] = defaultdict(MovAvg)
        self._best_score = 0.0
        self._best_reward = 0.0
        self._best_reward_std = 0.0
        self._start_epoch = 0
        # This is only used for logging but creeps into the implementations
        # of the trainers. I believe it would be better to remove
        self._gradient_step = 0
        self._env_step = 0
        """
        the step counter which is used to track progress of the training process.
        For online learning (i.e. on-policy and off-policy learning), this is the total number of
        environment steps collected, and for offline training, it is the total number of environment
        steps that have been sampled from the replay buffer to perform gradient updates.
        """
        self._policy_update_time = 0.0

        self._compute_score_fn: Callable[[CollectStats], float] = (
            config.compute_score_fn or self._compute_score_fn_default
        )

        self._epoch = self._start_epoch
        self._best_epoch = self._start_epoch
        self._stop_fn_flag = False

    @staticmethod
    def _compute_score_fn_default(stat: CollectStats) -> float:
        """
        The default score function, which returns the mean return/reward.

        :param stat: the collection stats
        :return: the mean return
        """
        assert stat.returns_stat is not None  # for mypy
        return stat.returns_stat.mean

    @property
    def _pbar(self) -> Callable[..., tqdm.tqdm]:
        """Use as context manager or iterator, i.e., `with self._pbar(...) as t:` or `for _ in self._pbar(...):`."""
        return partial(
            tqdm.tqdm,
            dynamic_ncols=True,
            ascii=True,
            disable=not self.config.show_progress,
        )

    def _reset_collectors(self, reset_buffer: bool = False) -> None:
        if self.config.test_collector is not None:
            self.config.test_collector.reset(reset_buffer=reset_buffer)

    def reset(self, reset_collectors: bool = True, reset_collector_buffers: bool = False) -> None:
        """Initializes the training process.

        :param reset_collectors: whether to reset the collectors prior to starting the training process.
            Specifically, this will reset the environments in the collectors (starting new episodes),
            and the statistics stored in the collector. Whether the contained buffers will be reset/cleared
            is determined by the `reset_buffer` parameter.
        :param reset_collector_buffers: whether, for the case where the collectors are reset, to reset/clear the
            contained buffers as well.
            This has no effect if `reset_collectors` is False.
        """
        self._env_step = 0
        if self.config.resume_from_log:
            (
                self._start_epoch,
                self._env_step,
                self._gradient_step,
            ) = self._logger.restore_data()

        self._start_time = time.time()

        if reset_collectors:
            self._reset_collectors(reset_buffer=reset_collector_buffers)

        if self.config.test_collector is not None:
            assert self.config.episode_per_test is not None
            assert not isinstance(self.config.test_collector, AsyncCollector)  # Issue 700
            test_result = test_episode(
                self.config.test_collector,
                self.config.test_fn,
                self._start_epoch,
                self.config.episode_per_test,
                self._logger,
                self._env_step,
                self.config.reward_metric,
            )
            assert test_result.returns_stat is not None  # for mypy
            self._best_epoch = self._start_epoch
            self._best_reward, self._best_reward_std = (
                test_result.returns_stat.mean,
                test_result.returns_stat.std,
            )
            self._best_score = self._compute_score_fn(test_result)
        if self.config.save_best_fn:
            self.config.save_best_fn(self.algorithm)

        self._epoch = self._start_epoch
        self._stop_fn_flag = False

    class _TrainingStepResult(ABC):
        @abstractmethod
        def get_steps_in_epoch_advancement(self):
            """
            :return: the number of steps that were done within the epoch, where the concrete semantics
                of what a step is depend on the type of algorith. See docstring of `TrainingConfig.step_per_epoch`.
            """

        @abstractmethod
        def get_collect_stats(self) -> CollectStats | None:
            pass

        @abstractmethod
        def get_training_stats(self) -> TrainingStats | None:
            pass

        @abstractmethod
        def is_training_done(self):
            """:return: whether the early stopping criterion is satisfied and training shall stop."""

        @abstractmethod
        def get_env_step_advancement(self) -> int:
            """
            :return: the number of steps by which to advance the env_step counter in the trainer (see docstring
                of trainer attribute). The semantics depend on the type of the algorithm.
            """

    @abstractmethod
    def _create_epoch_pbar_data_dict(
        self, training_step_result: _TrainingStepResult
    ) -> dict[str, str]:
        pass

    def execute_epoch(self) -> EpochStats:
        self._epoch += 1

        # perform the required number of steps for the epoch (`step_per_epoch`)
        steps_done_in_this_epoch = 0
        train_collect_stats, training_stats = None, None
        with self._pbar(
            total=self.config.step_per_epoch, desc=f"Epoch #{self._epoch}", position=1
        ) as t:
            while steps_done_in_this_epoch < self.config.step_per_epoch and not self._stop_fn_flag:
                # perform a training step and update progress
                training_step_result = self._training_step()
                steps_done_in_this_epoch += training_step_result.get_steps_in_epoch_advancement()
                t.update(training_step_result.get_steps_in_epoch_advancement())
                self._stop_fn_flag = training_step_result.is_training_done()
                self._env_step += training_step_result.get_env_step_advancement()

                collect_stats = training_step_result.get_collect_stats()
                if collect_stats is not None:
                    self._logger.log_train_data(asdict(collect_stats), self._env_step)
                training_stats = training_step_result.get_training_stats()

                pbar_data_dict = self._create_epoch_pbar_data_dict(training_step_result)
                pbar_data_dict = set_numerical_fields_to_precision(pbar_data_dict)
                pbar_data_dict["gradient_step"] = str(self._gradient_step)
                t.set_postfix(**pbar_data_dict)

        test_collect_stats = None
        if not self._stop_fn_flag:
            self._logger.save_data(
                self._epoch,
                self._env_step,
                self._gradient_step,
                self.config.save_checkpoint_fn,
            )

            # test step
            if self.config.test_collector is not None:
                test_collect_stats, self._stop_fn_flag = self._test_step()

        info_stats = gather_info(
            start_time=self._start_time,
            policy_update_time=self._policy_update_time,
            gradient_step=self._gradient_step,
            best_score=self._best_score,
            best_reward=self._best_reward,
            best_reward_std=self._best_reward_std,
            train_collector=self.config.train_collector
            if isinstance(self.config, OnlineTrainingConfig)
            else None,
            test_collector=self.config.test_collector,
        )

        self._logger.log_info_data(asdict(info_stats), self._epoch)

        return EpochStats(
            epoch=self._epoch,
            train_collect_stat=train_collect_stats,
            test_collect_stat=test_collect_stats,
            training_stat=training_stats,
            info_stat=info_stats,
        )

    def _test_step(self) -> tuple[CollectStats, bool]:
        """Perform one test step."""
        assert self.config.episode_per_test is not None
        assert self.config.test_collector is not None
        stop_fn_flag = False
        test_stat = test_episode(
            self.config.test_collector,
            self.config.test_fn,
            self._epoch,
            self.config.episode_per_test,
            self._logger,
            self._env_step,
            self.config.reward_metric,
        )
        assert test_stat.returns_stat is not None  # for mypy
        rew, rew_std = test_stat.returns_stat.mean, test_stat.returns_stat.std
        score = self._compute_score_fn(test_stat)
        if self._best_epoch < 0 or self._best_score < score:
            self._best_score = score
            self._best_epoch = self._epoch
            self._best_reward = float(rew)
            self._best_reward_std = rew_std
            if self.config.save_best_fn:
                self.config.save_best_fn(self.algorithm)
        cur_info, best_info = "", ""
        if score != rew:
            cur_info, best_info = f", score: {score: .6f}", f", best_score: {self._best_score:.6f}"
        log_msg = (
            f"Epoch #{self._epoch}: test_reward: {rew:.6f} ± {rew_std:.6f},{cur_info}"
            f" best_reward: {self._best_reward:.6f} ± "
            f"{self._best_reward_std:.6f}{best_info} in #{self._best_epoch}"
        )
        log.info(log_msg)
        if self.config.verbose:
            print(log_msg, flush=True)

        if self.config.stop_fn and self.config.stop_fn(self._best_reward):
            stop_fn_flag = True

        return test_stat, stop_fn_flag

    @abstractmethod
    def _training_step(self) -> _TrainingStepResult:
        """Performs one training step."""

    # TODO: move moving average computation and logging into its own logger
    # TODO: maybe think about a command line logger instead of always printing data dict
    def _update_moving_avg_stats_and_log_update_data(self, update_stat: TrainingStats) -> None:
        """Log losses, update moving average stats, and also modify the smoothed_loss in update_stat."""
        cur_losses_dict = update_stat.get_loss_stats_dict()
        update_stat.smoothed_loss = self._update_moving_avg_stats_and_get_averaged_data(
            cur_losses_dict,
        )
        self._logger.log_update_data(asdict(update_stat), self._gradient_step)

    # TODO: seems convoluted, there should be a better way of dealing with the moving average stats
    def _update_moving_avg_stats_and_get_averaged_data(
        self,
        data: dict[str, float],
    ) -> dict[str, float]:
        """Add entries to the moving average object in the trainer and retrieve the averaged results.

        :param data: any entries to be tracked in the moving average object.
        :return: A dictionary containing the averaged values of the tracked entries.

        """
        smoothed_data = {}
        for key, loss_item in data.items():
            self._stat[key].add(loss_item)
            smoothed_data[key] = self._stat[key].get()
        return smoothed_data

    def run(
        self, reset_collectors: bool = True, reset_collector_buffers: bool = False
    ) -> InfoStats:
        """Runs the training process with the configuration given at construction.

        :param reset_collectors: whether to reset the collectors prior to starting the training process.
            Specifically, this will reset the environments in the collectors (starting new episodes),
            and the statistics stored in the collector. Whether the contained buffers will be reset/cleared
            is determined by the `reset_buffer` parameter.
        :param reset_collector_buffers: whether, for the case where the collectors are reset, to reset/clear the
            contained buffers as well.
            This has no effect if `reset_collectors` is False.
        """
        self.reset(
            reset_collectors=reset_collectors, reset_collector_buffers=reset_collector_buffers
        )

        while self._epoch < self.config.max_epoch and not self._stop_fn_flag:
            self.execute_epoch()

        return gather_info(
            start_time=self._start_time,
            policy_update_time=self._policy_update_time,
            gradient_step=self._gradient_step,
            best_score=self._best_score,
            best_reward=self._best_reward,
            best_reward_std=self._best_reward_std,
            train_collector=self.config.train_collector
            if isinstance(self.config, OnlineTrainingConfig)
            else None,
            test_collector=self.config.test_collector,
        )


class OfflineTrainer(Trainer[OfflineTrainingConfig]):
    """An offline trainer, which samples mini-batches from a given buffer and passes them to
    the algorithm's update function.
    """

    def __init__(
        self,
        policy: "Algorithm",
        config: OfflineTrainingConfig,
    ):
        super().__init__(policy, config)
        self._buffer = policy.process_buffer(self.config.buffer)

    class _TrainingStepResult(Trainer._TrainingStepResult):
        def __init__(self, training_stats: TrainingStats, env_step_advancement: int):
            self._training_stats = training_stats
            self._env_step_advancement = env_step_advancement

        def get_steps_in_epoch_advancement(self):
            return 1

        def get_collect_stats(self) -> None:
            return None

        def get_training_stats(self) -> TrainingStats:
            return self._training_stats

        def is_training_done(self) -> bool:
            return False

        def get_env_step_advancement(self) -> int:
            return self._env_step_advancement

    def _training_step(self) -> _TrainingStepResult:
        with policy_within_training_step(self.algorithm.policy):
            self._gradient_step += 1
            # Note: since sample_size=batch_size, this will perform
            # exactly one gradient step. This is why we don't need to calculate the
            # number of gradient steps, like in the on-policy case.
            training_stats = self.algorithm.update(
                sample_size=self.config.batch_size, buffer=self._buffer
            )
            self._update_moving_avg_stats_and_log_update_data(training_stats)
            self._policy_update_time += training_stats.train_time
            return self._TrainingStepResult(
                training_stats=training_stats, env_step_advancement=self.config.batch_size
            )

    def _create_epoch_pbar_data_dict(
        self, training_step_result: _TrainingStepResult
    ) -> dict[str, str]:
        return {}


class OnlineTrainer(Trainer[TOnlineTrainingConfig], Generic[TOnlineTrainingConfig], ABC):
    """
    An online trainer, which collects data from the environment in each training step and
    uses the collected data to perform an update step, the nature of which is to be defined
    in subclasses.
    """

    def __init__(
        self,
        policy: "Algorithm",
        config: OnlineTrainingConfig,
    ):
        super().__init__(policy, config)
        self._env_episode = 0
        """
        the total number of episodes collected in the environment
        """

    def _reset_collectors(self, reset_buffer: bool = False) -> None:
        super()._reset_collectors(reset_buffer=reset_buffer)
        self.config.train_collector.reset(reset_buffer=reset_buffer)

    def reset(self, reset_collectors: bool = True, reset_collector_buffers: bool = False) -> None:
        super().reset(
            reset_collectors=reset_collectors, reset_collector_buffers=reset_collector_buffers
        )

        if (
            self.config.test_in_train
            and self.config.train_collector.algorithm is not self.algorithm
        ):
            log.warning(
                "The training data collector's algorithm is not the same as the one being trained, "
                "yet test_in_train is enabled. This may lead to unexpected results."
            )

        self._env_episode = 0

    class _TrainingStepResult(Trainer._TrainingStepResult):
        def __init__(
            self,
            collect_stats: CollectStats,
            training_stats: TrainingStats | None,
            is_training_done: bool,
        ):
            self._collect_stats = collect_stats
            self._training_stats = training_stats
            self._is_training_done = is_training_done

        def get_steps_in_epoch_advancement(self):
            return self.get_env_step_advancement()

        def get_collect_stats(self) -> CollectStats:
            return self._collect_stats

        def get_training_stats(self) -> TrainingStats | None:
            return self._training_stats

        def is_training_done(self):
            return self._is_training_done

        def get_env_step_advancement(self) -> int:
            return self._collect_stats.n_collected_steps

    def _training_step(self) -> _TrainingStepResult:
        """Perform one training step.

        For an online algorithm, a training step involves:
          * collecting data
          * for the case where `test_in_train` is activated,
            determining whether the stop condition has been reached
            (and returning without performing any actual training if so)
          * performing a gradient update step
        """
        with policy_within_training_step(self.algorithm.policy):
            # collect data
            collect_stats = self._collect_training_data()

            # determine whether we should stop training based on the data collected
            should_stop_training = self._test_in_train(
                collect_stats,
            )

            # perform gradient update step (if not already done)
            training_stats: TrainingStats | None = None
            if not should_stop_training:
                training_stats = self._update_step(collect_stats)

            return self._TrainingStepResult(
                collect_stats=collect_stats,
                training_stats=training_stats,
                is_training_done=should_stop_training,
            )

    def _collect_training_data(self) -> CollectStats:
        """Performs training data collection.

        :return: the data collection stats
        """
        assert self.config.episode_per_test is not None
        assert self.config.train_collector is not None

        if self.config.train_fn:
            self.config.train_fn(self._epoch, self._env_step)

        collect_stats = self.config.train_collector.collect(
            n_step=self.config.step_per_collect,
            n_episode=self.config.episode_per_collect,
        )

        if self.config.train_collector.buffer.hasnull():
            from tianshou.data.collector import EpisodeRolloutHook
            from tianshou.env import DummyVectorEnv

            raise MalformedBufferError(
                f"Encountered NaNs in buffer after {self._env_step} steps."
                f"Such errors are usually caused by either a bug in the environment or by "
                f"problematic implementations {EpisodeRolloutHook.__class__.__name__}. "
                f"For debugging such issues it is recommended to run the training in a single process, "
                f"e.g., by using {DummyVectorEnv.__class__.__name__}.",
            )

        if collect_stats.n_collected_episodes > 0:
            assert collect_stats.returns_stat is not None  # for mypy
            assert collect_stats.lens_stat is not None  # for mypy
            if self.config.reward_metric:  # TODO: move inside collector
                rew = self.config.reward_metric(collect_stats.returns)
                collect_stats.returns = rew
                collect_stats.returns_stat = SequenceSummaryStats.from_sequence(rew)

        # update collection stats specific to this specialization
        self._env_episode += collect_stats.n_collected_episodes

        return collect_stats

    def _test_in_train(
        self,
        collect_stats: CollectStats,
    ) -> bool:
        """
        Performs performance testing based on the early stopping criterion being satisfied based on the
        data collected in the current training step:
        If the stop criterion is satisfied, it collects `episode_per_test` test episodes (as in a test step)
        and determines whether the stop criterion is also satisfied by the episodes thus collected,
        and if so, returns True, indicating that training stops early.

        Therefore, if the early stopping criterion is satisfied on the data collected for training,
        this effectively carries out a test step and updates the respective metrics (best_reward, etc.)
        accordingly.

        :param collect_stats: the data collection stats from the preceding collection step
        :return: flag indicating whether to stop training early
        """
        should_stop_training = False

        # Because we need to evaluate the policy, we need to temporarily leave the "is_training_step" semantics
        with policy_within_training_step(self.algorithm.policy, enabled=False):
            if (
                collect_stats.n_collected_episodes > 0
                and self.config.test_in_train
                and self.config.stop_fn
                and self.config.stop_fn(collect_stats.returns_stat.mean)  # type: ignore
            ):
                assert self.config.test_collector is not None
                assert self.config.episode_per_test is not None and self.config.episode_per_test > 0
                test_result = test_episode(
                    self.config.test_collector,
                    self.config.test_fn,
                    self._epoch,
                    self.config.episode_per_test,
                    self._logger,
                    self._env_step,
                )
                assert test_result.returns_stat is not None  # for mypy
                if self.config.stop_fn(test_result.returns_stat.mean):
                    should_stop_training = True
                    self._best_reward = test_result.returns_stat.mean
                    self._best_reward_std = test_result.returns_stat.std
                    self._best_score = self._compute_score_fn(test_result)

        return should_stop_training

    @abstractmethod
    def _update_step(
        self,
        collect_stats: CollectStatsBase,
    ) -> TrainingStats:
        """Performs a gradient update step, calling the algorithm's update method accordingly.

        :param collect_stats: provides info about the preceding data collection step.
        """

    def _create_epoch_pbar_data_dict(
        self, training_step_result: _TrainingStepResult
    ) -> dict[str, str]:
        collect_stats = training_step_result.get_collect_stats()
        result = {
            "env_step": str(self._env_step),
            "env_episode": str(self._env_episode),
            "n_ep": str(collect_stats.n_collected_episodes),
            "n_st": str(collect_stats.n_collected_steps),
        }
        # return and episode length info is only available if at least one episode was completed
        if collect_stats.n_collected_episodes > 0:
            result.update(
                {
                    "rew": f"{collect_stats.returns_stat.mean:.2f}",
                    "len": str(int(collect_stats.lens_stat.mean)),
                }
            )
        return result


class OffPolicyTrainer(OnlineTrainer[OffPolicyTrainingConfig]):
    """An off-policy trainer, which samples mini-batches from the buffer of collected data and passes them to
    algorithm's `update` function.

    The algorithm's `update` method is expected to not perform additional mini-batching but just update
    model parameters from the received mini-batch.
    """

    def _update_step(
        self,
        # TODO: this is the only implementation where collect_stats is actually needed. Maybe change interface?
        collect_stats: CollectStatsBase,
    ) -> TrainingStats:
        """Perform `update_per_step * n_collected_steps` gradient steps by sampling mini-batches from the buffer.

        :param collect_stats: the :class:`~TrainingStats` instance returned by the last gradient step. Some values
            in it will be replaced by their moving averages.
        """
        assert self.config.train_collector is not None
        n_collected_steps = collect_stats.n_collected_steps
        n_gradient_steps = round(self.config.update_per_step * n_collected_steps)
        if n_gradient_steps == 0:
            raise ValueError(
                f"n_gradient_steps is 0, n_collected_steps={n_collected_steps}, "
                f"update_per_step={self.config.update_per_step}",
            )

        update_stat = None
        for _ in self._pbar(
            range(n_gradient_steps),
            desc="Offpolicy gradient update",
            position=0,
            leave=False,
        ):
            update_stat = self._sample_and_update(self.config.train_collector.buffer)
            self._policy_update_time += update_stat.train_time

        # TODO: only the last update_stat is returned, should be improved
        return update_stat

    def _sample_and_update(self, buffer: ReplayBuffer) -> TrainingStats:
        """Sample a mini-batch, perform one gradient step, and update the _gradient_step counter."""
        self._gradient_step += 1
        # Note: since sample_size=batch_size, this will perform
        # exactly one gradient step. This is why we don't need to calculate the
        # number of gradient steps, like in the on-policy case.
        update_stat = self.algorithm.update(sample_size=self.config.batch_size, buffer=buffer)
        self._update_moving_avg_stats_and_log_update_data(update_stat)
        return update_stat


class OnPolicyTrainer(OnlineTrainer[OnPolicyTrainingConfig]):
    """An on-policy trainer, which passes the entire buffer to the algorithm's `update` methods and
    resets the buffer thereafter.

    Note that it is expected that the update method of the algorithm will perform
    batching when using this trainer.
    """

    def _update_step(
        self,
        result: CollectStatsBase | None = None,
    ) -> TrainingStats:
        """Perform one on-policy update by passing the entire buffer to the policy's update method."""
        assert self.config.train_collector is not None
        # TODO: add logging like in off-policy. Iteration over minibatches currently happens in the algorithms themselves.
        log.info(
            f"Performing on-policy update on buffer of length {len(self.config.train_collector.buffer)}",
        )
        training_stat = self.algorithm.update(
            sample_size=0,
            buffer=self.config.train_collector.buffer,
            # Note: sample_size is None, so the whole buffer is used for the update.
            # The kwargs are in the end passed to the .learn method, which uses
            # batch_size to iterate through the buffer in mini-batches
            # Off-policy algos typically don't use the batch_size kwarg at all
            batch_size=self.config.batch_size,
            repeat=self.config.repeat_per_collect,
        )

        # just for logging, no functional role
        self._policy_update_time += training_stat.train_time
        # TODO: remove the gradient step counting in trainers? Doesn't seem like
        #   it's important and it adds complexity
        self._gradient_step += 1
        if self.config.batch_size is None:
            self._gradient_step += 1
        elif self.config.batch_size > 0:
            self._gradient_step += int(
                (len(self.config.train_collector.buffer) - 0.1) // self.config.batch_size,
            )

        # Note 1: this is the main difference to the off-policy trainer!
        # The second difference is that batches of data are sampled without replacement
        # during training, whereas in off-policy or offline training, the batches are
        # sampled with replacement (and potentially custom prioritization).
        # Note 2: in the policy-update we modify the buffer, which is not very clean.
        # currently the modification will erase previous samples but keep things like
        # _ep_rew and _ep_len. This means that such quantities can no longer be computed
        # from samples still contained in the buffer, which is also not clean
        # TODO: improve this situation
        self.config.train_collector.reset_buffer(keep_statistics=True)

        # The step is the number of mini-batches used for the update, so essentially
        self._update_moving_avg_stats_and_log_update_data(training_stat)

        return training_stat
