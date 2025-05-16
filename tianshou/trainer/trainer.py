"""
This module contains Tianshou's trainer classes, which orchestrate the training and call upon an RL algorithm's
specific network updating logic to perform the actual gradient updates.

Training is structured as follows (hierarchical glossary):
- **epoch**: The outermost iteration level of the training loop. Each epoch consists of a number of training steps
  and one test step (see :attr:`TrainerParams.max_epoch` for a detailed explanation):
    - **training step**: A training step performs the steps necessary in order to apply a single update of the neural
      network components as defined by the underlying RL algorithm (:class:`Algorithm`). This involves the following sub-steps:
        - for online learning algorithms:
            - **collection step**: collecting environment steps/transitions to be used for training.
            - (potentially) a test step (see below) if the early stopping criterion is satisfied based on
              the data collected (see :attr:`OnlineTrainerParams.test_in_train`).
        - **update step**: applying the actual gradient updates using the RL algorithm.
          The update is based on either ...
            - data from only the preceding collection step (on-policy learning),
            - data from the collection step and previously collected data (off-policy learning), or
            - data from the user-provided replay buffer (offline learning).
      For offline learning algorithms, a training step is thus equivalent to an update step.
    - **test step**: Collects test episodes from dedicated test environments which are used to evaluate the performance
      of the policy. Optionally, the performance result can be used to determine whether training shall stop early
      (see :attr:`TrainerParams.stop_fn`).
"""

import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from typing import Generic, TypeVar

import numpy as np
import torch
import tqdm
from sensai.util.helper import count_none
from sensai.util.string import ToStringMixin

from tianshou.algorithm.algorithm_base import (
    Algorithm,
    OfflineAlgorithm,
    OffPolicyAlgorithm,
    OnPolicyAlgorithm,
    TrainingStats,
)
from tianshou.data import (
    AsyncCollector,
    CollectStats,
    EpochStats,
    InfoStats,
    ReplayBuffer,
    SequenceSummaryStats,
    TimingStats,
)
from tianshou.data.buffer.buffer_base import MalformedBufferError
from tianshou.data.collector import BaseCollector, CollectStatsBase
from tianshou.utils import (
    BaseLogger,
    LazyLogger,
    MovAvg,
)
from tianshou.utils.determinism import TraceLogger, torch_param_hash
from tianshou.utils.logging import set_numerical_fields_to_precision
from tianshou.utils.torch_utils import policy_within_training_step

log = logging.getLogger(__name__)


@dataclass(kw_only=True)
class TrainerParams(ToStringMixin):
    max_epochs: int = 100
    """
    the (maximum) number of epochs to run training for. An **epoch** is the outermost iteration level and each
    epoch consists of a number of training steps and one test step, where each training step

      * [for the online case] collects environment steps/transitions (**collection step**),
        adding them to the (replay) buffer (see :attr:`collection_step_num_env_steps` and :attr:`collection_step_num_episodes`)
      * performs an **update step** via the RL algorithm being used, which can involve
        one or more actual gradient updates, depending on the algorithm

    and the test step collects :attr:`num_episodes_per_test` test episodes in order to evaluate
    agent performance.

    Training may be stopped early if the stop criterion is met (see :attr:`stop_fn`).

    For online training, the number of training steps in each epoch is indirectly determined by
    :attr:`epoch_num_steps`: As many training steps will be performed as are required in
    order to reach :attr:`epoch_num_steps` total steps in the training environments.
    Specifically, if the number of transitions collected per step is `c` (see
    :attr:`collection_step_num_env_steps`) and :attr:`epoch_num_steps` is set to `s`, then the number
    of training steps per epoch is `ceil(s / c)`.
    Therefore, if `max_epochs = e`, the total number of environment steps taken during training
    can be computed as `e * ceil(s / c) * c`.

    For offline training, the number of training steps per epoch is equal to :attr:`epoch_num_steps`.
    """

    epoch_num_steps: int = 30000
    """
    For an online algorithm, this is the total number of environment steps to be collected per epoch, and,
    for an offline algorithm, it is the total number of training steps to take per epoch.
    See :attr:`max_epochs` for an explanation of epoch semantics.
    """

    test_collector: BaseCollector | None = None
    """
    the collector to use for test episode collection (test steps); if None, perform no test steps.
    """

    test_step_num_episodes: int = 1
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
    is achieved in a test step. It should have the signature ``f(algorithm: Algorithm) -> None``.
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

    multi_agent_return_reduction: Callable[[np.ndarray], np.ndarray] | None = None
    """
    a function with signature
    ``f(returns: np.ndarray with shape (num_episode, agent_num)) -> np.ndarray with shape (num_episode,)``,
    which is used in multi-agent RL. We need to return a single scalar for each episode's return
    to monitor training in the multi-agent RL setting. This function specifies what is the desired metric,
    e.g., the return achieved by agent 1 or the average return over all agents.
    """

    logger: BaseLogger | None = None
    """
    the logger with which to log statistics during training/testing/updating. To not log anything, use None.

    Relevant step types for logger update intervals:
      * `update_interval`: update step
      * `train_interval`: env step
      * `test_interval`: env step
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

    def __post_init__(self) -> None:
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
        else:
            if self.test_step_num_episodes < 1:
                raise ValueError(
                    "test_step_num_episodes must be positive if test steps are enabled "
                    "(test_collector not None)"
                )


@dataclass(kw_only=True)
class OnlineTrainerParams(TrainerParams):
    train_collector: BaseCollector
    """
    the collector with which to gather new data for training in each training step
    """

    collection_step_num_env_steps: int | None = 2048
    """
    the number of environment steps/transitions to collect in each collection step before the
    network update within each training step.

    This is mutually exclusive with :attr:`collection_step_num_episodes`, and one of the two must be set.

    Note that the exact number can be reached only if this is a multiple of the number of
    training environments being used, as each training environment will produce the same
    (non-zero) number of transitions.
    Specifically, if this is set to `n` and `m` training environments are used, then the total
    number of transitions collected per collection step is `ceil(n / m) * m =: c`.

    See :attr:`max_epochs` for information on the total number of environment steps being
    collected during training.
    """

    collection_step_num_episodes: int | None = None
    """
    the number of episodes to collect in each collection step before the network update within
    each training step. If this is set, the number of environment steps collected in each
    collection step is the sum of the lengths of the episodes collected.

    This is mutually exclusive with :attr:`collection_step_num_env_steps`, and one of the two must be set.
    """

    test_in_train: bool = False
    """
    Whether to apply a test step within a training step depending on the early stopping criterion
    (given by :attr:`stop_fn`) being satisfied based on the data collected within the training step.
    Specifically, after each collect step, we check whether the early stopping criterion (:attr:`stop_fn`)
    would be satisfied by data we collected (provided that at least one episode was indeed completed, such
    that we can evaluate returns, etc.). If the criterion is satisfied, we perform a full test step
    (collecting :attr:`test_step_num_episodes` episodes in order to evaluate performance), and if the early
    stopping criterion is also satisfied based on the test data, we stop training early.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if count_none(self.collection_step_num_env_steps, self.collection_step_num_episodes) != 1:
            raise ValueError(
                "Exactly one of {collection_step_num_env_steps, collection_step_num_episodes} must be set"
            )
        if self.test_in_train and (self.test_collector is None or self.stop_fn is None):
            raise ValueError("test_in_train requires test_collector and stop_fn to be set")


@dataclass(kw_only=True)
class OnPolicyTrainerParams(OnlineTrainerParams):
    batch_size: int | None = 64
    """
    Use mini-batches of this size for gradient updates (causing the gradient to be less accurate,
    a form of regularization).
    Set ``batch_size=None`` for the full buffer that was collected within the training step to be
    used for the gradient update (no mini-batching).
    """

    update_step_num_repetitions: int = 1
    """
    controls, within one update step of an on-policy algorithm, the number of times
    the full collected data is applied for gradient updates, i.e. if the parameter is
    5, then the collected data shall be used five times to update the policy within the same
    update step.
    """


@dataclass(kw_only=True)
class OffPolicyTrainerParams(OnlineTrainerParams):
    batch_size: int = 64
    """
    the the number of environment steps/transitions to sample from the buffer for a gradient update.
    """

    update_step_num_gradient_steps_per_sample: float = 1.0
    """
    the number of gradient steps to perform per sample collected (see :attr:`collection_step_num_env_steps`).
    Specifically, if this is set to `u` and the number of samples collected in the preceding
    collection step is `n`, then `round(u * n)` gradient steps will be performed.
    """


@dataclass(kw_only=True)
class OfflineTrainerParams(TrainerParams):
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


TTrainerParams = TypeVar("TTrainerParams", bound=TrainerParams)
TOnlineTrainerParams = TypeVar("TOnlineTrainerParams", bound=OnlineTrainerParams)
TAlgorithm = TypeVar("TAlgorithm", bound=Algorithm)


class Trainer(Generic[TAlgorithm, TTrainerParams], ABC):
    """
    Base class for trainers in Tianshou, which orchestrate the training process and call upon an RL algorithm's
    specific network updating logic to perform the actual gradient updates.

    The base class already implements the fundamental epoch logic and fully implements the test step
    logic, which is common to all trainers. The training step logic is left to be implemented by subclasses.
    """

    def __init__(
        self,
        algorithm: TAlgorithm,
        params: TTrainerParams,
    ):
        self.algorithm = algorithm
        self.params = params

        self._logger = params.logger or LazyLogger()

        self._start_time = time.time()
        self._stat: defaultdict[str, MovAvg] = defaultdict(MovAvg)
        self._start_epoch = 0

        self._epoch = self._start_epoch

        # initialize stats on the best model found during a test step
        # NOTE: The values don't matter, as in the first test step (which is taken in reset()
        #   at the beginning of the training process), these will all be updated
        self._best_score = 0.0
        self._best_reward = 0.0
        self._best_reward_std = 0.0
        self._best_epoch = self._start_epoch

        self._current_update_step = 0
        """
        the current (1-based) update step/training step number (to be incremented before the actual step is taken)
        """

        self._env_step = 0
        """
        the step counter which is used to track progress of the training process.
        For online learning (i.e. on-policy and off-policy learning), this is the total number of
        environment steps collected, and for offline training, it is the total number of environment
        steps that have been sampled from the replay buffer to perform gradient updates.
        """

        self._policy_update_time = 0.0

        self._compute_score_fn: Callable[[CollectStats], float] = (
            params.compute_score_fn or self._compute_score_fn_default
        )

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
            disable=not self.params.show_progress,
        )

    def _reset_collectors(self, reset_buffer: bool = False) -> None:
        if self.params.test_collector is not None:
            self.params.test_collector.reset(reset_buffer=reset_buffer)

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
        TraceLogger.log(log, lambda: "Trainer reset")
        self._env_step = 0
        self._current_update_step = 0

        if self.params.resume_from_log:
            (
                self._start_epoch,
                self._env_step,
                self._current_update_step,
            ) = self._logger.restore_data()

        self._epoch = self._start_epoch

        self._start_time = time.time()

        if reset_collectors:
            self._reset_collectors(reset_buffer=reset_collector_buffers)

        # make an initial test step to determine the initial best model
        if self.params.test_collector is not None:
            assert self.params.test_step_num_episodes is not None
            assert not isinstance(self.params.test_collector, AsyncCollector)  # Issue 700
            self._test_step(force_update_best=True, log_msg_prefix="Initial test step")

        self._stop_fn_flag = False

        self._log_params(self.algorithm)

    def _log_params(self, module: torch.nn.Module) -> None:
        """Logs the parameters of the module to the trace logger by subcomponent (if the trace logger is enabled)."""
        if not TraceLogger.is_enabled:
            return

        def module_has_params(m: torch.nn.Module) -> bool:
            return any(p.requires_grad for p in m.parameters())

        relevant_modules = {}

        def gather_modules(m: torch.nn.Module) -> None:
            for name, submodule in m.named_children():
                if name == "policy":
                    gather_modules(submodule)
                else:
                    if module_has_params(submodule):
                        relevant_modules[name] = submodule

        gather_modules(module)

        for name, module in sorted(relevant_modules.items()):
            TraceLogger.log(
                log,
                lambda: f"Params[{name}]: {torch_param_hash(module)}",
            )

    class _TrainingStepResult(ABC):
        @abstractmethod
        def get_steps_in_epoch_advancement(self) -> int:
            """
            :return: the number of steps that were done within the epoch, where the concrete semantics
                of what a step is depend on the type of algorithm. See docstring of `TrainerParams.epoch_num_steps`.
            """

        @abstractmethod
        def get_collect_stats(self) -> CollectStats | None:
            pass

        @abstractmethod
        def get_training_stats(self) -> TrainingStats | None:
            pass

        @abstractmethod
        def is_training_done(self) -> bool:
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

    def _create_info_stats(
        self,
    ) -> InfoStats:
        test_collector = self.params.test_collector
        if isinstance(self.params, OnlineTrainerParams):
            train_collector = self.params.train_collector
        else:
            train_collector = None

        duration = max(0.0, time.time() - self._start_time)
        test_time = 0.0
        update_speed = 0.0
        train_time_collect = 0.0
        if test_collector is not None:
            test_time = test_collector.collect_time

        if train_collector is not None:
            train_time_collect = train_collector.collect_time
            update_speed = train_collector.collect_step / (duration - test_time)

        timing_stat = TimingStats(
            total_time=duration,
            train_time=duration - test_time,
            train_time_collect=train_time_collect,
            train_time_update=self._policy_update_time,
            test_time=test_time,
            update_speed=update_speed,
        )

        return InfoStats(
            update_step=self._current_update_step,
            best_score=self._best_score,
            best_reward=self._best_reward,
            best_reward_std=self._best_reward_std,
            train_step=train_collector.collect_step if train_collector is not None else 0,
            train_episode=train_collector.collect_episode if train_collector is not None else 0,
            test_step=test_collector.collect_step if test_collector is not None else 0,
            test_episode=test_collector.collect_episode if test_collector is not None else 0,
            timing=timing_stat,
        )

    def execute_epoch(self) -> EpochStats:
        self._epoch += 1
        TraceLogger.log(log, lambda: f"Epoch #{self._epoch} start")

        # perform the required number of steps for the epoch (`epoch_num_steps`)
        steps_done_in_this_epoch = 0
        train_collect_stats, training_stats = None, None
        with self._pbar(
            total=self.params.epoch_num_steps, desc=f"Epoch #{self._epoch}", position=1
        ) as t:
            while steps_done_in_this_epoch < self.params.epoch_num_steps and not self._stop_fn_flag:
                # perform a training step and update progress
                TraceLogger.log(log, lambda: "Training step")
                self._current_update_step += 1
                training_step_result = self._training_step()
                steps_done_in_this_epoch += training_step_result.get_steps_in_epoch_advancement()
                t.update(training_step_result.get_steps_in_epoch_advancement())
                self._stop_fn_flag = training_step_result.is_training_done()
                self._env_step += training_step_result.get_env_step_advancement()
                training_stats = training_step_result.get_training_stats()
                TraceLogger.log(
                    log,
                    lambda: f"Training step complete: stats={training_stats.get_loss_stats_dict() if training_stats is not None else None}",
                )
                self._log_params(self.algorithm)

                collect_stats = training_step_result.get_collect_stats()
                if collect_stats is not None:
                    self._logger.log_train_data(asdict(collect_stats), self._env_step)

                pbar_data_dict = self._create_epoch_pbar_data_dict(training_step_result)
                pbar_data_dict = set_numerical_fields_to_precision(pbar_data_dict)
                pbar_data_dict["update_step"] = str(self._current_update_step)
                t.set_postfix(**pbar_data_dict)

        test_collect_stats = None
        if not self._stop_fn_flag:
            self._logger.save_data(
                self._epoch,
                self._env_step,
                self._current_update_step,
                self.params.save_checkpoint_fn,
            )

            # test step
            if self.params.test_collector is not None:
                test_collect_stats, self._stop_fn_flag = self._test_step()

        info_stats = self._create_info_stats()

        self._logger.log_info_data(asdict(info_stats), self._epoch)

        return EpochStats(
            epoch=self._epoch,
            train_collect_stat=train_collect_stats,
            test_collect_stat=test_collect_stats,
            training_stat=training_stats,
            info_stat=info_stats,
        )

    def _should_stop_training_early(
        self, *, score: float | None = None, collect_stats: CollectStats | None = None
    ) -> bool:
        """
        Determine whether, given the early stopping criterion stop_fn, training shall be stopped early
        based on the score achieved or the collection stats (from which the score could be computed).
        """
        # If no stop criterion is defined, we can never stop training early
        if self.params.stop_fn is None:
            return False

        if score is None:
            if collect_stats is None:
                raise ValueError("Must provide collect_stats if score is not given")

            # If no episodes were collected, we have no episode returns and thus cannot compute a score
            if collect_stats.n_collected_episodes == 0:
                return False

            score = self._compute_score_fn(collect_stats)

        return self.params.stop_fn(score)

    def _collect_test_episodes(
        self,
    ) -> CollectStats:
        assert self.params.test_collector is not None
        collector = self.params.test_collector
        collector.reset(reset_stats=False)
        if self.params.test_fn:
            self.params.test_fn(self._epoch, self._env_step)
        result = collector.collect(n_episode=self.params.test_step_num_episodes)
        if self.params.multi_agent_return_reduction:
            rew = self.params.multi_agent_return_reduction(result.returns)
            result.returns = rew
            result.returns_stat = SequenceSummaryStats.from_sequence(rew)
        if self._logger and self._env_step is not None:
            assert result.n_collected_episodes > 0
            self._logger.log_test_data(asdict(result), self._env_step)
        return result

    def _test_step(
        self, force_update_best: bool = False, log_msg_prefix: str | None = None
    ) -> tuple[CollectStats, bool]:
        """Performs one test step.

        :param log_msg_prefix: a prefix to prepend to the log message, which is to establish the context within
            which the test step is being carried out
        :param force_update_best: whether to force updating of the best model stats (best score, reward, etc.)
            and call the `save_best_fn` callback
        """
        assert self.params.test_step_num_episodes is not None
        assert self.params.test_collector is not None

        # collect test episodes
        test_stat = self._collect_test_episodes()
        assert test_stat.returns_stat is not None  # for mypy

        # check whether we have a new best score and, if so, update stats and save the model
        # (or if forced)
        rew, rew_std = test_stat.returns_stat.mean, test_stat.returns_stat.std
        score = self._compute_score_fn(test_stat)
        if score > self._best_score or force_update_best:
            self._best_score = score
            self._best_epoch = self._epoch
            self._best_reward = float(rew)
            self._best_reward_std = rew_std
            if self.params.save_best_fn:
                self.params.save_best_fn(self.algorithm)

        # log results
        cur_info, best_info = "", ""
        if score != rew:
            cur_info, best_info = f", score: {score: .6f}", f", best_score: {self._best_score:.6f}"
        if log_msg_prefix is None:
            log_msg_prefix = f"Epoch #{self._epoch}"
        log_msg = (
            f"{log_msg_prefix}: test_reward: {rew:.6f} ± {rew_std:.6f},{cur_info}"
            f" best_reward: {self._best_reward:.6f} ± "
            f"{self._best_reward_std:.6f}{best_info} in #{self._best_epoch}"
        )
        log.info(log_msg)
        if self.params.verbose:
            print(log_msg, flush=True)

        # determine whether training shall be stopped early
        stop_fn_flag = self._should_stop_training_early(score=self._best_score)

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
        self._logger.log_update_data(asdict(update_stat), self._current_update_step)

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

        while self._epoch < self.params.max_epochs and not self._stop_fn_flag:
            self.execute_epoch()

        return self._create_info_stats()


class OfflineTrainer(Trainer[OfflineAlgorithm, OfflineTrainerParams]):
    """An offline trainer, which samples mini-batches from a given buffer and passes them to
    the algorithm's update function.
    """

    def __init__(
        self,
        algorithm: OfflineAlgorithm,
        params: OfflineTrainerParams,
    ):
        super().__init__(algorithm, params)
        self._buffer = algorithm.process_buffer(self.params.buffer)

    class _TrainingStepResult(Trainer._TrainingStepResult):
        def __init__(self, training_stats: TrainingStats, env_step_advancement: int):
            self._training_stats = training_stats
            self._env_step_advancement = env_step_advancement

        def get_steps_in_epoch_advancement(self) -> int:
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
            # Note: since sample_size=batch_size, this will perform
            # exactly one gradient step. This is why we don't need to calculate the
            # number of gradient steps, like in the on-policy case.
            training_stats = self.algorithm.update(
                sample_size=self.params.batch_size, buffer=self._buffer
            )
            self._update_moving_avg_stats_and_log_update_data(training_stats)
            self._policy_update_time += training_stats.train_time
            return self._TrainingStepResult(
                training_stats=training_stats, env_step_advancement=self.params.batch_size
            )

    def _create_epoch_pbar_data_dict(
        self, training_step_result: Trainer._TrainingStepResult
    ) -> dict[str, str]:
        return {}


class OnlineTrainer(
    Trainer[TAlgorithm, TOnlineTrainerParams], Generic[TAlgorithm, TOnlineTrainerParams], ABC
):
    """
    An online trainer, which collects data from the environment in each training step and
    uses the collected data to perform an update step, the nature of which is to be defined
    in subclasses.
    """

    def __init__(
        self,
        algorithm: TAlgorithm,
        params: OnlineTrainerParams,
    ):
        super().__init__(algorithm, params)
        self._env_episode = 0
        """
        the total number of episodes collected in the environment
        """

    def _reset_collectors(self, reset_buffer: bool = False) -> None:
        super()._reset_collectors(reset_buffer=reset_buffer)
        self.params.train_collector.reset(reset_buffer=reset_buffer)

    def reset(self, reset_collectors: bool = True, reset_collector_buffers: bool = False) -> None:
        super().reset(
            reset_collectors=reset_collectors, reset_collector_buffers=reset_collector_buffers
        )

        if (
            self.params.test_in_train
            and self.params.train_collector.policy is not self.algorithm.policy
        ):
            log.warning(
                "The training data collector's policy is not the same as the one being trained, "
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

        def get_steps_in_epoch_advancement(self) -> int:
            return self.get_env_step_advancement()

        def get_collect_stats(self) -> CollectStats:
            return self._collect_stats

        def get_training_stats(self) -> TrainingStats | None:
            return self._training_stats

        def is_training_done(self) -> bool:
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
            should_stop_training = False
            if self.params.test_in_train:
                should_stop_training = self._test_in_train(collect_stats)

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
        assert self.params.test_step_num_episodes is not None
        assert self.params.train_collector is not None

        if self.params.train_fn:
            self.params.train_fn(self._epoch, self._env_step)

        collect_stats = self.params.train_collector.collect(
            n_step=self.params.collection_step_num_env_steps,
            n_episode=self.params.collection_step_num_episodes,
        )
        TraceLogger.log(
            log,
            lambda: f"Collected {collect_stats.n_collected_steps} steps, {collect_stats.n_collected_episodes} episodes",
        )

        if self.params.train_collector.buffer.hasnull():
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
            if self.params.multi_agent_return_reduction:
                rew = self.params.multi_agent_return_reduction(collect_stats.returns)
                collect_stats.returns = rew
                collect_stats.returns_stat = SequenceSummaryStats.from_sequence(rew)

        # update collection stats specific to this specialization
        self._env_episode += collect_stats.n_collected_episodes

        return collect_stats

    def _test_in_train(
        self,
        train_collect_stats: CollectStats,
    ) -> bool:
        """
        Performs a test step if the data collected in the current training step suggests that performance
        is good enough to stop training early. If the test step confirms that performance is indeed good
        enough, returns True, and False otherwise.

        Specifically, applies the early stopping criterion to the data collected in the current training step,
        and if the criterion is satisfied, performs a test step which returns the relevant result.

        :param train_collect_stats: the data collection stats from the preceding collection step
        :return: flag indicating whether to stop training early
        """
        should_stop_training = False

        # check whether the stop criterion is satisfied based on the data collected in the training step
        # (if any full episodes were indeed collected)
        if train_collect_stats.n_collected_episodes > 0 and self._should_stop_training_early(
            collect_stats=train_collect_stats
        ):
            # apply a test step, temporarily switching out of "is_training_step" semantics such that the policy can
            # be evaluated, in order to determine whether we should stop training
            with policy_within_training_step(self.algorithm.policy, enabled=False):
                _, should_stop_training = self._test_step(
                    log_msg_prefix=f"Test step triggered by train stats (env_step={self._env_step})"
                )

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
        self, training_step_result: Trainer._TrainingStepResult
    ) -> dict[str, str]:
        collect_stats = training_step_result.get_collect_stats()
        assert collect_stats is not None
        result = {
            "env_step": str(self._env_step),
            "env_episode": str(self._env_episode),
            "n_ep": str(collect_stats.n_collected_episodes),
            "n_st": str(collect_stats.n_collected_steps),
        }
        # return and episode length info is only available if at least one episode was completed
        if collect_stats.n_collected_episodes > 0:
            assert collect_stats.returns_stat is not None
            assert collect_stats.lens_stat is not None
            result.update(
                {
                    "rew": f"{collect_stats.returns_stat.mean:.2f}",
                    "len": str(int(collect_stats.lens_stat.mean)),
                }
            )
        return result


class OffPolicyTrainer(OnlineTrainer[OffPolicyAlgorithm, OffPolicyTrainerParams]):
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
        """Perform `update_step_num_gradient_steps_per_sample * n_collected_steps` gradient steps by sampling
        mini-batches from the buffer.

        :param collect_stats: the :class:`~TrainingStats` instance returned by the last gradient step. Some values
            in it will be replaced by their moving averages.
        """
        assert self.params.train_collector is not None
        n_collected_steps = collect_stats.n_collected_steps
        n_gradient_steps = round(
            self.params.update_step_num_gradient_steps_per_sample * n_collected_steps
        )
        if n_gradient_steps == 0:
            raise ValueError(
                f"n_gradient_steps is 0, n_collected_steps={n_collected_steps}, "
                f"update_step_num_gradient_steps_per_sample={self.params.update_step_num_gradient_steps_per_sample}",
            )

        update_stat = None
        for _ in self._pbar(
            range(n_gradient_steps),
            desc="Offpolicy gradient update",
            position=0,
            leave=False,
        ):
            update_stat = self._sample_and_update(self.params.train_collector.buffer)
            self._policy_update_time += update_stat.train_time

        # TODO: only the last update_stat is returned, should be improved
        assert update_stat is not None
        return update_stat

    def _sample_and_update(self, buffer: ReplayBuffer) -> TrainingStats:
        """Sample a mini-batch, perform one gradient step, and update the _gradient_step counter."""
        # Note: since sample_size=batch_size, this will perform
        # exactly one gradient step. This is why we don't need to calculate the
        # number of gradient steps, like in the on-policy case.
        update_stat = self.algorithm.update(sample_size=self.params.batch_size, buffer=buffer)
        self._update_moving_avg_stats_and_log_update_data(update_stat)
        return update_stat


class OnPolicyTrainer(OnlineTrainer[OnPolicyAlgorithm, OnPolicyTrainerParams]):
    """An on-policy trainer, which passes the entire buffer to the algorithm's `update` methods and
    resets the buffer thereafter.

    Note that it is expected that the update method of the algorithm will perform
    batching when using this trainer.
    """

    def _update_step(
        self,
        result: CollectStatsBase | None = None,
    ) -> TrainingStats:
        """Perform one on-policy update by passing the entire buffer to the algorithm's update method."""
        assert self.params.train_collector is not None
        # TODO: add logging like in off-policy. Iteration over minibatches currently happens in the algorithms themselves.
        log.info(
            f"Performing on-policy update on buffer of length {len(self.params.train_collector.buffer)}",
        )
        training_stat = self.algorithm.update(
            buffer=self.params.train_collector.buffer,
            batch_size=self.params.batch_size,
            repeat=self.params.update_step_num_repetitions,
        )

        # just for logging, no functional role
        self._policy_update_time += training_stat.train_time

        # Note 1: this is the main difference to the off-policy trainer!
        # The second difference is that batches of data are sampled without replacement
        # during training, whereas in off-policy or offline training, the batches are
        # sampled with replacement (and potentially custom prioritization).
        # Note 2: in the policy-update we modify the buffer, which is not very clean.
        # currently the modification will erase previous samples but keep things like
        # _ep_rew and _ep_len. This means that such quantities can no longer be computed
        # from samples still contained in the buffer, which is also not clean
        # TODO: improve this situation
        self.params.train_collector.reset_buffer(keep_statistics=True)

        # The step is the number of mini-batches used for the update, so essentially
        self._update_moving_avg_stats_and_log_update_data(training_stat)

        return training_stat
