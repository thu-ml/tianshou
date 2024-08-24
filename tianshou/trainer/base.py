import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import asdict
from functools import partial

import numpy as np
import tqdm

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
from tianshou.policy import BasePolicy
from tianshou.policy.base import TrainingStats
from tianshou.trainer.utils import gather_info, test_episode
from tianshou.utils import (
    BaseLogger,
    LazyLogger,
    MovAvg,
)
from tianshou.utils.logging import set_numerical_fields_to_precision
from tianshou.utils.torch_utils import policy_within_training_step

log = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """An iterator base class for trainers.

    Returns an iterator that yields a 3-tuple (epoch, stats, info) of train results
    on every epoch.

    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy` class.
    :param batch_size: the batch size of sample data, which is going to feed in
        the policy network. If None, will use the whole buffer in each gradient step.
    :param train_collector: the collector used for training.
    :param test_collector: the collector used for testing. If it's None,
        then no testing will be performed.
    :param buffer: the replay buffer used for off-policy algorithms or for pre-training.
        If a policy overrides the ``process_buffer`` method, the replay buffer will
        be pre-processed before training.
    :param max_epoch: the maximum number of epochs for training. The training
        process might be finished before reaching ``max_epoch`` if ``stop_fn``
        is set.
    :param step_per_epoch: the number of transitions collected per epoch.
    :param repeat_per_collect: the number of repeat time for policy learning,
        for example, set it to 2 means the policy needs to learn each given batch
        data twice. Only used in on-policy algorithms
    :param episode_per_test: the number of episodes for one policy evaluation.
    :param update_per_step: only used in off-policy algorithms.
        How many gradient steps to perform per step in the environment
        (i.e., per sample added to the buffer).
    :param step_per_collect: the number of transitions the collector would
        collect before the network update, i.e., trainer will collect
        "step_per_collect" transitions and do some policy network update repeatedly
        in each epoch.
    :param episode_per_collect: the number of episodes the collector would
        collect before the network update, i.e., trainer will collect
        "episode_per_collect" episodes and do some policy network update repeatedly
        in each epoch.
    :param train_fn: a hook called at the beginning of training in each
        epoch. It can be used to perform custom additional operations, with the
        signature ``f(num_epoch: int, step_idx: int) -> None``.
    :param test_fn: a hook called at the beginning of testing in each
        epoch. It can be used to perform custom additional operations, with the
        signature ``f(num_epoch: int, step_idx: int) -> None``.
    :param compute_score_fn: Calculate the test batch performance score to
        determine whether it is the best model, the mean reward will be used as score if not provided.
    :param save_best_fn: a hook called when the undiscounted average mean
        reward in evaluation phase gets better, with the signature
        ``f(policy: BasePolicy) -> None``.
    :param save_checkpoint_fn: a function to save training process and
        return the saved checkpoint path, with the signature ``f(epoch: int,
        env_step: int, gradient_step: int) -> str``; you can save whatever you want.
    :param resume_from_log: resume env_step/gradient_step and other metadata
        from existing tensorboard log.
    :param stop_fn: a function with signature ``f(mean_rewards: float) ->
        bool``, receives the average undiscounted returns of the testing result,
        returns a boolean which indicates whether reaching the goal.
    :param reward_metric: a function with signature
        ``f(rewards: np.ndarray with shape (num_episode, agent_num)) -> np.ndarray
        with shape (num_episode,)``, used in multi-agent RL. We need to return a
        single scalar for each episode's result to monitor training in the
        multi-agent RL setting. This function specifies what is the desired metric,
        e.g., the reward of agent 1 or the average reward over all agents.
    :param logger: A logger that logs statistics during
        training/testing/updating. To not log anything, keep the default logger.
    :param verbose: whether to print status information to stdout.
        If set to False, status information will still be logged (provided that
        logging is enabled via the `logging` module).
    :param show_progress: whether to display a progress bar when training.
    :param test_in_train: whether to test in the training phase.
    """

    __doc__: str

    @staticmethod
    def gen_doc(learning_type: str) -> str:
        """Document string for subclass trainer."""
        step_means = f'The "step" in {learning_type} trainer means '
        if learning_type != "offline":
            step_means += "an environment step (a.k.a. transition)."
        else:  # offline
            step_means += "a gradient step."

        trainer_name = learning_type.capitalize() + "Trainer"

        return f"""An iterator class for {learning_type} trainer procedure.

        Returns an iterator that yields a 3-tuple (epoch, stats, info) of
        train results on every epoch.

        {step_means}

        Example usage:

        ::

            trainer = {trainer_name}(...)
            for epoch, epoch_stat, info in trainer:
                print("Epoch:", epoch)
                print(epoch_stat)
                print(info)
                do_something_with_policy()
                query_something_about_policy()
                make_a_plot_with(epoch_stat)
                display(info)

        - epoch int: the epoch number
        - epoch_stat dict: a large collection of metrics of the current epoch
        - info dict: result returned from :func:`~tianshou.trainer.gather_info`

        You can even iterate on several trainers at the same time:

        ::

            trainer1 = {trainer_name}(...)
            trainer2 = {trainer_name}(...)
            for result1, result2, ... in zip(trainer1, trainer2, ...):
                compare_results(result1, result2, ...)
        """

    def __init__(
        self,
        policy: BasePolicy,
        max_epoch: int,
        batch_size: int | None,
        train_collector: BaseCollector | None = None,
        test_collector: BaseCollector | None = None,
        buffer: ReplayBuffer | None = None,
        step_per_epoch: int | None = None,
        repeat_per_collect: int | None = None,
        episode_per_test: int | None = None,
        update_per_step: float = 1.0,
        step_per_collect: int | None = None,
        episode_per_collect: int | None = None,
        train_fn: Callable[[int, int], None] | None = None,
        test_fn: Callable[[int, int | None], None] | None = None,
        stop_fn: Callable[[float], bool] | None = None,
        compute_score_fn: Callable[[CollectStats], float] | None = None,
        save_best_fn: Callable[[BasePolicy], None] | None = None,
        save_checkpoint_fn: Callable[[int, int, int], str] | None = None,
        resume_from_log: bool = False,
        reward_metric: Callable[[np.ndarray], np.ndarray] | None = None,
        logger: BaseLogger | None = None,
        verbose: bool = True,
        show_progress: bool = True,
        test_in_train: bool = True,
    ):
        logger = logger or LazyLogger()
        self.policy = policy

        if buffer is not None:
            buffer = policy.process_buffer(buffer)
        self.buffer = buffer

        self.train_collector = train_collector
        self.test_collector = test_collector

        self.logger = logger
        self.start_time = time.time()
        self.stat: defaultdict[str, MovAvg] = defaultdict(MovAvg)
        self.best_score = 0.0
        self.best_reward = 0.0
        self.best_reward_std = 0.0
        self.start_epoch = 0
        # This is only used for logging but creeps into the implementations
        # of the trainers. I believe it would be better to remove
        self._gradient_step = 0
        self.env_step = 0
        self.env_episode = 0
        self.policy_update_time = 0.0
        self.max_epoch = max_epoch
        assert (
            step_per_epoch is not None
        ), "The trainer requires step_per_epoch to be set, sorry for the wrong type hint"
        self.step_per_epoch: int = step_per_epoch

        # either on of these two
        self.step_per_collect = step_per_collect
        self.episode_per_collect = episode_per_collect

        self.update_per_step = update_per_step
        self.repeat_per_collect = repeat_per_collect

        self.episode_per_test = episode_per_test

        self.batch_size = batch_size

        self.train_fn = train_fn
        self.test_fn = test_fn
        self.stop_fn = stop_fn
        self.compute_score_fn: Callable[[CollectStats], float]
        if compute_score_fn is None:

            def compute_score_fn(stat: CollectStats) -> float:
                assert stat.returns_stat is not None  # for mypy
                return stat.returns_stat.mean

        self.compute_score_fn = compute_score_fn
        self.save_best_fn = save_best_fn
        self.save_checkpoint_fn = save_checkpoint_fn

        self.reward_metric = reward_metric
        self.verbose = verbose
        self.show_progress = show_progress
        self.test_in_train = test_in_train
        self.resume_from_log = resume_from_log

        self.is_run = False
        self.last_rew, self.last_len = 0.0, 0.0

        self.epoch = self.start_epoch
        self.best_epoch = self.start_epoch
        self.stop_fn_flag = False
        self.iter_num = 0

    @property
    def _pbar(self) -> type[tqdm.tqdm]:
        """Use as context manager or iterator, i.e., `with self._pbar(...) as t:` or `for _ in self._pbar(...):`."""
        return partial(
            tqdm.tqdm,
            dynamic_ncols=True,
            ascii=True,
            disable=not self.show_progress,
        )  # type: ignore[return-value]

    def _reset_collectors(self, reset_buffer: bool = False) -> None:
        if self.train_collector is not None:
            self.train_collector.reset(reset_buffer=reset_buffer)
        if self.test_collector is not None:
            self.test_collector.reset(reset_buffer=reset_buffer)

    def reset(self, reset_collectors: bool = True, reset_buffer: bool = False) -> None:
        """Initialize or reset the instance to yield a new iterator from zero."""
        self.is_run = False
        self.env_step = 0
        if self.resume_from_log:
            (
                self.start_epoch,
                self.env_step,
                self._gradient_step,
            ) = self.logger.restore_data()

        self.last_rew, self.last_len = 0.0, 0.0
        self.start_time = time.time()

        if reset_collectors:
            self._reset_collectors(reset_buffer=reset_buffer)

        if self.train_collector is not None and (
            self.train_collector.policy != self.policy or self.test_collector is None
        ):
            self.test_in_train = False

        if self.test_collector is not None:
            assert self.episode_per_test is not None
            assert not isinstance(self.test_collector, AsyncCollector)  # Issue 700
            test_result = test_episode(
                self.test_collector,
                self.test_fn,
                self.start_epoch,
                self.episode_per_test,
                self.logger,
                self.env_step,
                self.reward_metric,
            )
            assert test_result.returns_stat is not None  # for mypy
            self.best_epoch = self.start_epoch
            self.best_reward, self.best_reward_std = (
                test_result.returns_stat.mean,
                test_result.returns_stat.std,
            )
            self.best_score = self.compute_score_fn(test_result)
        if self.save_best_fn:
            self.save_best_fn(self.policy)

        self.epoch = self.start_epoch
        self.stop_fn_flag = False
        self.iter_num = 0

    def __iter__(self):  # type: ignore
        self.reset(reset_collectors=True, reset_buffer=False)
        return self

    def __next__(self) -> EpochStats:
        """Perform one epoch (both train and eval)."""
        self.epoch += 1
        self.iter_num += 1

        if self.iter_num > 1:
            # iterator exhaustion check
            if self.epoch > self.max_epoch:
                raise StopIteration

            # exit flag 1, when stop_fn succeeds in train_step or test_step
            if self.stop_fn_flag:
                raise StopIteration

        # perform n step_per_epoch
        steps_done_in_this_epoch = 0
        with self._pbar(total=self.step_per_epoch, desc=f"Epoch #{self.epoch}", position=1) as t:
            train_stat: CollectStatsBase
            while steps_done_in_this_epoch < self.step_per_epoch and not self.stop_fn_flag:
                train_stat, update_stat, self.stop_fn_flag = self.training_step()

                if isinstance(train_stat, CollectStats):
                    pbar_data_dict = {
                        "env_step": str(self.env_step),
                        "env_episode": str(self.env_episode),
                        "rew": f"{self.last_rew:.2f}",
                        "len": str(int(self.last_len)),
                        "n/ep": str(train_stat.n_collected_episodes),
                        "n/st": str(train_stat.n_collected_steps),
                    }

                    # t might be disabled, we track the steps manually
                    t.update(train_stat.n_collected_steps)
                    steps_done_in_this_epoch += train_stat.n_collected_steps

                    if self.stop_fn_flag:
                        t.set_postfix(**pbar_data_dict)
                else:
                    # TODO: there is no iteration happening here, it's the offline case
                    #   Code should be restructured!
                    pbar_data_dict = {}
                    assert self.buffer, "No train_collector or buffer specified"
                    train_stat = CollectStatsBase(
                        n_collected_steps=len(self.buffer),
                    )

                    # t might be disabled, we track the steps manually
                    t.update()
                    steps_done_in_this_epoch += 1

                pbar_data_dict = set_numerical_fields_to_precision(pbar_data_dict)
                pbar_data_dict["gradient_step"] = str(self._gradient_step)
                t.set_postfix(**pbar_data_dict)

                if self.stop_fn_flag:
                    break

            if steps_done_in_this_epoch <= self.step_per_epoch and not self.stop_fn_flag:
                # t might be disabled, we track the steps manually
                t.update()
                steps_done_in_this_epoch += 1

        # for offline RL
        if self.train_collector is None:
            assert self.buffer is not None
            batch_size = self.batch_size or len(self.buffer)
            self.env_step = self._gradient_step * batch_size

        test_stat = None
        if not self.stop_fn_flag:
            self.logger.save_data(
                self.epoch,
                self.env_step,
                self._gradient_step,
                self.save_checkpoint_fn,
            )
            # test
            if self.test_collector is not None:
                test_stat, self.stop_fn_flag = self.test_step()

        info_stat = gather_info(
            start_time=self.start_time,
            policy_update_time=self.policy_update_time,
            gradient_step=self._gradient_step,
            best_score=self.best_score,
            best_reward=self.best_reward,
            best_reward_std=self.best_reward_std,
            train_collector=self.train_collector,
            test_collector=self.test_collector,
        )

        self.logger.log_info_data(asdict(info_stat), self.epoch)

        # in case trainer is used with run(), epoch_stat will not be returned
        return EpochStats(
            epoch=self.epoch,
            train_collect_stat=train_stat,
            test_collect_stat=test_stat,
            training_stat=update_stat,
            info_stat=info_stat,
        )

    def test_step(self) -> tuple[CollectStats, bool]:
        """Perform one testing step."""
        assert self.episode_per_test is not None
        assert self.test_collector is not None
        stop_fn_flag = False
        test_stat = test_episode(
            self.test_collector,
            self.test_fn,
            self.epoch,
            self.episode_per_test,
            self.logger,
            self.env_step,
            self.reward_metric,
        )
        assert test_stat.returns_stat is not None  # for mypy
        rew, rew_std = test_stat.returns_stat.mean, test_stat.returns_stat.std
        score = self.compute_score_fn(test_stat)
        if self.best_epoch < 0 or self.best_score < score:
            self.best_score = score
            self.best_epoch = self.epoch
            self.best_reward = float(rew)
            self.best_reward_std = rew_std
            if self.save_best_fn:
                self.save_best_fn(self.policy)
        cur_info, best_info = "", ""
        if score != rew:
            cur_info, best_info = f", score: {score: .6f}", f", best_score: {self.best_score:.6f}"
        log_msg = (
            f"Epoch #{self.epoch}: test_reward: {rew:.6f} ± {rew_std:.6f},{cur_info}"
            f" best_reward: {self.best_reward:.6f} ± "
            f"{self.best_reward_std:.6f}{best_info} in #{self.best_epoch}"
        )
        log.info(log_msg)
        if self.verbose:
            print(log_msg, flush=True)

        if self.stop_fn and self.stop_fn(self.best_reward):
            stop_fn_flag = True

        return test_stat, stop_fn_flag

    def training_step(self) -> tuple[CollectStatsBase, TrainingStats | None, bool]:
        """Perform one training iteration.

        A training iteration includes collecting data (for online RL), determining whether to stop training,
        and performing a policy update if the training iteration should continue.

        :return: the iteration's collect stats, training stats, and a flag indicating whether to stop training.
            If training is to be stopped, no gradient steps will be performed and the training stats will be `None`.
        """
        with policy_within_training_step(self.policy):
            should_stop_training = False

            collect_stats: CollectStatsBase | CollectStats
            if self.train_collector is not None:
                collect_stats = self._collect_training_data()
                should_stop_training = self._update_best_reward_and_return_should_stop_training(
                    collect_stats,
                )
            else:
                assert self.buffer is not None, "Either train_collector or buffer must be provided."
                collect_stats = CollectStatsBase(
                    n_collected_episodes=len(self.buffer),
                )

            if not should_stop_training:
                training_stats = self.policy_update_fn(collect_stats)
            else:
                training_stats = None

            return collect_stats, training_stats, should_stop_training

    def _collect_training_data(self) -> CollectStats:
        """Performs training data collection.

        :return: the data collection stats
        """
        assert self.episode_per_test is not None
        assert self.train_collector is not None
        if self.train_fn:
            self.train_fn(self.epoch, self.env_step)
        collect_stats = self.train_collector.collect(
            n_step=self.step_per_collect,
            n_episode=self.episode_per_collect,
        )

        if self.train_collector.buffer.hasnull():
            from tianshou.data.collector import EpisodeRolloutHook
            from tianshou.env import DummyVectorEnv

            raise MalformedBufferError(
                f"Encountered NaNs in buffer after {self.env_step} steps."
                f"Such errors are usually caused by either a bug in the environment or by "
                f"problematic implementations {EpisodeRolloutHook.__class__.__name__}. "
                f"For debugging such issues it is recommended to run the training in a single process, "
                f"e.g., by using {DummyVectorEnv.__class__.__name__}.",
            )

        self.env_step += collect_stats.n_collected_steps
        self.env_episode += collect_stats.n_collected_episodes

        if collect_stats.n_collected_episodes > 0:
            assert collect_stats.returns_stat is not None  # for mypy
            assert collect_stats.lens_stat is not None  # for mypy
            self.last_rew = collect_stats.returns_stat.mean
            self.last_len = collect_stats.lens_stat.mean
            if self.reward_metric:  # TODO: move inside collector
                rew = self.reward_metric(collect_stats.returns)
                collect_stats.returns = rew
                collect_stats.returns_stat = SequenceSummaryStats.from_sequence(rew)

            self.logger.log_train_data(asdict(collect_stats), self.env_step)
        return collect_stats

    # TODO (maybe): separate out side effect, simplify name?
    def _update_best_reward_and_return_should_stop_training(
        self,
        collect_stats: CollectStats,
    ) -> bool:
        """If `test_in_train` and `stop_fn` are set, will compute the `stop_fn` on the mean return of the training data.
        Then, if the `stop_fn` is True there, will collect test data also compute the stop_fn of the mean return
        on it.
        Finally, if the latter is also True, will return True.

        **NOTE:** has a side effect of updating the best reward and corresponding std.


        :param collect_stats: the data collection stats
        :return: flag indicating whether to stop training
        """
        should_stop_training = False

        # Because we need to evaluate the policy, we need to temporarily leave the "is_training_step" semantics
        with policy_within_training_step(self.policy, enabled=False):
            if (
                collect_stats.n_collected_episodes > 0
                and self.test_in_train
                and self.stop_fn
                and self.stop_fn(collect_stats.returns_stat.mean)  # type: ignore
            ):
                assert self.test_collector is not None
                assert self.episode_per_test is not None and self.episode_per_test > 0
                test_result = test_episode(
                    self.test_collector,
                    self.test_fn,
                    self.epoch,
                    self.episode_per_test,
                    self.logger,
                    self.env_step,
                )
                assert test_result.returns_stat is not None  # for mypy
                if self.stop_fn(test_result.returns_stat.mean):
                    should_stop_training = True
                    self.best_reward = test_result.returns_stat.mean
                    self.best_reward_std = test_result.returns_stat.std
                    self.best_score = self.compute_score_fn(test_result)

        return should_stop_training

    # TODO: move moving average computation and logging into its own logger
    # TODO: maybe think about a command line logger instead of always printing data dict
    def _update_moving_avg_stats_and_log_update_data(self, update_stat: TrainingStats) -> None:
        """Log losses, update moving average stats, and also modify the smoothed_loss in update_stat."""
        cur_losses_dict = update_stat.get_loss_stats_dict()
        update_stat.smoothed_loss = self._update_moving_avg_stats_and_get_averaged_data(
            cur_losses_dict,
        )
        self.logger.log_update_data(asdict(update_stat), self._gradient_step)

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
            self.stat[key].add(loss_item)
            smoothed_data[key] = self.stat[key].get()
        return smoothed_data

    @abstractmethod
    def policy_update_fn(
        self,
        collect_stats: CollectStatsBase,
    ) -> TrainingStats:
        """Policy update function for different trainer implementation.

        :param collect_stats: provides info about the most recent collection. In the offline case, this will contain
            stats of the whole dataset
        """

    def run(self, reset_prior_to_run: bool = True, reset_buffer: bool = False) -> InfoStats:
        """Consume iterator.

        See itertools - recipes. Use functions that consume iterators at C speed
        (feed the entire iterator into a zero-length deque).

        :param reset_prior_to_run: whether to reset collectors prior to run
        :param reset_buffer: only has effect if `reset_prior_to_run` is True.
            Then it will also reset the buffer. This is usually not necessary, use
            with caution.
        """
        if reset_prior_to_run:
            self.reset(reset_buffer=reset_buffer)
        try:
            self.is_run = True
            deque(self, maxlen=0)  # feed the entire iterator into a zero-length deque
            info = gather_info(
                start_time=self.start_time,
                policy_update_time=self.policy_update_time,
                gradient_step=self._gradient_step,
                best_score=self.best_score,
                best_reward=self.best_reward,
                best_reward_std=self.best_reward_std,
                train_collector=self.train_collector,
                test_collector=self.test_collector,
            )
        finally:
            self.is_run = False

        return info

    def _sample_and_update(self, buffer: ReplayBuffer) -> TrainingStats:
        """Sample a mini-batch, perform one gradient step, and update the _gradient_step counter."""
        self._gradient_step += 1
        # Note: since sample_size=batch_size, this will perform
        # exactly one gradient step. This is why we don't need to calculate the
        # number of gradient steps, like in the on-policy case.
        update_stat = self.policy.update(sample_size=self.batch_size, buffer=buffer)
        self._update_moving_avg_stats_and_log_update_data(update_stat)
        return update_stat


class OfflineTrainer(BaseTrainer):
    """Offline trainer, samples mini-batches from buffer and passes them to update.

    Uses a buffer directly and usually does not have a collector.
    """

    # for mypy
    assert isinstance(BaseTrainer.__doc__, str)
    __doc__ += BaseTrainer.gen_doc("offline") + "\n".join(BaseTrainer.__doc__.split("\n")[1:])

    def policy_update_fn(
        self,
        collect_stats: CollectStatsBase | None = None,
    ) -> TrainingStats:
        """Perform one off-line policy update."""
        assert self.buffer
        update_stat = self._sample_and_update(self.buffer)
        # logging
        self.policy_update_time += update_stat.train_time
        return update_stat


class OffpolicyTrainer(BaseTrainer):
    """Offpolicy trainer, samples mini-batches from buffer and passes them to update.

    Note that with this trainer, it is expected that the policy's `learn` method
    does not perform additional mini-batching but just updates params from the received
    mini-batch.
    """

    # for mypy
    assert isinstance(BaseTrainer.__doc__, str)
    __doc__ += BaseTrainer.gen_doc("offpolicy") + "\n".join(BaseTrainer.__doc__.split("\n")[1:])

    def policy_update_fn(
        self,
        # TODO: this is the only implementation where collect_stats is actually needed. Maybe change interface?
        collect_stats: CollectStatsBase,
    ) -> TrainingStats:
        """Perform `update_per_step * n_collected_steps` gradient steps by sampling mini-batches from the buffer.

        :param collect_stats: the :class:`~TrainingStats` instance returned by the last gradient step. Some values
            in it will be replaced by their moving averages.
        """
        assert self.train_collector is not None
        n_collected_steps = collect_stats.n_collected_steps
        n_gradient_steps = round(self.update_per_step * n_collected_steps)
        if n_gradient_steps == 0:
            raise ValueError(
                f"n_gradient_steps is 0, n_collected_steps={n_collected_steps}, "
                f"update_per_step={self.update_per_step}",
            )

        for _ in self._pbar(
            range(n_gradient_steps),
            desc="Offpolicy gradient update",
            position=0,
            leave=False,
        ):
            update_stat = self._sample_and_update(self.train_collector.buffer)
            self.policy_update_time += update_stat.train_time
        # TODO: only the last update_stat is returned, should be improved
        return update_stat


class OnpolicyTrainer(BaseTrainer):
    """On-policy trainer, passes the entire buffer to .update and resets it after.

    Note that it is expected that the learn method of a policy will perform
    batching when using this trainer.
    """

    # for mypy
    assert isinstance(BaseTrainer.__doc__, str)
    __doc__ = BaseTrainer.gen_doc("onpolicy") + "\n".join(BaseTrainer.__doc__.split("\n")[1:])

    def policy_update_fn(
        self,
        result: CollectStatsBase | None = None,
    ) -> TrainingStats:
        """Perform one on-policy update by passing the entire buffer to the policy's update method."""
        assert self.train_collector is not None
        # TODO: add logging like in off-policy. Iteration over minibatches currently happens in the learn implementation of
        #   on-policy algos like PG or PPO
        log.info(
            f"Performing on-policy update on buffer of length {len(self.train_collector.buffer)}",
        )
        training_stat = self.policy.update(
            sample_size=0,
            buffer=self.train_collector.buffer,
            # Note: sample_size is None, so the whole buffer is used for the update.
            # The kwargs are in the end passed to the .learn method, which uses
            # batch_size to iterate through the buffer in mini-batches
            # Off-policy algos typically don't use the batch_size kwarg at all
            batch_size=self.batch_size,
            repeat=self.repeat_per_collect,
        )

        # just for logging, no functional role
        self.policy_update_time += training_stat.train_time
        # TODO: remove the gradient step counting in trainers? Doesn't seem like
        #   it's important and it adds complexity
        self._gradient_step += 1
        if self.batch_size is None:
            self._gradient_step += 1
        elif self.batch_size > 0:
            self._gradient_step += int((len(self.train_collector.buffer) - 0.1) // self.batch_size)

        # Note 1: this is the main difference to the off-policy trainer!
        # The second difference is that batches of data are sampled without replacement
        # during training, whereas in off-policy or offline training, the batches are
        # sampled with replacement (and potentially custom prioritization).
        # Note 2: in the policy-update we modify the buffer, which is not very clean.
        # currently the modification will erase previous samples but keep things like
        # _ep_rew and _ep_len. This means that such quantities can no longer be computed
        # from samples still contained in the buffer, which is also not clean
        # TODO: improve this situation
        self.train_collector.reset_buffer(keep_statistics=True)

        # The step is the number of mini-batches used for the update, so essentially
        self._update_moving_avg_stats_and_log_update_data(training_stat)

        return training_stat
