import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Any, Callable, DefaultDict, Dict, Optional, Tuple, Union

import numpy as np
import tqdm

from tianshou.data import AsyncCollector, Collector, ReplayBuffer
from tianshou.policy import BasePolicy
from tianshou.trainer.utils import gather_info, test_episode
from tianshou.utils import (
    BaseLogger,
    DummyTqdm,
    LazyLogger,
    MovAvg,
    deprecation,
    tqdm_config,
)


class BaseTrainer(ABC):
    """An iterator base class for trainers procedure.

    Returns an iterator that yields a 3-tuple (epoch, stats, info) of train results
    on every epoch.

    :param learning_type str: type of learning iterator, available choices are
        "offpolicy", "onpolicy" and "offline".
    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy` class.
    :param Collector train_collector: the collector used for training.
    :param Collector test_collector: the collector used for testing. If it's None,
        then no testing will be performed.
    :param int max_epoch: the maximum number of epochs for training. The training
        process might be finished before reaching ``max_epoch`` if ``stop_fn``
        is set.
    :param int step_per_epoch: the number of transitions collected per epoch.
    :param int repeat_per_collect: the number of repeat time for policy learning,
        for example, set it to 2 means the policy needs to learn each given batch
        data twice.
    :param int episode_per_test: the number of episodes for one policy evaluation.
    :param int batch_size: the batch size of sample data, which is going to feed in
        the policy network.
    :param int step_per_collect: the number of transitions the collector would
        collect before the network update, i.e., trainer will collect
        "step_per_collect" transitions and do some policy network update repeatedly
        in each epoch.
    :param int episode_per_collect: the number of episodes the collector would
        collect before the network update, i.e., trainer will collect
        "episode_per_collect" episodes and do some policy network update repeatedly
        in each epoch.
    :param function train_fn: a hook called at the beginning of training in each
        epoch. It can be used to perform custom additional operations, with the
        signature ``f(num_epoch: int, step_idx: int) -> None``.
    :param function test_fn: a hook called at the beginning of testing in each
        epoch. It can be used to perform custom additional operations, with the
        signature ``f(num_epoch: int, step_idx: int) -> None``.
    :param function save_best_fn: a hook called when the undiscounted average mean
        reward in evaluation phase gets better, with the signature
        ``f(policy: BasePolicy) -> None``. It was ``save_fn`` previously.
    :param function save_checkpoint_fn: a function to save training process and
        return the saved checkpoint path, with the signature ``f(epoch: int,
        env_step: int, gradient_step: int) -> str``; you can save whatever you want.
    :param bool resume_from_log: resume env_step/gradient_step and other metadata
        from existing tensorboard log. Default to False.
    :param function stop_fn: a function with signature ``f(mean_rewards: float) ->
        bool``, receives the average undiscounted returns of the testing result,
        returns a boolean which indicates whether reaching the goal.
    :param function reward_metric: a function with signature
        ``f(rewards: np.ndarray with shape (num_episode, agent_num)) -> np.ndarray
        with shape (num_episode,)``, used in multi-agent RL. We need to return a
        single scalar for each episode's result to monitor training in the
        multi-agent RL setting. This function specifies what is the desired metric,
        e.g., the reward of agent 1 or the average reward over all agents.
    :param BaseLogger logger: A logger that logs statistics during
        training/testing/updating. Default to a logger that doesn't log anything.
    :param bool verbose: whether to print the information. Default to True.
    :param bool show_progress: whether to display a progress bar when training.
        Default to True.
    :param bool test_in_train: whether to test in the training phase.
        Default to True.
    """

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
        learning_type: str,
        policy: BasePolicy,
        max_epoch: int,
        batch_size: int,
        train_collector: Optional[Collector] = None,
        test_collector: Optional[Collector] = None,
        buffer: Optional[ReplayBuffer] = None,
        step_per_epoch: Optional[int] = None,
        repeat_per_collect: Optional[int] = None,
        episode_per_test: Optional[int] = None,
        update_per_step: Union[int, float] = 1,
        update_per_epoch: Optional[int] = None,
        step_per_collect: Optional[int] = None,
        episode_per_collect: Optional[int] = None,
        train_fn: Optional[Callable[[int, int], None]] = None,
        test_fn: Optional[Callable[[int, Optional[int]], None]] = None,
        stop_fn: Optional[Callable[[float], bool]] = None,
        save_best_fn: Optional[Callable[[BasePolicy], None]] = None,
        save_checkpoint_fn: Optional[Callable[[int, int, int], str]] = None,
        resume_from_log: bool = False,
        reward_metric: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        logger: BaseLogger = LazyLogger(),
        verbose: bool = True,
        show_progress: bool = True,
        test_in_train: bool = True,
        save_fn: Optional[Callable[[BasePolicy], None]] = None,
    ):
        if save_fn:
            deprecation(
                "save_fn in trainer is marked as deprecated and will be "
                "removed in the future. Please use save_best_fn instead."
            )
            assert save_best_fn is None
            save_best_fn = save_fn

        self.policy = policy
        self.buffer = buffer

        self.train_collector = train_collector
        self.test_collector = test_collector

        self.logger = logger
        self.start_time = time.time()
        self.stat: DefaultDict[str, MovAvg] = defaultdict(MovAvg)
        self.best_reward = 0.0
        self.best_reward_std = 0.0
        self.start_epoch = 0
        self.gradient_step = 0
        self.env_step = 0
        self.max_epoch = max_epoch
        self.step_per_epoch = step_per_epoch

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
        self.save_best_fn = save_best_fn
        self.save_checkpoint_fn = save_checkpoint_fn

        self.reward_metric = reward_metric
        self.verbose = verbose
        self.show_progress = show_progress
        self.test_in_train = test_in_train
        self.resume_from_log = resume_from_log

        self.is_run = False
        self.last_rew, self.last_len = 0.0, 0

        self.epoch = self.start_epoch
        self.best_epoch = self.start_epoch
        self.stop_fn_flag = False
        self.iter_num = 0

    def reset(self) -> None:
        """Initialize or reset the instance to yield a new iterator from zero."""
        self.is_run = False
        self.env_step = 0
        if self.resume_from_log:
            self.start_epoch, self.env_step, self.gradient_step = \
                self.logger.restore_data()

        self.last_rew, self.last_len = 0.0, 0
        self.start_time = time.time()
        if self.train_collector is not None:
            self.train_collector.reset_stat()

            if self.train_collector.policy != self.policy:
                self.test_in_train = False
            elif self.test_collector is None:
                self.test_in_train = False

        if self.test_collector is not None:
            assert self.episode_per_test is not None
            assert not isinstance(self.test_collector, AsyncCollector)  # Issue 700
            self.test_collector.reset_stat()
            test_result = test_episode(
                self.policy, self.test_collector, self.test_fn, self.start_epoch,
                self.episode_per_test, self.logger, self.env_step, self.reward_metric
            )
            self.best_epoch = self.start_epoch
            self.best_reward, self.best_reward_std = \
                test_result["rew"], test_result["rew_std"]
        if self.save_best_fn:
            self.save_best_fn(self.policy)

        self.epoch = self.start_epoch
        self.stop_fn_flag = False
        self.iter_num = 0

    def __iter__(self):  # type: ignore
        self.reset()
        return self

    def __next__(self) -> Union[None, Tuple[int, Dict[str, Any], Dict[str, Any]]]:
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

        # set policy in train mode
        self.policy.train()

        epoch_stat: Dict[str, Any] = dict()

        if self.show_progress:
            progress = tqdm.tqdm
        else:
            progress = DummyTqdm

        # perform n step_per_epoch
        with progress(
            total=self.step_per_epoch, desc=f"Epoch #{self.epoch}", **tqdm_config
        ) as t:
            while t.n < t.total and not self.stop_fn_flag:
                data: Dict[str, Any] = dict()
                result: Dict[str, Any] = dict()
                if self.train_collector is not None:
                    data, result, self.stop_fn_flag = self.train_step()
                    t.update(result["n/st"])
                    if self.stop_fn_flag:
                        t.set_postfix(**data)
                        break
                else:
                    assert self.buffer, "No train_collector or buffer specified"
                    result["n/ep"] = len(self.buffer)
                    result["n/st"] = int(self.gradient_step)
                    t.update()

                self.policy_update_fn(data, result)
                t.set_postfix(**data)

            if t.n <= t.total and not self.stop_fn_flag:
                t.update()

        # for offline RL
        if self.train_collector is None:
            self.env_step = self.gradient_step * self.batch_size

        if not self.stop_fn_flag:
            self.logger.save_data(
                self.epoch, self.env_step, self.gradient_step, self.save_checkpoint_fn
            )
            # test
            if self.test_collector is not None:
                test_stat, self.stop_fn_flag = self.test_step()
                if not self.is_run:
                    epoch_stat.update(test_stat)

        if not self.is_run:
            epoch_stat.update({k: v.get() for k, v in self.stat.items()})
            epoch_stat["gradient_step"] = self.gradient_step
            epoch_stat.update(
                {
                    "env_step": self.env_step,
                    "rew": self.last_rew,
                    "len": int(self.last_len),
                    "n/ep": int(result["n/ep"]),
                    "n/st": int(result["n/st"]),
                }
            )
            info = gather_info(
                self.start_time, self.train_collector, self.test_collector,
                self.best_reward, self.best_reward_std
            )
            return self.epoch, epoch_stat, info
        else:
            return None

    def test_step(self) -> Tuple[Dict[str, Any], bool]:
        """Perform one testing step."""
        assert self.episode_per_test is not None
        assert self.test_collector is not None
        stop_fn_flag = False
        test_result = test_episode(
            self.policy, self.test_collector, self.test_fn, self.epoch,
            self.episode_per_test, self.logger, self.env_step, self.reward_metric
        )
        rew, rew_std = test_result["rew"], test_result["rew_std"]
        if self.best_epoch < 0 or self.best_reward < rew:
            self.best_epoch = self.epoch
            self.best_reward = float(rew)
            self.best_reward_std = rew_std
            if self.save_best_fn:
                self.save_best_fn(self.policy)
        if self.verbose:
            print(
                f"Epoch #{self.epoch}: test_reward: {rew:.6f} ± {rew_std:.6f},"
                f" best_reward: {self.best_reward:.6f} ± "
                f"{self.best_reward_std:.6f} in #{self.best_epoch}",
                flush=True
            )
        if not self.is_run:
            test_stat = {
                "test_reward": rew,
                "test_reward_std": rew_std,
                "best_reward": self.best_reward,
                "best_reward_std": self.best_reward_std,
                "best_epoch": self.best_epoch
            }
        else:
            test_stat = {}
        if self.stop_fn and self.stop_fn(self.best_reward):
            stop_fn_flag = True

        return test_stat, stop_fn_flag

    def train_step(self) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
        """Perform one training step."""
        assert self.episode_per_test is not None
        assert self.train_collector is not None
        stop_fn_flag = False
        if self.train_fn:
            self.train_fn(self.epoch, self.env_step)
        result = self.train_collector.collect(
            n_step=self.step_per_collect, n_episode=self.episode_per_collect
        )
        if result["n/ep"] > 0 and self.reward_metric:
            rew = self.reward_metric(result["rews"])
            result.update(rews=rew, rew=rew.mean(), rew_std=rew.std())
        self.env_step += int(result["n/st"])
        self.logger.log_train_data(result, self.env_step)
        self.last_rew = result["rew"] if result["n/ep"] > 0 else self.last_rew
        self.last_len = result["len"] if result["n/ep"] > 0 else self.last_len
        data = {
            "env_step": str(self.env_step),
            "rew": f"{self.last_rew:.2f}",
            "len": str(int(self.last_len)),
            "n/ep": str(int(result["n/ep"])),
            "n/st": str(int(result["n/st"])),
        }
        if result["n/ep"] > 0:
            if self.test_in_train and self.stop_fn and self.stop_fn(result["rew"]):
                assert self.test_collector is not None
                test_result = test_episode(
                    self.policy, self.test_collector, self.test_fn, self.epoch,
                    self.episode_per_test, self.logger, self.env_step
                )
                if self.stop_fn(test_result["rew"]):
                    stop_fn_flag = True
                    self.best_reward = test_result["rew"]
                    self.best_reward_std = test_result["rew_std"]
                else:
                    self.policy.train()

        return data, result, stop_fn_flag

    def log_update_data(self, data: Dict[str, Any], losses: Dict[str, Any]) -> None:
        """Log losses to current logger."""
        for k in losses.keys():
            self.stat[k].add(losses[k])
            losses[k] = self.stat[k].get()
            data[k] = f"{losses[k]:.3f}"
        self.logger.log_update_data(losses, self.gradient_step)

    @abstractmethod
    def policy_update_fn(self, data: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Policy update function for different trainer implementation.

        :param data: information in progress bar.
        :param result: collector's return value.
        """

    def run(self) -> Dict[str, Union[float, str]]:
        """Consume iterator.

        See itertools - recipes. Use functions that consume iterators at C speed
        (feed the entire iterator into a zero-length deque).
        """
        try:
            self.is_run = True
            deque(self, maxlen=0)  # feed the entire iterator into a zero-length deque
            info = gather_info(
                self.start_time, self.train_collector, self.test_collector,
                self.best_reward, self.best_reward_std
            )
        finally:
            self.is_run = False

        return info
