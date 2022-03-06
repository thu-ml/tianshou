import time
from collections import defaultdict, deque
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import tqdm

from tianshou.data import Collector
from tianshou.policy import BasePolicy
from tianshou.trainer import gather_info, test_episode
from tianshou.utils import BaseLogger, LazyLogger, MovAvg, tqdm_config


class OffPolicyTrainer:
    """An iterator wrapper for off-policy trainer procedure.

       Returns an iterator that yields a 3-tuple (epoch, stats, info) of train results
       on every epoch.

       The "step" in trainer means an environment step (a.k.a. transition)."""

    def __init__(
        self,
        policy: BasePolicy,
        train_collector: Collector,
        test_collector: Optional[Collector],
        max_epoch: int,
        step_per_epoch: int,
        step_per_collect: int,
        episode_per_test: int,
        batch_size: int,
        update_per_step: Union[int, float] = 1,
        train_fn: Optional[Callable[[int, int], None]] = None,
        test_fn: Optional[Callable[[int, Optional[int]], None]] = None,
        stop_fn: Optional[Callable[[float], bool]] = None,
        save_fn: Optional[Callable[[BasePolicy], None]] = None,
        save_checkpoint_fn: Optional[Callable[[int, int, int], None]] = None,
        resume_from_log: bool = False,
        reward_metric: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        logger: BaseLogger = LazyLogger(),
        verbose: bool = True,
        test_in_train: bool = True,
    ):
        """
        :param policy: an instance of the :class:`~tianshou.policy.BasePolicy` class.
        :param Collector train_collector: the collector used for training.
        :param Collector test_collector: the collector used for testing. If it's None,
            then no testing will be performed.
        :param int max_epoch: the maximum number of epochs for training. The training
            process might be finished before reaching ``max_epoch`` if ``stop_fn`` is
            set.
        :param int step_per_epoch: the number of transitions collected per epoch.
        :param int step_per_collect: the number of transitions the collector would
            collect before the network update, i.e., trainer will collect
            "step_per_collect" transitions and do some policy network update repeatedly
             in each epoch.
        :param episode_per_test: the number of episodes for one policy evaluation.
        :param int batch_size: the batch size of sample data, which is going to feed in
            the policy network.
        :param int/float update_per_step: the number of times the policy network would
            be updated per transition after (step_per_collect) transitions are
            collected, e.g., if update_per_step set to 0.3, and step_per_collect is 256
            , policy will be updated round(256 * 0.3 = 76.8) = 77 times after 256
            transitions are collected by the collector. Default to 1.
        :param function train_fn: a hook called at the beginning of training in each
            epoch. It can be used to perform custom additional operations, with the
             signature ``f(num_epoch: int, step_idx: int) -> None``.
        :param function test_fn: a hook called at the beginning of testing in each
            epoch. It can be used to perform custom additional operations, with the
            signature ``f(num_epoch: int, step_idx: int) -> None``.
        :param function save_fn: a hook called when the undiscounted average mean
           reward in evaluation phase gets better, with the signature
           ``f(policy: BasePolicy) ->  None``.
        :param function save_checkpoint_fn: a function to save training process, with
            the signature ``f(epoch: int, env_step: int, gradient_step: int) -> None``;
             you can save whatever you want.
        :param bool resume_from_log: resume env_step/gradient_step and other metadata
            from existing tensorboard log. Default to False.
        :param function stop_fn: a function with signature ``f(mean_rewards: float) ->
            bool``, receives the average undiscounted returns of the testing result,
            returns a boolean which indicates whether reaching the goal.
        :param function reward_metric: a function with signature
            ``f(rewards: np.ndarray with shape (num_episode, agent_num)) ->
            np.ndarray with shape (num_episode,)``, used in multi-agent RL. We need to
            return a single scalar for each episode's result to monitor training in the
            multi-agent RL setting. This function specifies what is the desired metric,
            e.g., the reward of agent 1 or the average reward over all agents.
        :param BaseLogger logger: A logger that logs statistics during
            training/testing/updating. Default to a logger that doesn't log anything.
        :param bool verbose: whether to print the information. Default to True.
        :param bool test_in_train: whether to test in the training phase.
            Default to True.

        """
        self.is_run = False
        self.policy = policy

        self.train_collector = train_collector
        self.test_collector = test_collector

        self.max_epoch = max_epoch
        self.step_per_epoch = step_per_epoch
        self.step_per_collect = step_per_collect
        self.episode_per_test = episode_per_test
        self.batch_size = batch_size
        self.update_per_step = update_per_step

        self.train_fn = train_fn
        self.test_fn = test_fn
        self.stop_fn = stop_fn
        self.save_fn = save_fn
        self.save_checkpoint_fn = save_checkpoint_fn

        self.reward_metric = reward_metric
        self.logger = logger
        self.verbose = verbose
        self.test_in_train = test_in_train

        self.start_epoch, self.env_step, self.gradient_step = 0, 0, 0
        self.best_reward, self.best_reward_std = 0.0, 0.0

        if resume_from_log:
            self.start_epoch, self.env_step, self.gradient_step = logger.restore_data()
        self.last_rew, self.last_len = 0.0, 0
        self.stat: Dict[str, MovAvg] = defaultdict(MovAvg)
        self.start_time = time.time()
        self.train_collector.reset_stat()
        self.test_in_train = self.test_in_train and (
            self.train_collector.policy == policy and self.test_collector is not None
        )

        if test_collector is not None:
            self.test_c: Collector = test_collector  # for mypy
            self.test_collector.reset_stat()
            test_result = test_episode(
                self.policy, self.test_c, self.test_fn, self.start_epoch,
                self.episode_per_test, self.logger, self.env_step, self.reward_metric
            )
            self.best_epoch = self.start_epoch
            self.best_reward, self.best_reward_std = test_result["rew"], test_result[
                "rew_std"]
        if save_fn:
            save_fn(policy)

        self.epoch = self.start_epoch
        self.exit_flag = 0

    def __iter__(self):  # type: ignore
        return self

    def __next__(self) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
        self.epoch += 1

        # exit flag 1, when test_in_train and stop_fn succeeds on result["rew"]
        if self.test_in_train and self.stop_fn and self.exit_flag == 1:
            raise StopIteration

        # iterator exhaustion check
        if self.epoch >= self.max_epoch:
            if self.test_collector is None and self.save_fn:
                self.save_fn(self.policy)
            raise StopIteration

        # stop_fn criterion
        if self.test_collector is not None and self.stop_fn and self.stop_fn(
            self.best_reward
        ):
            raise StopIteration

        # set policy in train mode
        self.policy.train()

        # Performs n step_per_epoch
        with tqdm.tqdm(
            total=self.step_per_epoch, desc=f"Epoch #{self.epoch}", **tqdm_config
        ) as t:
            while t.n < t.total:
                if self.train_fn:
                    self.train_fn(self.epoch, self.env_step)
                result = self.train_collector.collect(n_step=self.step_per_collect)
                if result["n/ep"] > 0 and self.reward_metric:
                    rew = self.reward_metric(result["rews"])
                    result.update(rews=rew, rew=rew.mean(), rew_std=rew.std())
                self.env_step += int(result["n/st"])
                t.update(result["n/st"])
                self.logger.log_train_data(result, self.env_step)
                self.last_rew = result['rew'] if result["n/ep"] > 0 else self.last_rew
                self.last_len = result['len'] if result["n/ep"] > 0 else self.last_len
                data = {
                    "env_step": str(self.env_step),
                    "rew": f"{self.last_rew:.2f}",
                    "len": str(int(self.last_len)),
                    "n/ep": str(int(result["n/ep"])),
                    "n/st": str(int(result["n/st"])),
                }
                if result["n/ep"] > 0:
                    if self.test_in_train and self.stop_fn and self.stop_fn(
                        result["rew"]
                    ):
                        test_result = test_episode(
                            self.policy, self.test_c, self.test_fn, self.epoch,
                            self.episode_per_test, self.logger, self.env_step
                        )
                        if self.stop_fn(test_result["rew"]):
                            if self.save_fn:
                                self.save_fn(self.policy)
                            self.logger.save_data(
                                self.epoch, self.env_step, self.gradient_step,
                                self.save_checkpoint_fn
                            )
                            t.set_postfix(**data)
                            if not self.is_run:
                                epoch_stat: Dict[str, Any] = {
                                    k: v.get()
                                    for k, v in self.stat.items()
                                }
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
                            self.exit_flag = 1
                            self.best_reward = test_result["rew"]
                            self.best_reward_std = test_result["rew_std"]

                            if not self.is_run:
                                info = gather_info(
                                    self.start_time, self.train_collector,
                                    self.test_collector, self.best_reward,
                                    self.best_reward_std
                                )
                                return self.epoch, epoch_stat, info
                            else:
                                return 0, {}, {}
                        else:
                            self.policy.train()
                for _ in range(round(self.update_per_step * result["n/st"])):
                    self.gradient_step += 1
                    losses = self.policy.update(
                        self.batch_size, self.train_collector.buffer
                    )
                    for k in losses.keys():
                        self.stat[k].add(losses[k])
                        losses[k] = self.stat[k].get()
                        data[k] = f"{losses[k]:.3f}"
                    self.logger.log_update_data(losses, self.gradient_step)
                    t.set_postfix(**data)
            if t.n <= t.total:
                t.update()
        self.logger.save_data(
            self.epoch, self.env_step, self.gradient_step, self.save_checkpoint_fn
        )

        if not self.is_run:
            epoch_stat = {k: v.get() for k, v in self.stat.items()}
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

        # test
        if self.test_collector is not None:
            test_result = test_episode(
                self.policy, self.test_c, self.test_fn, self.epoch,
                self.episode_per_test, self.logger, self.env_step, self.reward_metric
            )
            rew, rew_std = test_result["rew"], test_result["rew_std"]
            if self.best_epoch < 0 or self.best_reward < rew:
                self.best_epoch = self.epoch
                self.best_reward = rew
                self.best_reward_std = rew_std
                if self.save_fn:
                    self.save_fn(self.policy)
            if self.verbose:
                print(
                    f"Epoch #{self.epoch}: test_reward: {rew:.6f} ± {rew_std:.6f},"
                    f" best_reward: {self.best_reward:.6f} ± "
                    f"{self.best_reward_std:.6f} in #{self.best_epoch}"
                )
            if not self.is_run:
                epoch_stat.update(
                    {
                        "test_reward": rew,
                        "test_reward_std": rew_std,
                        "best_reward": self.best_reward,
                        "best_reward_std": self.best_reward_std,
                        "best_epoch": self.best_epoch
                    }
                )

        # return iterator -> next(self)
        if not self.is_run:
            info = gather_info(
                self.start_time, self.train_collector, self.test_collector,
                self.best_reward, self.best_reward_std
            )
            return self.epoch, epoch_stat, info
        else:
            return 0, {}, {}

    def run(self) -> Dict[str, Union[float, str]]:
        """
        Consume iterator, see itertools-recipes. Use functions that consume
        iterators at C speed (feed the entire iterator into a zero-length deque).
        """
        try:
            self.is_run = True
            i = iter(self)
            deque(i, maxlen=0)  # feed the entire iterator into a zero-length deque
            info = gather_info(
                self.start_time, None, self.test_collector, self.best_reward,
                self.best_reward_std
            )
        finally:
            self.is_run = False

        return info


def offpolicy_trainer(*args, **kwargs) -> Dict[str, Union[float, str]]:  # type: ignore
    return OffPolicyTrainer(*args, **kwargs).run()


offpolicy_trainer_iter = OffPolicyTrainer
