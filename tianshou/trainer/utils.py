import time
from collections.abc import Callable
from dataclasses import asdict

import numpy as np

from tianshou.data import (
    Collector,
    CollectStats,
    InfoStats,
    SequenceSummaryStats,
    TimingStats,
)
from tianshou.policy import BasePolicy
from tianshou.utils import BaseLogger


def test_episode(
    policy: BasePolicy,
    collector: Collector,
    test_fn: Callable[[int, int | None], None] | None,
    epoch: int,
    n_episode: int,
    logger: BaseLogger | None = None,
    global_step: int | None = None,
    reward_metric: Callable[[np.ndarray], np.ndarray] | None = None,
) -> CollectStats:
    """A simple wrapper of testing policy in collector."""
    collector.reset_env()
    collector.reset_buffer()
    policy.eval()
    if test_fn:
        test_fn(epoch, global_step)
    result = collector.collect(n_episode=n_episode)
    if reward_metric:  # TODO: move into collector
        rew = reward_metric(result.returns)
        result.returns = rew
        result.returns_stat = SequenceSummaryStats.from_sequence(rew)
    if logger and global_step is not None:
        assert result.n_collected_episodes > 0
        logger.log_test_data(asdict(result), global_step)
    return result


def gather_info(
    start_time: float,
    policy_update_time: float,
    gradient_step: int,
    best_reward: float,
    best_reward_std: float,
    train_collector: Collector | None = None,
    test_collector: Collector | None = None,
) -> InfoStats:
    """A simple wrapper of gathering information from collectors.

    :return: A dataclass object with the following members (depending on available collectors):

        * ``gradient_step`` the total number of gradient steps;
        * ``best_reward`` the best reward over the test results;
        * ``best_reward_std`` the standard deviation of best reward over the test results;
        * ``train_step`` the total collected step of training collector;
        * ``train_episode`` the total collected episode of training collector;
        * ``test_step`` the total collected step of test collector;
        * ``test_episode`` the total collected episode of test collector;
        * ``timing`` the timing statistics, with the following members:
        * ``total_time`` the total time elapsed;
        * ``train_time`` the total time elapsed for learning training (collecting samples plus model update);
        * ``train_time_collect`` the time for collecting transitions in the \
            training collector;
        * ``train_time_update`` the time for training models;
        * ``test_time`` the time for testing;
        * ``update_speed`` the speed of updating (env_step per second).
    """
    duration = max(0.0, time.time() - start_time)
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
        train_time_update=policy_update_time,
        test_time=test_time,
        update_speed=update_speed,
    )

    return InfoStats(
        gradient_step=gradient_step,
        best_reward=best_reward,
        best_reward_std=best_reward_std,
        train_step=train_collector.collect_step if train_collector is not None else 0,
        train_episode=train_collector.collect_episode if train_collector is not None else 0,
        test_step=test_collector.collect_step if test_collector is not None else 0,
        test_episode=test_collector.collect_episode if test_collector is not None else 0,
        timing=timing_stat,
    )
