import time
from collections.abc import Callable
from dataclasses import asdict

import numpy as np

from tianshou.data import (
    CollectStats,
    InfoStats,
    SequenceSummaryStats,
    TimingStats,
)
from tianshou.data.collector import BaseCollector
from tianshou.utils import BaseLogger


def test_episode(
    collector: BaseCollector,
    test_fn: Callable[[int, int | None], None] | None,
    epoch: int,
    n_episode: int,
    logger: BaseLogger | None = None,
    global_step: int | None = None,
    reward_metric: Callable[[np.ndarray], np.ndarray] | None = None,
) -> CollectStats:
    """A simple wrapper of testing policy in collector."""
    collector.reset(reset_stats=False)
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
    best_score: float,
    best_reward: float,
    best_reward_std: float,
    train_collector: BaseCollector | None = None,
    test_collector: BaseCollector | None = None,
) -> InfoStats:
    """A simple wrapper of gathering information from collectors.

    :return: InfoStats object with times computed based on the `start_time` and
        episode/step counts read off the collectors. No computation of
        expensive statistics is done here.
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
        best_score=best_score,
        best_reward=best_reward,
        best_reward_std=best_reward_std,
        train_step=train_collector.collect_step if train_collector is not None else 0,
        train_episode=train_collector.collect_episode if train_collector is not None else 0,
        test_step=test_collector.collect_step if test_collector is not None else 0,
        test_episode=test_collector.collect_episode if test_collector is not None else 0,
        timing=timing_stat,
    )
