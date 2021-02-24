import time
import numpy as np
from typing import Any, Dict, Union, Callable, Optional

from tianshou.data import Collector
from tianshou.policy import BasePolicy
from tianshou.utils import BaseLogger


def test_episode(
    policy: BasePolicy,
    collector: Collector,
    test_fn: Optional[Callable[[int, Optional[int]], None]],
    epoch: int,
    n_episode: int,
    logger: Optional[BaseLogger] = None,
    global_step: Optional[int] = None,
    reward_metric: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, Any]:
    """A simple wrapper of testing policy in collector."""
    collector.reset_env()
    collector.reset_buffer()
    policy.eval()
    if test_fn:
        test_fn(epoch, global_step)
    result = collector.collect(n_episode=n_episode)
    if reward_metric:
        result["rews"] = reward_metric(result["rews"])
    if logger and global_step is not None:
        logger.log_test_data(result, global_step)
    return result


def gather_info(
    start_time: float,
    train_c: Optional[Collector],
    test_c: Collector,
    best_reward: float,
    best_reward_std: float,
) -> Dict[str, Union[float, str]]:
    """A simple wrapper of gathering information from collectors.

    :return: A dictionary with the following keys:

        * ``train_step`` the total collected step of training collector;
        * ``train_episode`` the total collected episode of training collector;
        * ``train_time/collector`` the time for collecting transitions in the \
            training collector;
        * ``train_time/model`` the time for training models;
        * ``train_speed`` the speed of training (env_step per second);
        * ``test_step`` the total collected step of test collector;
        * ``test_episode`` the total collected episode of test collector;
        * ``test_time`` the time for testing;
        * ``test_speed`` the speed of testing (env_step per second);
        * ``best_reward`` the best reward over the test results;
        * ``duration`` the total elapsed time.
    """
    duration = time.time() - start_time
    model_time = duration - test_c.collect_time
    test_speed = test_c.collect_step / test_c.collect_time
    result: Dict[str, Union[float, str]] = {
        "test_step": test_c.collect_step,
        "test_episode": test_c.collect_episode,
        "test_time": f"{test_c.collect_time:.2f}s",
        "test_speed": f"{test_speed:.2f} step/s",
        "best_reward": best_reward,
        "best_result": f"{best_reward:.2f} ± {best_reward_std:.2f}",
        "duration": f"{duration:.2f}s",
        "train_time/model": f"{model_time:.2f}s",
    }
    if train_c is not None:
        model_time -= train_c.collect_time
        train_speed = train_c.collect_step / (duration - test_c.collect_time)
        result.update({
            "train_step": train_c.collect_step,
            "train_episode": train_c.collect_episode,
            "train_time/collector": f"{train_c.collect_time:.2f}s",
            "train_time/model": f"{model_time:.2f}s",
            "train_speed": f"{train_speed:.2f} step/s",
        })
    return result
