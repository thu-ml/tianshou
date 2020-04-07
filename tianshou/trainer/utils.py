import time
import numpy as np


def test_episode(policy, collector, test_fn, epoch, n_episode):
    """A simple wrapper of testing policy in collector."""
    collector.reset_env()
    collector.reset_buffer()
    policy.eval()
    if test_fn:
        test_fn(epoch)
    if collector.get_env_num() > 1 and np.isscalar(n_episode):
        n = collector.get_env_num()
        n_ = np.zeros(n) + n_episode // n
        n_[:n_episode % n] += 1
        n_episode = list(n_)
    return collector.collect(n_episode=n_episode)


def gather_info(start_time, train_c, test_c, best_reward):
    """A simple wrapper of gathering information from collectors.

    :return: A dictionary with the following keys:

        * ``train_step`` the total collected step of training collector;
        * ``train_episode`` the total collected episode of training collector;
        * ``train_time/collector`` the time for collecting frames in the \
            training collector;
        * ``train_time/model`` the time for training models;
        * ``train_speed`` the speed of training (frames per second);
        * ``test_step`` the total collected step of test collector;
        * ``test_episode`` the total collected episode of test collector;
        * ``test_time`` the time for testing;
        * ``test_speed`` the speed of testing (frames per second);
        * ``best_reward`` the best reward over the test results;
        * ``duration`` the total elapsed time.
    """
    duration = time.time() - start_time
    model_time = duration - train_c.collect_time - test_c.collect_time
    train_speed = train_c.collect_step / (duration - test_c.collect_time)
    test_speed = test_c.collect_step / test_c.collect_time
    return {
        'train_step': train_c.collect_step,
        'train_episode': train_c.collect_episode,
        'train_time/collector': f'{train_c.collect_time:.2f}s',
        'train_time/model': f'{model_time:.2f}s',
        'train_speed': f'{train_speed:.2f} step/s',
        'test_step': test_c.collect_step,
        'test_episode': test_c.collect_episode,
        'test_time': f'{test_c.collect_time:.2f}s',
        'test_speed': f'{test_speed:.2f} step/s',
        'best_reward': best_reward,
        'duration': f'{duration:.2f}s',
    }
