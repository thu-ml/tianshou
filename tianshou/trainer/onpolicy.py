import time
import tqdm
import numpy as np
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Union, Callable, Optional

from tianshou.data import Collector
from tianshou.policy import BasePolicy
from tianshou.utils import tqdm_config, MovAvg
from tianshou.trainer import test_episode, gather_info


def onpolicy_trainer(
    policy: BasePolicy,
    train_collector: Collector,
    test_collector: Collector,
    max_epoch: int,
    step_per_epoch: int,
    step_per_collect: int,
    repeat_per_collect: int,
    episode_per_test: int,
    batch_size: int,
    train_fn: Optional[Callable[[int, int], None]] = None,
    test_fn: Optional[Callable[[int, Optional[int]], None]] = None,
    stop_fn: Optional[Callable[[float], bool]] = None,
    save_fn: Optional[Callable[[BasePolicy], None]] = None,
    reward_metric: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    writer: Optional[SummaryWriter] = None,
    log_interval: int = 1,
    verbose: bool = True,
    test_in_train: bool = True,
    collect_method: str = "episode",
) -> Dict[str, Union[float, str]]:
    """A wrapper for on-policy trainer procedure.

    The "step" in trainer means a policy network update.

    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy` class.
    :param train_collector: the collector used for training.
    :type train_collector: :class:`~tianshou.data.Collector`
    :param test_collector: the collector used for testing.
    :type test_collector: :class:`~tianshou.data.Collector`
    :param int max_epoch: the maximum number of epochs for training. The
        training process might be finished before reaching the ``max_epoch``.
    :param int step_per_epoch: the number of environment frames collected per epoch.
    :param int step_per_collect: the number of episodes the collector would
            collect before the network update in "episode" collect mode(defalut),
            the number of frames the collector would collect in "step" collect
            mode.
    :param int step_per_collect: the number of episodes the collector would
        collect before the network update. In other words, collect some
        episodes and do one policy network update.
    :param int repeat_per_collect: the number of repeat time for policy
        learning, for example, set it to 2 means the policy needs to learn each
        given batch data twice.
    :param episode_per_test: the number of episodes for one policy evaluation.
    :type episode_per_test: int or list of ints
    :param int batch_size: the batch size of sample data, which is going to
        feed in the policy network.
    :param function train_fn: a hook called at the beginning of training in
        each epoch. It can be used to perform custom additional operations,
        with the signature ``f(num_epoch: int, step_idx: int) -> None``.
    :param function test_fn: a hook called at the beginning of testing in each
        epoch. It can be used to perform custom additional operations, with the
        signature ``f(num_epoch: int, step_idx: int) -> None``.
    :param function save_fn: a hook called when the undiscounted average mean
        reward in evaluation phase gets better, with the signature ``f(policy:
        BasePolicy) -> None``.
    :param function stop_fn: a function with signature ``f(mean_rewards: float)
        -> bool``, receives the average undiscounted returns of the testing
        result, returns a boolean which indicates whether reaching the goal.
    :param function reward_metric: a function with signature ``f(rewards: np.ndarray
        with shape (num_episode, agent_num)) -> np.ndarray with shape (num_episode,)``,
        used in multi-agent RL. We need to return a single scalar for each episode's
        result to monitor training in the multi-agent RL setting. This function
        specifies what is the desired metric, e.g., the reward of agent 1 or the
        average reward over all agents.
    :param torch.utils.tensorboard.SummaryWriter writer: a TensorBoard
        SummaryWriter; if None is given, it will not write logs to TensorBoard.
    :param int log_interval: the log interval of the writer.
    :param bool verbose: whether to print the information.
    :param bool test_in_train: whether to test in the training phase.
    :param string collect_method: specifies collect mode. Can be either "episode"
        or "step".

    :return: See :func:`~tianshou.trainer.gather_info`.
    """
    env_step, gradient_step = 0, 0
    best_epoch, best_reward, best_reward_std = -1, -1.0, 0.0
    stat: Dict[str, MovAvg] = defaultdict(MovAvg)
    start_time = time.time()
    train_collector.reset_stat()
    test_collector.reset_stat()
    test_in_train = test_in_train and train_collector.policy == policy
    test_episode(policy, test_collector, test_fn, 0,
                 episode_per_test, writer, env_step)
    for epoch in range(1, 1 + max_epoch):
        # train
        policy.train()
        with tqdm.tqdm(
            total=step_per_epoch, desc=f"Epoch #{epoch}", **tqdm_config
        ) as t:
            while t.n < t.total:
                if train_fn:
                    train_fn(epoch, env_step)
                result = train_collector.collect(
                                **{"n_" + collect_method: step_per_collect})
                if reward_metric:
                    result["rews"] = reward_metric(result["rews"])
                env_step += int(result["n/st"])
                t.update(result["n/st"])
                data = {
                    "env_step": str(env_step),
                    "rew": f"{result['rews'].mean():.2f}",
                    "len": str(int(result["lens"].mean())),
                    "n/ep": str(int(result["n/ep"])),
                    "n/st": str(int(result["n/st"])),
                }
                if writer and env_step % log_interval == 0:
                    writer.add_scalar(
                        "train/rew", result['rews'].mean(), global_step=env_step)
                    writer.add_scalar(
                        "train/len", result['lens'].mean(), global_step=env_step)
                if test_in_train and stop_fn and stop_fn(result["rews"].mean()):
                    test_result = test_episode(
                        policy, test_collector, test_fn,
                        epoch, episode_per_test, writer, env_step)
                    if stop_fn(test_result["rews"].mean()):
                        if save_fn:
                            save_fn(policy)
                        t.set_postfix(**data)
                        return gather_info(
                            start_time, train_collector, test_collector,
                            test_result["rews"].mean(), test_result["rews"].std())
                    else:
                        policy.train()
                losses = policy.update(
                    0, train_collector.buffer,
                    batch_size=batch_size, repeat=repeat_per_collect)
                train_collector.reset_buffer()
                step = max([1] + [
                    len(v) for v in losses.values() if isinstance(v, list)])
                gradient_step += step
                for k in losses.keys():
                    stat[k].add(losses[k])
                    data[k] = f"{stat[k].get():.6f}"
                    if writer and gradient_step % log_interval == 0:
                        writer.add_scalar(
                            k, stat[k].get(), global_step=gradient_step)
                t.set_postfix(**data)
            if t.n <= t.total:
                t.update()
        # test
        test_result = test_episode(policy, test_collector, test_fn, epoch,
                                   episode_per_test, writer, env_step)
        if best_epoch == -1 or best_reward < result["rews"].mean():
            best_reward, best_reward_std = result["rews"].mean(), result["rews"].std()
            best_epoch = epoch
            if save_fn:
                save_fn(policy)
        if verbose:
            print(f"Epoch #{epoch}: test_reward: {result['rews'].mean():.6f} ± "
                  f"{result['rews'].std():.6f}, best_reward: {best_reward:.6f} ± "
                  f"{best_reward_std:.6f} in #{best_epoch}")
        if stop_fn and stop_fn(best_reward):
            break
    return gather_info(start_time, train_collector, test_collector,
                       best_reward, best_reward_std)
