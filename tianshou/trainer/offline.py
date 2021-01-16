import time
import tqdm
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Union, Callable, Optional

from tianshou.policy import BasePolicy
from tianshou.utils import tqdm_config, MovAvg
from tianshou.data import Collector, ReplayBuffer
from tianshou.trainer import test_episode, gather_info


def offline_trainer(
    policy: BasePolicy,
    buffer: ReplayBuffer,
    test_collector: Collector,
    max_epoch: int,
    step_per_epoch: int,
    episode_per_test: Union[int, List[int]],
    batch_size: int,
    test_fn: Optional[Callable[[int, Optional[int]], None]] = None,
    stop_fn: Optional[Callable[[float], bool]] = None,
    save_fn: Optional[Callable[[BasePolicy], None]] = None,
    writer: Optional[SummaryWriter] = None,
    log_interval: int = 1,
    verbose: bool = True,
) -> Dict[str, Union[float, str]]:
    """A wrapper for offline trainer procedure.

    The "step" in trainer means a policy network update.

    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy`
        class.
    :param test_collector: the collector used for testing.
    :type test_collector: :class:`~tianshou.data.Collector`
    :param int max_epoch: the maximum number of epochs for training. The
        training process might be finished before reaching the ``max_epoch``.
    :param int step_per_epoch: the number of policy network updates, so-called
        gradient steps, per epoch.
    :param episode_per_test: the number of episodes for one policy evaluation.
    :param int batch_size: the batch size of sample data, which is going to
        feed in the policy network.
    :param function test_fn: a hook called at the beginning of testing in each
        epoch. It can be used to perform custom additional operations, with the
        signature ``f(num_epoch: int, step_idx: int) -> None``.
    :param function save_fn: a hook called when the undiscounted average mean
        reward in evaluation phase gets better, with the signature ``f(policy:
        BasePolicy) -> None``.
    :param function stop_fn: a function with signature ``f(mean_rewards: float)
        -> bool``, receives the average undiscounted returns of the testing
        result, returns a boolean which indicates whether reaching the goal.
    :param torch.utils.tensorboard.SummaryWriter writer: a TensorBoard
        SummaryWriter; if None is given, it will not write logs to TensorBoard.
    :param int log_interval: the log interval of the writer.
    :param bool verbose: whether to print the information.

    :return: See :func:`~tianshou.trainer.gather_info`.
    """
    gradient_step = 0
    best_epoch, best_reward, best_reward_std = -1, -1.0, 0.0
    stat: Dict[str, MovAvg] = defaultdict(MovAvg)
    start_time = time.time()
    test_collector.reset_stat()

    for epoch in range(1, 1 + max_epoch):
        policy.train()
        with tqdm.trange(
            step_per_epoch, desc=f"Epoch #{epoch}", **tqdm_config
        ) as t:
            for i in t:
                gradient_step += 1
                losses = policy.update(batch_size, buffer)
                data = {"gradient_step": str(gradient_step)}
                for k in losses.keys():
                    stat[k].add(losses[k])
                    data[k] = f"{stat[k].get():.6f}"
                    if writer and gradient_step % log_interval == 0:
                        writer.add_scalar(
                            "train/" + k, stat[k].get(),
                            global_step=gradient_step)
                t.set_postfix(**data)
        # test
        result = test_episode(policy, test_collector, test_fn, epoch,
                              episode_per_test, writer, gradient_step)
        if best_epoch == -1 or best_reward < result["rew"]:
            best_reward, best_reward_std = result["rew"], result["rew_std"]
            best_epoch = epoch
            if save_fn:
                save_fn(policy)
        if verbose:
            print(f"Epoch #{epoch}: test_reward: {result['rew']:.6f} ± "
                  f"{result['rew_std']:.6f}, best_reward: {best_reward:.6f} ± "
                  f"{best_reward_std:.6f} in #{best_epoch}")
        if stop_fn and stop_fn(best_reward):
            break
    return gather_info(start_time, None, test_collector,
                       best_reward, best_reward_std)
