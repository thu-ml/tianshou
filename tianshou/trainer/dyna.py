import time
from collections import defaultdict
from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np
import tqdm

from tianshou.data import Collector, SimpleReplayBuffer
from tianshou.env.fake import GaussianModel
from tianshou.policy import BasePolicy, MBPOPolicy
from tianshou.trainer import gather_info, test_episode
from tianshou.utils import BaseLogger, LazyLogger, MovAvg, tqdm_config


def dyna_trainer(
    policy: MBPOPolicy,
    model: GaussianModel,
    train_collector: Collector,
    test_collector: Collector,
    model_collector: Collector,
    max_epoch: int,
    step_per_epoch: int,
    step_per_collect: int,
    episode_per_test: int,
    batch_size: int = 256,
    rollout_batch_size: int = 100000,
    rollout_schedule: Sequence[int] = (1, 1, 1, 1),
    real_ratio: float = 0.1,
    start_timesteps: int = 0,
    model_train_freq: int = 250,
    model_retain_epochs: int = 1,
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
) -> Dict[str, Union[float, str]]:
    """A wrapper for dyna style trainer procedure.

    The "step" in trainer means an environment step (a.k.a. transition).

    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy` class.
    :param Collector train_collector: the collector used for training.
    :param Collector test_collector: the collector used for testing.
    :param Collector model_collector: the collector used for collecting model rollouts.
    :param int max_epoch: the maximum number of epochs for training. The training
        process might be finished before reaching ``max_epoch`` if ``stop_fn`` is set.
    :param int step_per_epoch: the number of transitions collected per epoch.
    :param int step_per_collect: the number of transitions the collector would collect
        before the network update, i.e., trainer will collect "step_per_collect"
        transitions and do some policy network update repeatly in each epoch.
    :param episode_per_test: the number of episodes for one policy evaluation.
    :param int batch_size: the batch size of sample data, which is going to feed in the
        policy network.
    :param int rollout_batch_size: the batch size of rollouts in parallel.
    :param Sequence rollout_schedule: scheduler for rollout length of each epoch.
    :param float real_ratio: ratio of samples from real environment interactions in
        each gradient update.
    :param int model_retain_epochs: Number of epochs that retains samples in the
        buffer.
    :param int/float update_per_step: the number of times the policy network would be
        updated per transition after (step_per_collect) transitions are collected,
        e.g., if update_per_step set to 0.3, and step_per_collect is 256, policy will
        be updated round(256 * 0.3 = 76.8) = 77 times after 256 transitions are
        collected by the collector. Default to 1.
    :param function train_fn: a hook called at the beginning of training in each epoch.
        It can be used to perform custom additional operations, with the signature ``f(
        num_epoch: int, step_idx: int) -> None``.
    :param function test_fn: a hook called at the beginning of testing in each epoch.
        It can be used to perform custom additional operations, with the signature ``f(
        num_epoch: int, step_idx: int) -> None``.
    :param function save_fn: a hook called when the undiscounted average mean reward in
        evaluation phase gets better, with the signature ``f(policy: BasePolicy) ->
        None``.
    :param function save_checkpoint_fn: a function to save training process, with the
        signature ``f(epoch: int, env_step: int, gradient_step: int) -> None``; you can
        save whatever you want.
    :param bool resume_from_log: resume env_step/gradient_step and other metadata from
        existing tensorboard log. Default to False.
    :param function stop_fn: a function with signature ``f(mean_rewards: float) ->
        bool``, receives the average undiscounted returns of the testing result,
        returns a boolean which indicates whether reaching the goal.
    :param function reward_metric: a function with signature ``f(rewards: np.ndarray
        with shape (num_episode, agent_num)) -> np.ndarray with shape (num_episode,)``,
        used in multi-agent RL. We need to return a single scalar for each episode's
        result to monitor training in the multi-agent RL setting. This function
        specifies what is the desired metric, e.g., the reward of agent 1 or the
        average reward over all agents.
    :param BaseLogger logger: A logger that logs statistics during
        training/testing/updating. Default to a logger that doesn't log anything.
    :param bool verbose: whether to print the information. Default to True.
    :param bool test_in_train: whether to test in the training phase. Default to True.

    :return: See :func:`~tianshou.trainer.gather_info`.
    """
    # Initial steps
    train_collector.collect(n_step=start_timesteps, random=True)

    env_batch_size = int(batch_size * real_ratio)
    model_batch_size = batch_size - env_batch_size
    assert env_batch_size > 0 and model_batch_size > 0

    start_epoch, env_step, gradient_step, last_train_step = 0, 0, 0, 0
    if resume_from_log:
        start_epoch, env_step, gradient_step = logger.restore_data()
    last_rew, last_len = 0.0, 0
    stat: Dict[str, MovAvg] = defaultdict(MovAvg)
    start_time = time.time()
    train_collector.reset_stat()
    test_collector.reset_stat()
    test_in_train = test_in_train and train_collector.policy == policy
    test_result = test_episode(
        policy, test_collector, test_fn, start_epoch, episode_per_test, logger,
        env_step, reward_metric
    )
    best_epoch = start_epoch
    best_reward, best_reward_std = test_result["rew"], test_result["rew_std"]
    if save_fn:
        save_fn(policy)

    epoch_low, epoch_high, min_length, max_length = rollout_schedule
    rollouts_per_epoch = rollout_batch_size * step_per_epoch / model_train_freq

    for epoch in range(1 + start_epoch, 1 + max_epoch):
        # train
        policy.train()

        # Determine rollout length
        if epoch <= epoch_low:
            rollout_length = min_length
        else:
            dx = (epoch - epoch_low) / (epoch_high - epoch_low)
            dx = min(dx, 1)
            rollout_length = int(dx * (max_length - min_length) + min_length)

        with tqdm.tqdm(
            total=step_per_epoch, desc=f"Epoch #{epoch}", **tqdm_config
        ) as t:
            while t.n < t.total:
                if (
                    env_step - last_train_step >= model_train_freq or env_step == 0
                ) and 0. <= model.ratio < 1.:
                    last_train_step = env_step
                    # Train model
                    batch, _ = train_collector.buffer.sample(batch_size=0)
                    train_info = model.train(batch)
                    train_info["model/rollout_length"] = rollout_length
                    logger.write(
                        step_type="",
                        step=env_step,
                        data=train_info,
                    )
                    # Rollout
                    model_steps_per_epoch = int(rollout_length * rollouts_per_epoch)
                    new_size = model_retain_epochs * model_steps_per_epoch
                    if model_collector.buffer.maxsize < new_size:
                        temp_buffer = model_collector.buffer
                        model_collector.buffer = SimpleReplayBuffer(new_size)
                        model_collector.buffer.update(temp_buffer)
                    model_collector.reset_env()
                    model_collector.collect(n_step=rollout_batch_size * rollout_length)

                if train_fn:
                    train_fn(epoch, env_step)
                result = train_collector.collect(n_step=step_per_collect)
                if result["n/ep"] > 0 and reward_metric:
                    rew = reward_metric(result["rews"])
                    result.update(rews=rew, rew=rew.mean(), rew_std=rew.std())
                env_step += int(result["n/st"])
                t.update(result["n/st"])
                logger.log_train_data(result, env_step)
                last_rew = result["rew"] if result["n/ep"] > 0 else last_rew
                last_len = result["len"] if result["n/ep"] > 0 else last_len
                data = {
                    "env_step": str(env_step),
                    "rew": f"{last_rew:.2f}",
                    "len": str(int(last_len)),
                    "n/ep": str(int(result["n/ep"])),
                    "n/st": str(int(result["n/st"])),
                }
                if result["n/ep"] > 0:
                    if test_in_train and stop_fn and stop_fn(result["rew"]):
                        test_result = test_episode(
                            policy, test_collector, test_fn, epoch, episode_per_test,
                            logger, env_step
                        )
                        if stop_fn(test_result["rew"]):
                            if save_fn:
                                save_fn(policy)
                            logger.save_data(
                                epoch, env_step, gradient_step, save_checkpoint_fn
                            )
                            t.set_postfix(**data)
                            return gather_info(
                                start_time, train_collector, test_collector,
                                test_result["rew"], test_result["rew_std"]
                            )
                        else:
                            policy.train()

                for _ in range(round(update_per_step * result["n/st"])):
                    gradient_step += 1
                    losses = policy.update(
                        env_batch_size,
                        train_collector.buffer,
                        model_batch_size,
                        model_collector.buffer,
                    )
                    for k in losses.keys():
                        stat[k].add(losses[k])
                        losses[k] = stat[k].get()
                        data[k] = f"{losses[k]:.3f}"
                    logger.log_update_data(losses, gradient_step)
                    t.set_postfix(**data)
            if t.n <= t.total:
                t.update()
        # test
        test_result = test_episode(
            policy, test_collector, test_fn, epoch, episode_per_test, logger, env_step,
            reward_metric
        )
        rew, rew_std = test_result["rew"], test_result["rew_std"]
        if best_epoch < 0 or best_reward < rew:
            best_epoch, best_reward, best_reward_std = epoch, rew, rew_std
            if save_fn:
                save_fn(policy)
        logger.save_data(epoch, env_step, gradient_step, save_checkpoint_fn)
        if verbose:
            print(
                f"Epoch #{epoch}: test_reward: {rew:.6f} ± {rew_std:.6f}, best_rew"
                f"ard: {best_reward:.6f} ± {best_reward_std:.6f} in #{best_epoch}"
            )
        if stop_fn and stop_fn(best_reward):
            break
    return gather_info(
        start_time, train_collector, test_collector, best_reward, best_reward_std
    )
