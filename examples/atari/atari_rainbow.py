#!/usr/bin/env python3

import datetime
import os
import pprint
import sys

import numpy as np
import torch
from sensai.util import logging

from tianshou.algorithm import C51, RainbowDQN
from tianshou.algorithm.algorithm_base import Algorithm
from tianshou.algorithm.modelfree.c51 import C51Policy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import (
    Collector,
    CollectStats,
    PrioritizedVectorReplayBuffer,
    VectorReplayBuffer,
)
from tianshou.env.atari.atari_network import RainbowNet
from tianshou.env.atari.atari_wrapper import make_atari_env
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.trainer import OffPolicyTrainerParams

log = logging.getLogger(__name__)


def main(
    task: str = "PongNoFrameskip-v4",
    seed: int = 0,
    scale_obs: int = 0,
    eps_test: float = 0.005,
    eps_train: float = 1.0,
    eps_train_final: float = 0.05,
    buffer_size: int = 100000,
    lr: float = 0.0000625,
    gamma: float = 0.99,
    num_atoms: int = 51,
    v_min: float = -10.0,
    v_max: float = 10.0,
    noisy_std: float = 0.1,
    no_dueling: bool = False,
    no_noisy: bool = False,
    no_priority: bool = False,
    alpha: float = 0.5,
    beta: float = 0.4,
    beta_final: float = 1.0,
    beta_anneal_step: int = 5000000,
    no_weight_norm: bool = False,
    n_step: int = 3,
    target_update_freq: int = 500,
    epoch: int = 100,
    epoch_num_steps: int = 100000,
    collection_step_num_env_steps: int = 10,
    update_per_step: float = 0.1,
    batch_size: int = 32,
    num_training_envs: int = 10,
    num_test_envs: int = 10,
    persistence_base_dir: str = "log",
    render: float = 0.0,
    device: str | None = None,
    frames_stack: int = 4,
    resume_path: str | None = None,
    resume_id: str | None = None,
    logger_type: str = "tensorboard",
    wandb_project: str = "atari.benchmark",
    watch: bool = False,
    save_buffer_name: str | None = None,
) -> None:
    # Set defaults for mutable arguments
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get all local variables as config (excluding internal/temporary ones)
    params_log_info = locals()
    log.info(f"Starting training with config:\n{params_log_info}")

    env, training_envs, test_envs = make_atari_env(
        task,
        seed,
        num_training_envs,
        num_test_envs,
        scale=scale_obs,
        frame_stack=frames_stack,
    )
    state_shape = env.observation_space.shape or env.observation_space.n  # type: ignore
    action_shape = env.action_space.shape or env.action_space.n  # type: ignore
    # should be N_FRAMES x H x W
    log.info(f"Observations shape: {state_shape}")
    log.info(f"Actions shape: {action_shape}")

    # seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # define model
    c, h, w = state_shape
    net = RainbowNet(
        c=c,
        h=h,
        w=w,
        action_shape=action_shape,
        num_atoms=num_atoms,
        noisy_std=noisy_std,
        is_dueling=not no_dueling,
        is_noisy=not no_noisy,
    )

    # define policy and algorithm
    policy = C51Policy(
        model=net,
        action_space=env.action_space,
        num_atoms=num_atoms,
        v_min=v_min,
        v_max=v_max,
        eps_training=eps_train,
        eps_inference=eps_test,
    )
    optim = AdamOptimizerFactory(lr=lr)
    algorithm: C51 = RainbowDQN(
        policy=policy,
        optim=optim,
        gamma=gamma,
        n_step_return_horizon=n_step,
        target_update_freq=target_update_freq,
    ).to(device)

    # load a previous policy
    if resume_path:
        algorithm.load_state_dict(torch.load(resume_path, map_location=device))
        log.info(f"Loaded agent from: {resume_path}")

    # replay buffer: `save_last_obs` and `stack_num` can be removed together
    # when you have enough RAM
    buffer: VectorReplayBuffer | PrioritizedVectorReplayBuffer
    if no_priority:
        buffer = VectorReplayBuffer(
            buffer_size,
            buffer_num=len(training_envs),
            ignore_obs_next=True,
            save_only_last_obs=True,
            stack_num=frames_stack,
        )
    else:
        buffer = PrioritizedVectorReplayBuffer(
            buffer_size,
            buffer_num=len(training_envs),
            ignore_obs_next=True,
            save_only_last_obs=True,
            stack_num=frames_stack,
            alpha=alpha,
            beta=beta,
            weight_norm=not no_weight_norm,
        )

    # collector
    training_collector = Collector[CollectStats](
        algorithm, training_envs, buffer, exploration_noise=True
    )
    test_collector = Collector[CollectStats](algorithm, test_envs, exploration_noise=True)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    algo_name = "rainbow"
    log_name = os.path.join(task, algo_name, str(seed), now)
    log_path = os.path.join(persistence_base_dir, log_name)

    # logger
    logger_factory = LoggerFactoryDefault()
    if logger_type == "wandb":
        logger_factory.logger_type = "wandb"
        logger_factory.wandb_project = wandb_project
    else:
        logger_factory.logger_type = "tensorboard"

    logger = logger_factory.create_logger(
        log_dir=log_path,
        experiment_name=log_name,
        run_id=resume_id,
        config_dict=params_log_info,
    )

    def save_best_fn(policy: Algorithm) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards: float) -> bool:
        if env.spec.reward_threshold:  # type: ignore
            return mean_rewards >= env.spec.reward_threshold  # type: ignore
        if "Pong" in task:
            return mean_rewards >= 20
        return False

    def train_fn(epoch: int, env_step: int) -> None:
        # nature DQN setting, linear decay in the first 1M steps
        if env_step <= 1e6:
            eps = eps_train - env_step / 1e6 * (eps_train - eps_train_final)
        else:
            eps = eps_train_final
        policy.set_eps_training(eps)
        if env_step % 1000 == 0:
            logger.write("train/env_step", env_step, {"train/eps": eps})
        if not no_priority:
            if env_step <= beta_anneal_step:
                beta_value = beta - env_step / beta_anneal_step * (beta - beta_final)
            else:
                beta_value = beta_final
            buffer.set_beta(beta_value)
            if env_step % 1000 == 0:
                logger.write("train/env_step", env_step, {"train/beta": beta_value})

    def watch_fn() -> None:
        log.info("Setup test envs ...")
        test_envs.seed(seed)
        if save_buffer_name:
            log.info(f"Generate buffer with size {buffer_size}")
            buffer = PrioritizedVectorReplayBuffer(
                buffer_size,
                buffer_num=len(test_envs),
                ignore_obs_next=True,
                save_only_last_obs=True,
                stack_num=frames_stack,
                alpha=alpha,
                beta=beta,
            )
            collector = Collector[CollectStats](
                algorithm, test_envs, buffer, exploration_noise=True
            )
            result = collector.collect(n_step=buffer_size, reset_before_collect=True)
            log.info(f"Save buffer into {save_buffer_name}")
            # Unfortunately, pickle will cause oom with 1M buffer size
            buffer.save_hdf5(save_buffer_name)
        else:
            log.info("Testing agent ...")
            test_collector.reset()
            result = test_collector.collect(n_episode=num_test_envs, render=render)
        result.pprint_asdict()

    if watch:
        watch_fn()
        sys.exit(0)

    # test training_collector and start filling replay buffer
    training_collector.reset()
    training_collector.collect(n_step=batch_size * num_training_envs)

    # train
    result = algorithm.run_training(
        OffPolicyTrainerParams(
            training_collector=training_collector,
            test_collector=test_collector,
            max_epochs=epoch,
            epoch_num_steps=epoch_num_steps,
            collection_step_num_env_steps=collection_step_num_env_steps,
            test_step_num_episodes=num_test_envs,
            batch_size=batch_size,
            training_fn=train_fn,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            update_step_num_gradient_steps_per_sample=update_per_step,
            test_in_training=False,
        )
    )

    pprint.pprint(result)
    watch_fn()


if __name__ == "__main__":
    result = logging.run_cli(main, level=logging.INFO)
