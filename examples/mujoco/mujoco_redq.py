#!/usr/bin/env python3

import datetime
import os
import pprint
from typing import Literal

import numpy as np
import torch
from mujoco_env import make_mujoco_env
from sensai.util import logging

from tianshou.algorithm import REDQ
from tianshou.algorithm.algorithm_base import Algorithm
from tianshou.algorithm.modelfree.redq import REDQPolicy
from tianshou.algorithm.modelfree.sac import AutoAlpha
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Collector, CollectStats, ReplayBuffer, VectorReplayBuffer
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.utils.net.common import EnsembleLinear, Net
from tianshou.utils.net.continuous import ContinuousActorProbabilistic, ContinuousCritic

log = logging.getLogger(__name__)


def main(
    task: str = "Ant-v4",
    persistence_base_dir: str = "log",
    seed: int = 0,
    buffer_size: int = 1000000,
    hidden_sizes: list | None = None,
    ensemble_size: int = 10,
    subset_size: int = 2,
    actor_lr: float = 1e-3,
    critic_lr: float = 1e-3,
    gamma: float = 0.99,
    tau: float = 0.005,
    alpha: float = 0.2,
    auto_alpha: bool = False,
    alpha_lr: float = 3e-4,
    start_timesteps: int = 10000,
    epoch: int = 200,
    epoch_num_steps: int = 5000,
    collection_step_num_env_steps: int = 1,
    update_per_step: int = 20,
    n_step: int = 1,
    batch_size: int = 256,
    target_mode: Literal["min", "mean"] = "min",
    num_train_envs: int = 1,
    num_test_envs: int = 10,
    render: float = 0.0,
    device: str | None = None,
    resume_path: str | None = None,
    resume_id: str | None = None,
    logger_type: str = "tensorboard",
    wandb_project: str = "mujoco.benchmark",
    watch: bool = False,
) -> None:
    # Set defaults for mutable arguments
    if hidden_sizes is None:
        hidden_sizes = [256, 256]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get all local variables as config
    params_log_info = locals()
    log.info(f"Starting training with config:\n{params_log_info}")

    env, train_envs, test_envs = make_mujoco_env(
        task,
        seed,
        num_train_envs,
        num_test_envs,
        obs_norm=False,
    )
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]
    log.info(f"Observations shape: {state_shape}")
    log.info(f"Actions shape: {action_shape}")
    log.info(f"Action range: {np.min(env.action_space.low)}, {np.max(env.action_space.high)}")
    # seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    # model
    net_a = Net(state_shape=state_shape, hidden_sizes=hidden_sizes)
    actor = ContinuousActorProbabilistic(
        preprocess_net=net_a,
        action_shape=action_shape,
        unbounded=True,
        conditioned_sigma=True,
    ).to(device)
    actor_optim = AdamOptimizerFactory(lr=actor_lr)

    def linear(x: int, y: int) -> EnsembleLinear:
        return EnsembleLinear(ensemble_size, x, y)

    net_c = Net(
        state_shape=state_shape,
        action_shape=action_shape,
        hidden_sizes=hidden_sizes,
        concat=True,
        linear_layer=linear,
    )
    critics = ContinuousCritic(
        preprocess_net=net_c,
        linear_layer=linear,
        flatten_input=False,
    ).to(device)
    critics_optim = AdamOptimizerFactory(lr=critic_lr)

    if auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = 0.0
        alpha_optim = AdamOptimizerFactory(lr=alpha_lr)
        alpha = AutoAlpha(target_entropy, log_alpha, alpha_optim).to(device)  # type: ignore

    policy = REDQPolicy(
        actor=actor,
        action_space=env.action_space,
    )
    algorithm: REDQ = REDQ(
        policy=policy,
        policy_optim=actor_optim,
        critic=critics,
        critic_optim=critics_optim,
        ensemble_size=ensemble_size,
        subset_size=subset_size,
        tau=tau,
        gamma=gamma,
        alpha=alpha,
        n_step_return_horizon=n_step,
        actor_delay=update_per_step,
        target_mode=target_mode,
    )

    # load a previous policy
    if resume_path:
        algorithm.load_state_dict(torch.load(resume_path, map_location=device))
        log.info(f"Loaded agent from: {resume_path}")

    # collector
    buffer: VectorReplayBuffer | ReplayBuffer
    if num_train_envs > 1:
        buffer = VectorReplayBuffer(buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(buffer_size)
    train_collector = Collector[CollectStats](algorithm, train_envs, buffer, exploration_noise=True)
    test_collector = Collector[CollectStats](algorithm, test_envs)
    train_collector.reset()
    train_collector.collect(n_step=start_timesteps, random=True)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    algo_name = "redq"
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

    if not watch:
        # train
        result = algorithm.run_training(
            OffPolicyTrainerParams(
                train_collector=train_collector,
                test_collector=test_collector,
                max_epochs=epoch,
                epoch_num_steps=epoch_num_steps,
                collection_step_num_env_steps=collection_step_num_env_steps,
                test_step_num_episodes=num_test_envs,
                batch_size=batch_size,
                save_best_fn=save_best_fn,
                logger=logger,
                update_step_num_gradient_steps_per_sample=update_per_step,
                test_in_train=False,
            )
        )
        pprint.pprint(result)

    # Let's watch its performance!
    test_envs.seed(seed)
    test_collector.reset()
    collector_stats = test_collector.collect(n_episode=num_test_envs, render=render)
    log.info(f"Collector stats: {collector_stats}")


if __name__ == "__main__":
    result = logging.run_cli(main, level=logging.INFO)
