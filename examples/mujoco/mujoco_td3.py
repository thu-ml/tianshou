#!/usr/bin/env python3

import datetime
import os
import pprint

import numpy as np
import torch
from mujoco_env import make_mujoco_env
from sensai.util import logging

from tianshou.algorithm import TD3
from tianshou.algorithm.algorithm_base import Algorithm
from tianshou.algorithm.modelfree.ddpg import ContinuousDeterministicPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Collector, CollectStats, ReplayBuffer, VectorReplayBuffer
from tianshou.exploration import GaussianNoise
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ContinuousActorDeterministic, ContinuousCritic

log = logging.getLogger(__name__)


def main(
    task: str = "Ant-v4",
    persistence_base_dir: str = "log",
    seed: int = 0,
    hidden_sizes: list | None = None,
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    gamma: float = 0.99,
    tau: float = 0.005,
    exploration_noise: float = 0.1,
    policy_noise: float = 0.2,
    noise_clip: float = 0.5,
    update_actor_freq: int = 2,
    start_timesteps: int = 25000,
    epoch: int = 200,
    epoch_num_steps: int = 5000,
    collection_step_num_env_steps: int = 1,
    update_per_step: int = 1,
    n_step: int = 1,
    batch_size: int = 256,
    buffer_size: int = 1000000,
    num_training_envs: int = 1,
    num_test_envs: int = 10,
    device: str | None = None,
    resume_path: str | None = None,
    resume_id: str | None = None,
    logger_type: str = "tensorboard",
    wandb_project: str = "mujoco.benchmark",
    watch: bool = False,
    render: float = 0.0,
) -> None:
    # Set defaults for mutable arguments
    if hidden_sizes is None:
        hidden_sizes = [256, 256]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # Get all local variables as config
    params_log_info = locals()
    log.info(f"Starting training with config:\n{params_log_info}")

    env, training_envs, test_envs = make_mujoco_env(
        task,
        seed,
        num_training_envs,
        num_test_envs,
        obs_norm=False,
    )
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]
    exploration_noise = exploration_noise * max_action
    policy_noise = policy_noise * max_action
    log.info(f"Observations shape: {state_shape}")
    log.info(f"Actions shape: {action_shape}")
    log.info(f"Action range: {np.min(env.action_space.low)}, {np.max(env.action_space.high)}")
    # seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    # model
    net_a = Net(state_shape=state_shape, hidden_sizes=hidden_sizes)
    actor = ContinuousActorDeterministic(
        preprocess_net=net_a, action_shape=action_shape, max_action=max_action
    ).to(device)
    actor_optim = AdamOptimizerFactory(lr=actor_lr)
    net_c1 = Net(
        state_shape=state_shape,
        action_shape=action_shape,
        hidden_sizes=hidden_sizes,
        concat=True,
    )
    net_c2 = Net(
        state_shape=state_shape,
        action_shape=action_shape,
        hidden_sizes=hidden_sizes,
        concat=True,
    )
    critic1 = ContinuousCritic(preprocess_net=net_c1).to(device)
    critic1_optim = AdamOptimizerFactory(lr=critic_lr)
    critic2 = ContinuousCritic(preprocess_net=net_c2).to(device)
    critic2_optim = AdamOptimizerFactory(lr=critic_lr)

    policy = ContinuousDeterministicPolicy(
        actor=actor,
        exploration_noise=GaussianNoise(sigma=exploration_noise),
        action_space=env.action_space,
    )
    algorithm: TD3 = TD3(
        policy=policy,
        policy_optim=actor_optim,
        critic=critic1,
        critic_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=tau,
        gamma=gamma,
        policy_noise=policy_noise,
        update_actor_freq=update_actor_freq,
        noise_clip=noise_clip,
        n_step_return_horizon=n_step,
    )

    # load a previous policy
    if resume_path:
        log.info(f"Loaded agent from: {resume_path}")

    # collector
    buffer: VectorReplayBuffer | ReplayBuffer
    if num_training_envs > 1:
        buffer = VectorReplayBuffer(buffer_size, len(training_envs))
    else:
        buffer = ReplayBuffer(buffer_size)
    training_collector = Collector[CollectStats](
        algorithm, training_envs, buffer, exploration_noise=True
    )
    test_collector = Collector[CollectStats](algorithm, test_envs)
    training_collector.reset()
    training_collector.collect(n_step=start_timesteps, random=True)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    algo_name = "td3"
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
                training_collector=training_collector,
                test_collector=test_collector,
                max_epochs=epoch,
                epoch_num_steps=epoch_num_steps,
                collection_step_num_env_steps=collection_step_num_env_steps,
                test_step_num_episodes=num_test_envs,
                batch_size=batch_size,
                save_best_fn=save_best_fn,
                logger=logger,
                update_step_num_gradient_steps_per_sample=update_per_step,
                test_in_training=False,
            )
        )
        pprint.pprint(result)

    # Let's watch its performance!
    test_envs.seed(seed)
    test_collector.reset()
    collector_stats = test_collector.collect(n_episode=num_test_envs, render=render)
    log.info(collector_stats)


if __name__ == "__main__":
    result = logging.run_cli(main, level=logging.INFO)
