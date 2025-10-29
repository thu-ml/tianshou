#!/usr/bin/env python3

import datetime
import os
import pprint
from typing import Literal

import numpy as np
import torch
from mujoco_env import make_mujoco_env
from sensai.util import logging
from torch import nn
from torch.distributions import Distribution, Independent, Normal

from tianshou.algorithm import A2C
from tianshou.algorithm.algorithm_base import Algorithm
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.optim import LRSchedulerFactoryLinear, RMSpropOptimizerFactory
from tianshou.data import Collector, CollectStats, ReplayBuffer, VectorReplayBuffer
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.trainer import OnPolicyTrainerParams
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ContinuousActorProbabilistic, ContinuousCritic

log = logging.getLogger(__name__)


def main(
    task: str = "Ant-v4",
    persistence_base_dir: str = "log",
    seed: int = 0,
    buffer_size: int = 4096,
    hidden_sizes: list | None = None,
    lr: float = 7e-4,
    gamma: float = 0.99,
    epoch: int = 100,
    epoch_num_steps: int = 30000,
    collection_step_num_env_steps: int = 80,
    update_step_num_repetitions: int = 1,
    batch_size: int | None = None,
    num_training_envs: int = 16,
    num_test_envs: int = 10,
    return_scaling: bool = True,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    gae_lambda: float = 0.95,
    action_bound_method: Literal["clip", "tanh"] | None = "clip",
    lr_decay: bool = True,
    max_grad_norm: float = 0.5,
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
        hidden_sizes = [64, 64]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get all local variables as config (excluding internal/temporary ones)
    params_log_info = locals()
    log.info(f"Starting training with config:\n{params_log_info}")

    env, training_envs, test_envs = make_mujoco_env(
        task,
        seed,
        num_training_envs,
        num_test_envs,
        obs_norm=True,
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
    net_a = Net(
        state_shape=state_shape,
        hidden_sizes=hidden_sizes,
        activation=nn.Tanh,
    )
    actor = ContinuousActorProbabilistic(
        preprocess_net=net_a,
        action_shape=action_shape,
        unbounded=True,
    ).to(device)
    net_c = Net(
        state_shape=state_shape,
        hidden_sizes=hidden_sizes,
        activation=nn.Tanh,
    )
    critic = ContinuousCritic(preprocess_net=net_c).to(device)
    actor_critic = ActorCritic(actor, critic)

    torch.nn.init.constant_(actor.sigma_param, -0.5)
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
    # do last policy layer scaling, this will make initial actions have (close to)
    # 0 mean and std, and will help boost performances,
    # see https://arxiv.org/abs/2006.05990, Fig.24 for details
    for m in actor.mu.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)

    optim = RMSpropOptimizerFactory(
        lr=lr,
        eps=1e-5,
        alpha=0.99,
    )

    if lr_decay:
        optim.with_lr_scheduler_factory(
            LRSchedulerFactoryLinear(
                max_epochs=epoch,
                epoch_num_steps=epoch_num_steps,
                collection_step_num_env_steps=collection_step_num_env_steps,
            )
        )

    def dist(loc_scale: tuple[torch.Tensor, torch.Tensor]) -> Distribution:
        loc, scale = loc_scale
        return Independent(Normal(loc, scale), 1)

    policy = ProbabilisticActorPolicy(
        actor=actor,
        dist_fn=dist,
        action_scaling=True,
        action_bound_method=action_bound_method,
        action_space=env.action_space,
    )
    algorithm: A2C = A2C(
        policy=policy,
        critic=critic,
        optim=optim,
        gamma=gamma,
        gae_lambda=gae_lambda,
        max_grad_norm=max_grad_norm,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        return_scaling=return_scaling,
    )

    # load a previous policy
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        algorithm.load_state_dict(ckpt["model"])
        training_envs.set_obs_rms(ckpt["obs_rms"])
        test_envs.set_obs_rms(ckpt["obs_rms"])
        print("Loaded agent from: ", resume_path)

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

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    algo_name = "a2c"
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
        state = {"model": policy.state_dict(), "obs_rms": training_envs.get_obs_rms()}
        torch.save(state, os.path.join(log_path, "policy.pth"))

    if not watch:
        # train
        result = algorithm.run_training(
            OnPolicyTrainerParams(
                training_collector=training_collector,
                test_collector=test_collector,
                max_epochs=epoch,
                epoch_num_steps=epoch_num_steps,
                update_step_num_repetitions=update_step_num_repetitions,
                test_step_num_episodes=num_test_envs,
                batch_size=batch_size,
                collection_step_num_env_steps=collection_step_num_env_steps,
                save_best_fn=save_best_fn,
                logger=logger,
                test_in_training=False,
            )
        )
        pprint.pprint(result)

    # Let's watch its performance!
    test_envs.seed(seed)
    test_collector.reset()
    collector_stats = test_collector.collect(n_episode=num_test_envs, render=render)
    print(collector_stats)


if __name__ == "__main__":
    result = logging.run_cli(main, level=logging.INFO)
