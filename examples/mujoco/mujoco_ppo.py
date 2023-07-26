#!/usr/bin/env python3

import argparse
import datetime
import os
import pprint
from collections.abc import Sequence
from typing import Literal, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from jsonargparse import CLI
from torch import nn
from torch.distributions import Independent, Normal
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from mujoco_env import make_mujoco_env
from tianshou.config import (
    BasicExperimentConfig,
    LoggerConfig,
    NNConfig,
    PGConfig,
    PPOConfig,
    RLAgentConfig,
    RLSamplingConfig,
)
from tianshou.config.utils import collect_configs
from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.env import VectorEnvNormObs
from tianshou.policy import BasePolicy, PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_logger_for_run(
    algo_name: str,
    task: str,
    logger_config: LoggerConfig,
    config: dict,
    seed: int,
    resume_id: Optional[Union[str, int]],
) -> Tuple[str, Union[WandbLogger, TensorboardLogger]]:
    """

    :param algo_name:
    :param task:
    :param logger_config:
    :param config: the experiment config
    :param seed:
    :param resume_id: used as run_id by wandb, unused for tensorboard
    :return:
    """
    """Returns the log_path and logger."""
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    log_name = os.path.join(task, algo_name, str(seed), now)
    log_path = os.path.join(logger_config.logdir, log_name)

    logger = get_logger(
        logger_config.logger,
        log_path,
        log_name=log_name,
        run_id=resume_id,
        config=config,
        wandb_project=logger_config.wandb_project,
    )
    return log_path, logger


def get_continuous_env_info(
    env: gym.Env,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], float]:
    if not isinstance(env.action_space, gym.spaces.Box):
        raise ValueError(
            "Only environments with continuous action space are supported here. "
            f"But got env with action space: {env.action_space.__class__}."
        )
    state_shape = env.observation_space.shape or env.observation_space.n
    if not state_shape:
        raise ValueError("Observation space shape is not defined")
    action_shape = env.action_space.shape
    max_action = env.action_space.high[0]
    return state_shape, action_shape, max_action


def resume_from_checkpoint(
    path: str,
    policy: BasePolicy,
    train_envs: VectorEnvNormObs | None = None,
    test_envs: VectorEnvNormObs | None = None,
    device: str | int | torch.device | None = None,
):
    ckpt = torch.load(path, map_location=device)
    policy.load_state_dict(ckpt["model"])
    if train_envs:
        train_envs.set_obs_rms(ckpt["obs_rms"])
    if test_envs:
        test_envs.set_obs_rms(ckpt["obs_rms"])
    print("Loaded agent and obs. running means from: ", path)


def watch_agent(n_episode, policy: BasePolicy, test_collector: Collector, render=0.0):
    policy.eval()
    test_collector.reset()
    result = test_collector.collect(n_episode=n_episode, render=render)
    print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')


def get_train_test_collector(
    buffer_size: int,
    policy: BasePolicy,
    train_envs: VectorEnvNormObs,
    test_envs: VectorEnvNormObs,
):
    if len(train_envs) > 1:
        buffer = VectorReplayBuffer(buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(buffer_size)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    return test_collector, train_collector


TShape = Union[int, Sequence[int]]


def get_actor_critic(
    state_shape: TShape,
    hidden_sizes: Sequence[int],
    action_shape: TShape,
    device: str | int | torch.device = "cpu",
):
    net_a = Net(
        state_shape, hidden_sizes=hidden_sizes, activation=nn.Tanh, device=device
    )
    actor = ActorProb(net_a, action_shape, unbounded=True, device=device).to(device)
    net_c = Net(
        state_shape, hidden_sizes=hidden_sizes, activation=nn.Tanh, device=device
    )
    # TODO: twice device?
    critic = Critic(net_c, device=device).to(device)
    return actor, critic


def get_logger(
    kind: Literal["wandb", "tensorboard"],
    log_path: str,
    log_name="",
    run_id: Optional[Union[str, int]] = None,
    config: Optional[Union[dict, argparse.Namespace]] = None,
    wandb_project: Optional[str] = None,
):
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(config))
    if kind == "wandb":
        logger = WandbLogger(
            save_interval=1,
            name=log_name.replace(os.path.sep, "__"),
            run_id=run_id,
            config=config,
            project=wandb_project,
        )
        logger.load(writer)
    elif kind == "tensorboard":
        logger = TensorboardLogger(writer)
    else:
        raise ValueError(f"Unknown logger: {kind}")
    return logger


def get_lr_scheduler(optim, step_per_epoch: int, step_per_collect: int, epochs: int):
    """Decay learning rate to 0 linearly."""
    max_update_num = np.ceil(step_per_epoch / step_per_collect) * epochs
    lr_scheduler = LambdaLR(optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num)
    return lr_scheduler


def init_and_get_optim(actor: nn.Module, critic: nn.Module, lr: float):
    """Initializes layers of actor and critic.

    :param actor:
    :param critic:
    :param lr:
    :return:
    """
    actor_critic = ActorCritic(actor, critic)
    torch.nn.init.constant_(actor.sigma_param, -0.5)
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
    if hasattr(actor, "mu"):
        # For continuous action spaces with Gaussian policies
        # do last policy layer scaling, this will make initial actions have (close to)
        # 0 mean and std, and will help boost performances,
        # see https://arxiv.org/abs/2006.05990, Fig.24 for details
        for m in actor.mu.modules():
            # TODO: seems like biases are initialized twice for the actor
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.zeros_(m.bias)
                m.weight.data.copy_(0.01 * m.weight.data)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=lr)
    return optim


def main(
    experiment_config: BasicExperimentConfig,
    logger_config: LoggerConfig,
    sampling_config: RLSamplingConfig,
    general_config: RLAgentConfig,
    pg_config: PGConfig,
    ppo_config: PPOConfig,
    nn_config: NNConfig,
):
    """
    Run the PPO test on the provided parameters.

    :param experiment_config: BasicExperimentConfig - not ML or RL specific
    :param logger_config: LoggerConfig
    :param sampling_config: SamplingConfig -
        sampling, epochs, parallelization, buffers, collectors, and batching.
    :param general_config: RLAgentConfig - general RL agent config
    :param pg_config: PGConfig: common to most policy gradient algorithms
    :param ppo_config: PPOConfig - PPO specific config
    :param nn_config: NNConfig - NN-training specific config

    :return: None
    """
    full_config = collect_configs(*locals().values())
    set_seed(experiment_config.seed)

    # create test and train envs, add env info to config
    env, train_envs, test_envs = make_mujoco_env(
        task=experiment_config.task,
        seed=experiment_config.seed,
        num_train_envs=sampling_config.num_train_envs,
        num_test_envs=sampling_config.num_test_envs,
        obs_norm=True,
    )

    # adding env_info to logged config
    state_shape, action_shape, max_action = get_continuous_env_info(env)
    full_config["env_info"] = {
        "state_shape": state_shape,
        "action_shape": action_shape,
        "max_action": max_action,
    }
    log_path, logger = get_logger_for_run(
        "ppo",
        experiment_config.task,
        logger_config,
        full_config,
        experiment_config.seed,
        experiment_config.resume_id,
    )

    # Setup NNs
    actor, critic = get_actor_critic(
        state_shape, nn_config.hidden_sizes, action_shape, experiment_config.device
    )
    optim = init_and_get_optim(actor, critic, nn_config.lr)

    lr_scheduler = None
    if nn_config.lr_decay:
        lr_scheduler = get_lr_scheduler(
            optim,
            sampling_config.step_per_epoch,
            sampling_config.step_per_collect,
            sampling_config.num_epochs,
        )

    # Create policy
    def dist_fn(*logits):
        return Independent(Normal(*logits), 1)

    policy = PPOPolicy(
        # nn-stuff
        actor,
        critic,
        optim,
        dist_fn=dist_fn,
        lr_scheduler=lr_scheduler,
        # env-stuff
        action_space=train_envs.action_space,
        action_scaling=True,
        # general_config
        discount_factor=general_config.gamma,
        gae_lambda=general_config.gae_lambda,
        reward_normalization=general_config.rew_norm,
        action_bound_method=general_config.action_bound_method,
        # pg_config
        max_grad_norm=pg_config.max_grad_norm,
        vf_coef=pg_config.vf_coef,
        ent_coef=pg_config.ent_coef,
        # ppo_config
        eps_clip=ppo_config.eps_clip,
        value_clip=ppo_config.value_clip,
        dual_clip=ppo_config.dual_clip,
        advantage_normalization=ppo_config.norm_adv,
        recompute_advantage=ppo_config.recompute_adv,
    )

    if experiment_config.resume_path:
        resume_from_checkpoint(
            experiment_config.resume_path,
            policy,
            train_envs=train_envs,
            test_envs=test_envs,
            device=experiment_config.device,
        )

    test_collector, train_collector = get_train_test_collector(
        sampling_config.buffer_size, policy, test_envs, train_envs
    )

    # TODO: test num is the number of test envs but used as episode_per_test
    #  here and in watch_agent
    if not experiment_config.watch:
        # RL training
        def save_best_fn(pol: nn.Module):
            state = {"model": pol.state_dict(), "obs_rms": train_envs.get_obs_rms()}
            torch.save(state, os.path.join(log_path, "policy.pth"))

        trainer = OnpolicyTrainer(
            policy,
            train_collector,
            test_collector,
            max_epoch=sampling_config.num_epochs,
            step_per_epoch=sampling_config.step_per_epoch,
            repeat_per_collect=sampling_config.repeat_per_collect,
            episode_per_test=sampling_config.num_test_envs,
            batch_size=sampling_config.batch_size,
            step_per_collect=sampling_config.step_per_collect,
            save_best_fn=save_best_fn,
            logger=logger,
            test_in_train=False,
        )
        result = trainer.run()
        pprint.pprint(result)

    watch_agent(
        sampling_config.num_test_envs,
        policy,
        test_collector,
        render=experiment_config.render,
    )


if __name__ == "__main__":
    CLI(main)
