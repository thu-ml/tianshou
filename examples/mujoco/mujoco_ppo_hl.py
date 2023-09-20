#!/usr/bin/env python3

import datetime
import os
from collections.abc import Sequence
from typing import Literal

from jsonargparse import CLI
from torch.distributions import Independent, Normal

from examples.mujoco.mujoco_env import MujocoEnvFactory
from tianshou.highlevel.agent import PPOAgentFactory, PPOConfig
from tianshou.highlevel.config import RLSamplingConfig
from tianshou.highlevel.experiment import (
    RLExperiment,
    RLExperimentConfig,
)
from tianshou.highlevel.logger import DefaultLoggerFactory
from tianshou.highlevel.module import (
    ContinuousActorProbFactory,
    ContinuousNetCriticFactory,
)
from tianshou.highlevel.optim import AdamOptimizerFactory, LinearLRSchedulerFactory


def main(
    experiment_config: RLExperimentConfig,
    task: str = "Ant-v4",
    buffer_size: int = 4096,
    hidden_sizes: Sequence[int] = (64, 64),
    lr: float = 3e-4,
    gamma: float = 0.99,
    epoch: int = 100,
    step_per_epoch: int = 30000,
    step_per_collect: int = 2048,
    repeat_per_collect: int = 10,
    batch_size: int = 64,
    training_num: int = 64,
    test_num: int = 10,
    rew_norm: bool = True,
    vf_coef: float = 0.25,
    ent_coef: float = 0.0,
    gae_lambda: float = 0.95,
    bound_action_method: Literal["clip", "tanh"] | None = "clip",
    lr_decay: bool = True,
    max_grad_norm: float = 0.5,
    eps_clip: float = 0.2,
    dual_clip: float | None = None,
    value_clip: bool = False,
    norm_adv: bool = False,
    recompute_adv: bool = True,
):
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    log_name = os.path.join(task, "ppo", str(experiment_config.seed), now)
    logger_factory = DefaultLoggerFactory()

    sampling_config = RLSamplingConfig(
        num_epochs=epoch,
        step_per_epoch=step_per_epoch,
        batch_size=batch_size,
        num_train_envs=training_num,
        num_test_envs=test_num,
        buffer_size=buffer_size,
        step_per_collect=step_per_collect,
        repeat_per_collect=repeat_per_collect,
    )

    env_factory = MujocoEnvFactory(task, experiment_config.seed, sampling_config)

    def dist_fn(*logits):
        return Independent(Normal(*logits), 1)

    ppo_config = PPOConfig(
        gamma=gamma,
        gae_lambda=gae_lambda,
        action_bound_method=bound_action_method,
        rew_norm=rew_norm,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        value_clip=value_clip,
        norm_adv=norm_adv,
        eps_clip=eps_clip,
        dual_clip=dual_clip,
        recompute_adv=recompute_adv,
    )
    actor_factory = ContinuousActorProbFactory(hidden_sizes)
    critic_factory = ContinuousNetCriticFactory(hidden_sizes)
    optim_factory = AdamOptimizerFactory()
    lr_scheduler_factory = LinearLRSchedulerFactory(sampling_config) if lr_decay else None
    agent_factory = PPOAgentFactory(
        ppo_config,
        sampling_config,
        actor_factory,
        critic_factory,
        optim_factory,
        dist_fn,
        lr,
        lr_scheduler_factory,
    )

    experiment = RLExperiment(experiment_config, env_factory, logger_factory, agent_factory)

    experiment.run(log_name)


if __name__ == "__main__":
    CLI(main)
