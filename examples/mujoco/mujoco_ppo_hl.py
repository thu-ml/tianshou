#!/usr/bin/env python3

import os
from collections.abc import Sequence
from typing import Literal

import torch

from examples.mujoco.mujoco_env import MujocoEnvFactory
from tianshou.highlevel.config import SamplingConfig
from tianshou.highlevel.experiment import (
    ExperimentConfig,
    PPOExperimentBuilder,
)
from tianshou.highlevel.params.dist_fn import (
    DistributionFunctionFactoryIndependentGaussians,
)
from tianshou.highlevel.params.lr_scheduler import LRSchedulerFactoryLinear
from tianshou.highlevel.params.policy_params import PPOParams
from tianshou.utils import logging
from tianshou.utils.logging import datetime_tag


def main(
    experiment_config: ExperimentConfig,
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
    log_name = os.path.join(task, "ppo", str(experiment_config.seed), datetime_tag())

    sampling_config = SamplingConfig(
        num_epochs=epoch,
        step_per_epoch=step_per_epoch,
        batch_size=batch_size,
        num_train_envs=training_num,
        num_test_envs=test_num,
        buffer_size=buffer_size,
        step_per_collect=step_per_collect,
        repeat_per_collect=repeat_per_collect,
    )

    env_factory = MujocoEnvFactory(task, experiment_config.seed, obs_norm=True)

    experiment = (
        PPOExperimentBuilder(env_factory, experiment_config, sampling_config)
        .with_ppo_params(
            PPOParams(
                discount_factor=gamma,
                gae_lambda=gae_lambda,
                action_bound_method=bound_action_method,
                reward_normalization=rew_norm,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                value_clip=value_clip,
                advantage_normalization=norm_adv,
                eps_clip=eps_clip,
                dual_clip=dual_clip,
                recompute_advantage=recompute_adv,
                lr=lr,
                lr_scheduler_factory=LRSchedulerFactoryLinear(sampling_config)
                if lr_decay
                else None,
                dist_fn=DistributionFunctionFactoryIndependentGaussians(),
            ),
        )
        .with_actor_factory_default(hidden_sizes, torch.nn.Tanh, continuous_unbounded=True)
        .with_critic_factory_default(hidden_sizes, torch.nn.Tanh)
        .build()
    )
    experiment.run(log_name)


if __name__ == "__main__":
    logging.run_cli(main)
