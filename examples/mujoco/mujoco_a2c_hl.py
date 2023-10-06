#!/usr/bin/env python3

import datetime
import os
from collections.abc import Sequence
from typing import Literal

from jsonargparse import CLI

from examples.mujoco.mujoco_env import MujocoEnvFactory
from tianshou.highlevel.config import SamplingConfig
from tianshou.highlevel.experiment import (
    A2CExperimentBuilder,
    ExperimentConfig,
)
from tianshou.highlevel.optim import OptimizerFactoryRMSprop
from tianshou.highlevel.params.lr_scheduler import LRSchedulerFactoryLinear
from tianshou.highlevel.params.policy_params import A2CParams
from tianshou.utils import logging


def main(
    experiment_config: ExperimentConfig,
    task: str = "Ant-v3",
    buffer_size: int = 4096,
    hidden_sizes: Sequence[int] = (64, 64),
    lr: float = 7e-4,
    gamma: float = 0.99,
    epoch: int = 100,
    step_per_epoch: int = 30000,
    step_per_collect: int = 80,
    repeat_per_collect: int = 1,
    batch_size: int = 99999,
    training_num: int = 16,
    test_num: int = 10,
    rew_norm: bool = True,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    gae_lambda: float = 0.95,
    bound_action_method: Literal["clip", "tanh"] = "clip",
    lr_decay: bool = True,
    max_grad_norm: float = 0.5,
):
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    log_name = os.path.join(task, "ppo", str(experiment_config.seed), now)

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

    env_factory = MujocoEnvFactory(task, experiment_config.seed, sampling_config)

    experiment = (
        A2CExperimentBuilder(env_factory, experiment_config, sampling_config)
        .with_a2c_params(
            A2CParams(
                discount_factor=gamma,
                gae_lambda=gae_lambda,
                action_bound_method=bound_action_method,
                reward_normalization=rew_norm,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                lr=lr,
                lr_scheduler_factory=LRSchedulerFactoryLinear(sampling_config)
                if lr_decay
                else None,
            ),
        )
        .with_optim_factory(OptimizerFactoryRMSprop(eps=1e-5, alpha=0.99))
        .with_actor_factory_default(hidden_sizes)
        .with_critic_factory_default(hidden_sizes)
        .build()
    )
    experiment.run(log_name)


if __name__ == "__main__":
    logging.run_main(lambda: CLI(main))
