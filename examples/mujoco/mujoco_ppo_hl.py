#!/usr/bin/env python3

import os
from collections.abc import Sequence
from typing import Literal

import torch
from sensai.util import logging
from sensai.util.logging import datetime_tag

from examples.mujoco.mujoco_env import MujocoEnvFactory
from tianshou.highlevel.config import OnPolicyTrainingConfig
from tianshou.highlevel.experiment import (
    ExperimentConfig,
    PPOExperimentBuilder,
)
from tianshou.highlevel.params.algorithm_params import PPOParams
from tianshou.highlevel.params.lr_scheduler import LRSchedulerFactoryFactoryLinear


def main(
    experiment_config: ExperimentConfig,
    task: str = "Ant-v4",
    buffer_size: int = 4096,
    hidden_sizes: Sequence[int] = (64, 64),
    lr: float = 3e-4,
    gamma: float = 0.99,
    epoch: int = 100,
    epoch_num_steps: int = 30000,
    collection_step_num_env_steps: int = 2048,
    update_step_num_repetitions: int = 10,
    batch_size: int = 64,
    num_train_envs: int = 10,
    test_num: int = 10,
    return_scaling: bool = True,
    vf_coef: float = 0.25,
    ent_coef: float = 0.0,
    gae_lambda: float = 0.95,
    bound_action_method: Literal["clip", "tanh"] | None = "clip",
    lr_decay: bool = True,
    max_grad_norm: float = 0.5,
    eps_clip: float = 0.2,
    dual_clip: float | None = None,
    value_clip: bool = False,
    advantage_normalization: bool = False,
    recompute_adv: bool = True,
) -> None:
    log_name = os.path.join(task, "ppo", str(experiment_config.seed), datetime_tag())

    training_config = OnPolicyTrainingConfig(
        max_epochs=epoch,
        epoch_num_steps=epoch_num_steps,
        batch_size=batch_size,
        num_train_envs=num_train_envs,
        num_test_envs=test_num,
        buffer_size=buffer_size,
        collection_step_num_env_steps=collection_step_num_env_steps,
        update_step_num_repetitions=update_step_num_repetitions,
    )

    env_factory = MujocoEnvFactory(
        task,
        train_seed=training_config.train_seed,
        test_seed=training_config.test_seed,
        obs_norm=True,
    )

    experiment = (
        PPOExperimentBuilder(env_factory, experiment_config, training_config)
        .with_ppo_params(
            PPOParams(
                gamma=gamma,
                gae_lambda=gae_lambda,
                action_bound_method=bound_action_method,
                return_scaling=return_scaling,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                value_clip=value_clip,
                advantage_normalization=advantage_normalization,
                eps_clip=eps_clip,
                dual_clip=dual_clip,
                recompute_advantage=recompute_adv,
                lr=lr,
                lr_scheduler=LRSchedulerFactoryFactoryLinear(training_config) if lr_decay else None,
            ),
        )
        .with_actor_factory_default(hidden_sizes, torch.nn.Tanh, continuous_unbounded=True)
        .with_critic_factory_default(hidden_sizes, torch.nn.Tanh)
        .build()
    )
    experiment.run(run_name=log_name)


if __name__ == "__main__":
    logging.run_cli(main)
