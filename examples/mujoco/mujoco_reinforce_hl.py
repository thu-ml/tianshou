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
    ReinforceExperimentBuilder,
)
from tianshou.highlevel.params.algorithm_params import ReinforceParams
from tianshou.highlevel.params.lr_scheduler import LRSchedulerFactoryFactoryLinear


def main(
    experiment_config: ExperimentConfig,
    task: str = "Ant-v4",
    buffer_size: int = 4096,
    hidden_sizes: Sequence[int] = (64, 64),
    lr: float = 1e-3,
    gamma: float = 0.99,
    epoch: int = 100,
    epoch_num_steps: int = 30000,
    collection_step_num_env_steps: int = 2048,
    update_step_num_repetitions: int = 1,
    batch_size: int | None = None,
    num_train_envs: int = 10,
    num_test_envs: int = 10,
    return_scaling: bool = True,
    action_bound_method: Literal["clip", "tanh"] = "tanh",
    lr_decay: bool = True,
) -> None:
    log_name = os.path.join(task, "reinforce", str(experiment_config.seed), datetime_tag())

    training_config = OnPolicyTrainingConfig(
        max_epochs=epoch,
        epoch_num_steps=epoch_num_steps,
        batch_size=batch_size,
        num_train_envs=num_train_envs,
        num_test_envs=num_test_envs,
        buffer_size=buffer_size,
        collection_step_num_env_steps=collection_step_num_env_steps,
        update_step_num_repetitions=update_step_num_repetitions,
    )

    env_factory = MujocoEnvFactory(task, obs_norm=True)

    experiment = (
        ReinforceExperimentBuilder(env_factory, experiment_config, training_config)
        .with_reinforce_params(
            ReinforceParams(
                gamma=gamma,
                action_bound_method=action_bound_method,
                return_standardization=return_scaling,
                lr=lr,
                lr_scheduler=LRSchedulerFactoryFactoryLinear(training_config) if lr_decay else None,
            ),
        )
        .with_actor_factory_default(hidden_sizes, torch.nn.Tanh, continuous_unbounded=True)
        .build()
    )
    experiment.run(run_name=log_name)


if __name__ == "__main__":
    logging.run_cli(main)
