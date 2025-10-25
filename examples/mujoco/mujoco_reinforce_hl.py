#!/usr/bin/env python3

import os
from typing import Literal

import torch
from sensai.util import logging

from examples.mujoco.mujoco_env import MujocoEnvFactory
from tianshou.highlevel.config import OnPolicyTrainingConfig
from tianshou.highlevel.experiment import (
    ExperimentConfig,
    ReinforceExperimentBuilder,
)
from tianshou.highlevel.params.algorithm_params import ReinforceParams
from tianshou.highlevel.params.lr_scheduler import LRSchedulerFactoryFactoryLinear


def main(
    task: str = "Ant-v4",
    persistence_base_dir: str = "log",
    num_experiments: int = 1,
    experiment_launcher: Literal["sequential", "joblib"] = "sequential",
) -> None:
    """
    Train an agent using REINFORCE on a specified MuJoCo task, potentially running multiple experiments with different seeds
    and evaluating the results using rliable.

    :param task: the MuJoCo task to train on.
    :param persistence_base_dir: the base directory for logging and saving experiment data,
        the task name will be appended to it.
    :param num_experiments: the number of experiments to run. The experiments differ exclusively in the seeds.
    :param experiment_launcher: the type of experiment launcher to use, only has an effect if `num_experiments>1`.
        You can use "joblib" for parallel execution of whole experiments.
    """
    persistence_base_dir = os.path.abspath(os.path.join(persistence_base_dir, task))
    experiment_config = ExperimentConfig(persistence_base_dir=persistence_base_dir, watch=False)

    training_config = OnPolicyTrainingConfig(
        max_epochs=100,
        epoch_num_steps=30000,
        batch_size=None,
        num_training_envs=64,
        num_test_envs=10,
        buffer_size=4096,
        collection_step_num_env_steps=2048,
        update_step_num_repetitions=1,
    )

    env_factory = MujocoEnvFactory(task, obs_norm=True)

    hidden_sizes = (64, 64)
    experiment_builder = (
        ReinforceExperimentBuilder(env_factory, experiment_config, training_config)
        .with_reinforce_params(
            ReinforceParams(
                gamma=0.99,
                action_bound_method="tanh",
                return_standardization=True,
                lr=1e-3,
                lr_scheduler=LRSchedulerFactoryFactoryLinear(training_config),
            ),
        )
        .with_actor_factory_default(hidden_sizes, torch.nn.Tanh, continuous_unbounded=True)
    )

    experiment_builder.build_and_run(num_experiments=num_experiments, launcher=experiment_launcher)


if __name__ == "__main__":
    result = logging.run_cli(main, level=logging.INFO)
