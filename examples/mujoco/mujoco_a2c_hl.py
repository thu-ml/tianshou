#!/usr/bin/env python3

import os
from typing import Literal

import torch
from sensai.util import logging

from examples.mujoco.mujoco_env import MujocoEnvFactory
from tianshou.highlevel.config import OnPolicyTrainingConfig
from tianshou.highlevel.experiment import (
    A2CExperimentBuilder,
    ExperimentConfig,
)
from tianshou.highlevel.params.algorithm_params import A2CParams
from tianshou.highlevel.params.lr_scheduler import LRSchedulerFactoryFactoryLinear
from tianshou.highlevel.params.optim import OptimizerFactoryFactoryRMSprop


def main(
    task: str = "Ant-v4",
    persistence_base_dir: str = "log",
    num_experiments: int = 1,
    experiment_launcher: Literal["sequential", "joblib"] = "sequential",
    max_epochs: int = 100,
    epoch_num_steps: int = 30000,
) -> None:
    """
    Train an agent using A2C on a specified MuJoCo task, potentially running multiple experiments with different seeds
    and evaluating the results using rliable.

    :param task: the MuJoCo task to train on.
    :param persistence_base_dir: the base directory for logging and saving experiment data,
        the task name will be appended to it.
    :param num_experiments: the number of experiments to run. The experiments differ exclusively in the seeds.
    :param experiment_launcher: the type of experiment launcher to use, only has an effect if `num_experiments>1`.
        You can use "joblib" for parallel execution of whole experiments.
    :param max_epochs: the maximum number of training epochs.
    :param epoch_num_steps: the number of environment steps per epoch.
    """
    persistence_base_dir = os.path.abspath(os.path.join(persistence_base_dir, task))
    experiment_config = ExperimentConfig(persistence_base_dir=persistence_base_dir, watch=False)

    training_config = OnPolicyTrainingConfig(
        max_epochs=max_epochs,
        epoch_num_steps=epoch_num_steps,
        batch_size=None,
        num_training_envs=16,
        num_test_envs=10,
        buffer_size=4096,
        collection_step_num_env_steps=80,
        update_step_num_repetitions=1,
    )

    env_factory = MujocoEnvFactory(task, obs_norm=True)

    hidden_sizes = (64, 64)
    experiment_builder = (
        A2CExperimentBuilder(env_factory, experiment_config, training_config)
        .with_a2c_params(
            A2CParams(
                gamma=0.99,
                gae_lambda=0.95,
                action_bound_method="clip",
                return_scaling=True,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                optim=OptimizerFactoryFactoryRMSprop(eps=1e-5, alpha=0.99),
                lr=7e-4,
                lr_scheduler=LRSchedulerFactoryFactoryLinear(training_config),
            ),
        )
        .with_actor_factory_default(hidden_sizes, torch.nn.Tanh, continuous_unbounded=True)
        .with_critic_factory_default(hidden_sizes, torch.nn.Tanh)
    )

    experiment_builder.build_and_run(num_experiments=num_experiments, launcher=experiment_launcher)


if __name__ == "__main__":
    result = logging.run_cli(main, level=logging.INFO)
