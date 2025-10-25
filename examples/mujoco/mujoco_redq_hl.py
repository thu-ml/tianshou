#!/usr/bin/env python3

import os
from typing import Literal

from sensai.util import logging

from examples.mujoco.mujoco_env import MujocoEnvFactory
from tianshou.highlevel.config import OffPolicyTrainingConfig
from tianshou.highlevel.experiment import (
    ExperimentConfig,
    REDQExperimentBuilder,
)
from tianshou.highlevel.params.algorithm_params import REDQParams


def main(
    task: str = "Ant-v4",
    persistence_base_dir: str = "log",
    num_experiments: int = 1,
    experiment_launcher: Literal["sequential", "joblib"] = "sequential",
    max_epochs: int = 200,
    epoch_num_steps: int = 5000,
) -> None:
    """
    Train an agent using REDQ on a specified MuJoCo task, potentially running multiple experiments with different seeds
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

    training_config = OffPolicyTrainingConfig(
        max_epochs=max_epochs,
        epoch_num_steps=epoch_num_steps,
        num_training_envs=1,
        num_test_envs=10,
        buffer_size=1000000,
        batch_size=256,
        collection_step_num_env_steps=1,
        update_step_num_gradient_steps_per_sample=20,
        start_timesteps=10000,
        start_timesteps_random=True,
    )

    env_factory = MujocoEnvFactory(task, obs_norm=False)

    hidden_sizes = (256, 256)
    experiment_builder = (
        REDQExperimentBuilder(env_factory, experiment_config, training_config)
        .with_redq_params(
            REDQParams(
                actor_lr=1e-3,
                critic_lr=1e-3,
                gamma=0.99,
                tau=0.005,
                alpha=0.2,
                n_step_return_horizon=1,
                target_mode="min",
                subset_size=2,
                ensemble_size=10,
            ),
        )
        .with_actor_factory_default(hidden_sizes)
        .with_critic_ensemble_factory_default(hidden_sizes)
    )

    experiment_builder.build_and_run(num_experiments=num_experiments, launcher=experiment_launcher)


if __name__ == "__main__":
    result = logging.run_cli(main, level=logging.INFO)
