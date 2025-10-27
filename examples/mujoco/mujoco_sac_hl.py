#!/usr/bin/env python3

import os
from typing import Literal

from sensai.util import logging

from examples.mujoco.mujoco_env import MujocoEnvFactory
from tianshou.highlevel.config import OffPolicyTrainingConfig
from tianshou.highlevel.experiment import (
    ExperimentConfig,
    SACExperimentBuilder,
)
from tianshou.highlevel.params.algorithm_params import SACParams


def main(
    task: str = "Ant-v4",
    persistence_base_dir: str = "log",
    num_experiments: int = 1,
    experiment_launcher: Literal["sequential", "joblib"] = "joblib",
    max_epochs: int = 250,
    epoch_num_steps: int = 5000,
) -> None:
    """
    Train an agent using SAC on a specified MuJoCo task, potentially running multiple experiments with different seeds
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
        update_step_num_gradient_steps_per_sample=1,
        start_timesteps=10000,
        start_timesteps_random=True,
    )

    env_factory = MujocoEnvFactory(task, obs_norm=False)

    hidden_sizes = (256, 256)
    experiment_builder = (
        SACExperimentBuilder(env_factory, experiment_config, training_config)
        .with_sac_params(
            SACParams(
                tau=0.005,
                gamma=0.99,
                alpha=0.2,
                n_step_return_horizon=1,
                actor_lr=1e-3,
                critic1_lr=1e-3,
                critic2_lr=1e-3,
            ),
        )
        .with_actor_factory_default(
            hidden_sizes,
            continuous_unbounded=True,
            continuous_conditioned_sigma=True,
        )
        .with_common_critic_factory_default(hidden_sizes)
    )

    experiment_builder.build_and_run(num_experiments=num_experiments, launcher=experiment_launcher)


if __name__ == "__main__":
    result = logging.run_cli(main, level=logging.INFO)
