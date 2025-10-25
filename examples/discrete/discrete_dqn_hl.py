#!/usr/bin/env python3

import os
from typing import Literal

from sensai.util import logging

from tianshou.highlevel.config import OffPolicyTrainingConfig
from tianshou.highlevel.env import (
    EnvFactoryRegistered,
    VectorEnvType,
)
from tianshou.highlevel.experiment import DQNExperimentBuilder, ExperimentConfig
from tianshou.highlevel.params.algorithm_params import DQNParams


def main(
    task: str = "CartPole-v1",
    persistence_base_dir: str = "log",
    num_experiments: int = 1,
    experiment_launcher: Literal["sequential", "joblib"] = "sequential",
) -> None:
    """
    Train an agent using DQN on a specified discrete task, potentially running multiple experiments with different seeds
    and evaluating the results using rliable.

    :param task: the discrete task to train on.
    :param persistence_base_dir: the base directory for logging and saving experiment data,
        the task name will be appended to it.
    :param num_experiments: the number of experiments to run. The experiments differ exclusively in the seeds.
    :param experiment_launcher: the type of experiment launcher to use, only has an effect if `num_experiments>1`.
        You can use "joblib" for parallel execution of whole experiments.
    """
    persistence_base_dir = os.path.abspath(os.path.join(persistence_base_dir, task))
    experiment_config = ExperimentConfig(persistence_base_dir=persistence_base_dir, watch=False)

    training_config = OffPolicyTrainingConfig(
        max_epochs=10,
        epoch_num_steps=10000,
        num_training_envs=10,
        num_test_envs=100,
        buffer_size=20000,
        batch_size=64,
        collection_step_num_env_steps=10,
        update_step_num_gradient_steps_per_sample=1 / 10,
        start_timesteps=0,
        start_timesteps_random=False,
    )

    env_factory = EnvFactoryRegistered(
        task=task, venv_type=VectorEnvType.DUMMY, train_seed=0, test_seed=10
    )

    hidden_sizes = (64, 64)
    experiment_builder = (
        DQNExperimentBuilder(env_factory, experiment_config, training_config)
        .with_dqn_params(
            DQNParams(
                lr=1e-3,
                gamma=0.9,
                n_step_return_horizon=3,
                target_update_freq=320,
                eps_training=0.3,
                eps_inference=0.0,
            ),
        )
        .with_model_factory_default(hidden_sizes=hidden_sizes)
    )

    experiment_builder.build_and_run(num_experiments=num_experiments, launcher=experiment_launcher)


if __name__ == "__main__":
    result = logging.run_cli(main, level=logging.INFO)
