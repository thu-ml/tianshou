#!/usr/bin/env python3
"""The high-level multi experiment script demonstrates how to use the high-level API of TianShou to train
a single configuration of an experiment (here a PPO agent on mujoco) with multiple non-intersecting seeds.
Thus, the experiment will be repeated `num_experiments` times.
For each repetition, a policy seed, train env seeds, and test env seeds are set that
are non-intersecting with the seeds of the other experiments.
Each experiment's results are stored in a separate subdirectory.

The final results are aggregated and turned into useful statistics with the rliable package.
The call to `eval_experiments` will load the results from the log directory and
create an interp-quantile mean plot for the returns as well as a performance profile plot.
These plots are saved in the log directory and displayed in the console.
"""

import os
from typing import Literal

import torch
from sensai.util import logging

from examples.mujoco.mujoco_env import MujocoEnvFactory
from tianshou.highlevel.config import OnPolicyTrainingConfig
from tianshou.highlevel.experiment import (
    ExperimentConfig,
    PPOExperimentBuilder,
)
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.highlevel.params.algorithm_params import PPOParams
from tianshou.highlevel.params.lr_scheduler import LRSchedulerFactoryFactoryLinear


def main(
    num_experiments: int = 5,
    experiment_launcher_type: Literal["sequential", "joblib"] = "sequential",
    logger_type: Literal["tensorboard", "wandb"] = "tensorboard",
) -> None:
    """:param num_experiments: the number of experiments to run. The experiments differ exclusively in the seeds.
    :param experiment_launcher_type: the type of experiment launcher to use. Currently, "sequential"
        and "joblib" (for parallel execution) are supported.
    :param logger_type: the type of logger to use. Currently, "wandb" and "tensorboard" are supported,
        where "wandb" cannot be used with "joblib" yet.
    """
    if num_experiments > 1 and experiment_launcher_type == "joblib" and logger_type == "wandb":
        raise NotImplementedError(
            "Parallel execution with wandb logger is still under development. Falling back to tensorboard.",
        )

    task = "Ant-v4"
    persistence_base_dir = os.path.abspath(os.path.join("log", task))

    experiment_config = ExperimentConfig(persistence_base_dir=persistence_base_dir, watch=False)

    training_config = OnPolicyTrainingConfig(
        max_epochs=1,
        epoch_num_steps=5000,
        batch_size=64,
        num_train_envs=5,
        num_test_envs=5,
        test_step_num_episodes=5,
        buffer_size=4096,
        collection_step_num_env_steps=2048,
        update_step_num_repetitions=1,
    )

    env_factory = MujocoEnvFactory(task, obs_norm=True)

    hidden_sizes = (64, 64)

    match logger_type:
        case "wandb":
            job_type = "ppo"
            logger_factory = LoggerFactoryDefault(
                logger_type="wandb",
                wandb_project="tianshou",
                group=task,
                job_type=job_type,
                save_interval=1,
            )
        case "tensorboard":
            logger_factory = LoggerFactoryDefault("tensorboard")
        case _:
            raise ValueError(f"Unknown logger type: {logger_type}")

    experiment_builder = (
        PPOExperimentBuilder(env_factory, experiment_config, training_config)
        .with_ppo_params(
            PPOParams(
                gamma=0.99,
                gae_lambda=0.95,
                action_bound_method="clip",
                return_scaling=True,
                ent_coef=0.0,
                vf_coef=0.25,
                max_grad_norm=0.5,
                value_clip=False,
                advantage_normalization=False,
                eps_clip=0.2,
                dual_clip=None,
                recompute_advantage=True,
                lr=3e-4,
                lr_scheduler=LRSchedulerFactoryFactoryLinear(training_config),
            ),
        )
        .with_actor_factory_default(hidden_sizes, torch.nn.Tanh, continuous_unbounded=True)
        .with_critic_factory_default(hidden_sizes, torch.nn.Tanh)
        .with_logger_factory(logger_factory)
    )

    experiment_builder.build_and_run(num_experiments=num_experiments)


if __name__ == "__main__":
    result = logging.run_cli(main, level=logging.INFO)
