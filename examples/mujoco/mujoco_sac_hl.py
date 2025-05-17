#!/usr/bin/env python3

import os
from collections.abc import Sequence

from sensai.util import logging
from sensai.util.logging import datetime_tag

from examples.mujoco.mujoco_env import MujocoEnvFactory
from tianshou.highlevel.config import OffPolicyTrainingConfig
from tianshou.highlevel.experiment import (
    ExperimentConfig,
    SACExperimentBuilder,
)
from tianshou.highlevel.params.algorithm_params import SACParams
from tianshou.highlevel.params.alpha import AutoAlphaFactoryDefault


def main(
    experiment_config: ExperimentConfig,
    task: str = "Ant-v4",
    buffer_size: int = 1000000,
    hidden_sizes: Sequence[int] = (256, 256),
    actor_lr: float = 1e-3,
    critic_lr: float = 1e-3,
    gamma: float = 0.99,
    tau: float = 0.005,
    alpha: float = 0.2,
    auto_alpha: bool = False,
    alpha_lr: float = 3e-4,
    start_timesteps: int = 10000,
    epoch: int = 200,
    epoch_num_steps: int = 5000,
    collection_step_num_env_steps: int = 1,
    update_per_step: int = 1,
    n_step: int = 1,
    batch_size: int = 256,
    num_train_envs: int = 1,
    test_num: int = 10,
) -> None:
    log_name = os.path.join(task, "sac", str(experiment_config.seed), datetime_tag())

    training_config = OffPolicyTrainingConfig(
        max_epochs=epoch,
        epoch_num_steps=epoch_num_steps,
        num_train_envs=num_train_envs,
        num_test_envs=test_num,
        buffer_size=buffer_size,
        batch_size=batch_size,
        collection_step_num_env_steps=collection_step_num_env_steps,
        update_step_num_gradient_steps_per_sample=update_per_step,
        start_timesteps=start_timesteps,
        start_timesteps_random=True,
    )

    env_factory = MujocoEnvFactory(
        task,
        train_seed=training_config.train_seed,
        test_seed=training_config.test_seed,
        obs_norm=False,
    )

    experiment = (
        SACExperimentBuilder(env_factory, experiment_config, training_config)
        .with_sac_params(
            SACParams(
                tau=tau,
                gamma=gamma,
                alpha=AutoAlphaFactoryDefault(lr=alpha_lr) if auto_alpha else alpha,
                n_step_return_horizon=n_step,
                actor_lr=actor_lr,
                critic1_lr=critic_lr,
                critic2_lr=critic_lr,
            ),
        )
        .with_actor_factory_default(
            hidden_sizes,
            continuous_unbounded=True,
            continuous_conditioned_sigma=True,
        )
        .with_common_critic_factory_default(hidden_sizes)
        .build()
    )
    experiment.run(run_name=log_name)


if __name__ == "__main__":
    logging.run_cli(main)
