#!/usr/bin/env python3

import datetime
import os
from collections.abc import Sequence

from jsonargparse import CLI

from examples.mujoco.mujoco_env import MujocoEnvFactory
from tianshou.highlevel.config import RLSamplingConfig
from tianshou.highlevel.experiment import (
    DDPGExperimentBuilder,
    RLExperimentConfig,
)
from tianshou.highlevel.params.noise import MaxActionScaledGaussian
from tianshou.highlevel.params.policy_params import DDPGParams


def main(
    experiment_config: RLExperimentConfig,
    task: str = "Ant-v3",
    buffer_size: int = 1000000,
    hidden_sizes: Sequence[int] = (256, 256),
    actor_lr: float = 1e-3,
    critic_lr: float = 1e-3,
    gamma: float = 0.99,
    tau: float = 0.005,
    exploration_noise: float = 0.1,
    start_timesteps: int = 25000,
    epoch: int = 200,
    step_per_epoch: int = 5000,
    step_per_collect: int = 1,
    update_per_step: int = 1,
    n_step: int = 1,
    batch_size: int = 256,
    training_num: int = 1,
    test_num: int = 10,
):
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    log_name = os.path.join(task, "ppo", str(experiment_config.seed), now)

    sampling_config = RLSamplingConfig(
        num_epochs=epoch,
        step_per_epoch=step_per_epoch,
        batch_size=batch_size,
        num_train_envs=training_num,
        num_test_envs=test_num,
        buffer_size=buffer_size,
        step_per_collect=step_per_collect,
        update_per_step=update_per_step,
        repeat_per_collect=None,
        start_timesteps=start_timesteps,
        start_timesteps_random=True,
    )

    env_factory = MujocoEnvFactory(task, experiment_config.seed, sampling_config)

    experiment = (
        DDPGExperimentBuilder(experiment_config, env_factory, sampling_config)
        .with_ddpg_params(
            DDPGParams(
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                gamma=gamma,
                tau=tau,
                exploration_noise=MaxActionScaledGaussian(exploration_noise),
                estimation_step=n_step,
            ),
        )
        .with_actor_factory_default(hidden_sizes)
        .with_critic_factory_default(hidden_sizes)
        .build()
    )
    experiment.run(log_name)


if __name__ == "__main__":
    CLI(main)
