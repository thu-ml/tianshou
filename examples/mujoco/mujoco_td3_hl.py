#!/usr/bin/env python3

import datetime
import os
from collections.abc import Sequence

from jsonargparse import CLI

from examples.mujoco.mujoco_env import MujocoEnvFactory
from tianshou.highlevel.config import RLSamplingConfig
from tianshou.highlevel.experiment import (
    RLExperimentConfig,
    TD3ExperimentBuilder,
)
from tianshou.highlevel.params.env_param import MaxActionScaledFloatEnvParamFactory
from tianshou.highlevel.params.noise import MaxActionScaledGaussianNoiseFactory
from tianshou.highlevel.params.policy_params import TD3Params


def main(
    experiment_config: RLExperimentConfig,
    task: str = "Ant-v3",
    buffer_size: int = 1000000,
    hidden_sizes: Sequence[int] = (256, 256),
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    gamma: float = 0.99,
    tau: float = 0.005,
    exploration_noise: float = 0.1,
    policy_noise: float = 0.2,
    noise_clip: float = 0.5,
    update_actor_freq: int = 2,
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
    log_name = os.path.join(task, "td3", str(experiment_config.seed), now)

    sampling_config = RLSamplingConfig(
        num_epochs=epoch,
        step_per_epoch=step_per_epoch,
        num_train_envs=training_num,
        num_test_envs=test_num,
        buffer_size=buffer_size,
        batch_size=batch_size,
        step_per_collect=step_per_collect,
        update_per_step=update_per_step,
        start_timesteps=start_timesteps,
        start_timesteps_random=True,
    )

    env_factory = MujocoEnvFactory(task, experiment_config.seed, sampling_config)

    experiment = (
        TD3ExperimentBuilder(experiment_config, env_factory, sampling_config)
        .with_td3_params(
            TD3Params(
                tau=tau,
                gamma=gamma,
                estimation_step=n_step,
                update_actor_freq=update_actor_freq,
                noise_clip=MaxActionScaledFloatEnvParamFactory(noise_clip),
                policy_noise=MaxActionScaledFloatEnvParamFactory(policy_noise),
                exploration_noise=MaxActionScaledGaussianNoiseFactory(exploration_noise),
                actor_lr=actor_lr,
                critic1_lr=critic_lr,
                critic2_lr=critic_lr,
            ),
        )
        .with_actor_factory_default(hidden_sizes)
        .with_common_critic_factory_default(hidden_sizes)
        .build()
    )
    experiment.run(log_name)


if __name__ == "__main__":
    CLI(main)