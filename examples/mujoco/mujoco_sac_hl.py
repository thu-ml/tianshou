#!/usr/bin/env python3

import datetime
import os
from collections.abc import Sequence

from jsonargparse import CLI

from examples.mujoco.mujoco_env import MujocoEnvFactory
from tianshou.highlevel.agent import SACAgentFactory, SACConfig
from tianshou.highlevel.experiment import (
    RLExperiment,
    RLExperimentConfig,
    RLSamplingConfig,
)
from tianshou.highlevel.logger import DefaultLoggerFactory
from tianshou.highlevel.module import (
    ContinuousActorProbFactory,
    ContinuousNetCriticFactory,
)
from tianshou.highlevel.optim import AdamOptimizerFactory


def main(
    experiment_config: RLExperimentConfig,
    sampling_config: RLSamplingConfig,
    sac_config: SACConfig,
    hidden_sizes: Sequence[int] = (256, 256),
    task: str = "Ant-v4",
):
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    log_name = os.path.join(task, "sac", str(experiment_config.seed), now)
    logger_factory = DefaultLoggerFactory()

    env_factory = MujocoEnvFactory(task, experiment_config.seed, sampling_config)

    actor_factory = ContinuousActorProbFactory(hidden_sizes, conditioned_sigma=True)
    critic_factory = ContinuousNetCriticFactory(hidden_sizes)
    optim_factory = AdamOptimizerFactory()
    agent_factory = SACAgentFactory(
        sac_config,
        sampling_config,
        actor_factory,
        critic_factory,
        critic_factory,
        optim_factory,
    )

    experiment = RLExperiment(experiment_config, env_factory, logger_factory, agent_factory)

    experiment.run(log_name)


if __name__ == "__main__":
    CLI(main)
