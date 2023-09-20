#!/usr/bin/env python3

import datetime
import os
from collections.abc import Sequence

from jsonargparse import CLI

from examples.mujoco.mujoco_env import MujocoEnvFactory
from tianshou.config import (
    BasicExperimentConfig,
    LoggerConfig,
    RLSamplingConfig,
)
from tianshou.highlevel.agent import SACAgentFactory
from tianshou.highlevel.experiment import RLExperiment
from tianshou.highlevel.logger import DefaultLoggerFactory
from tianshou.highlevel.module import (
    ContinuousActorProbFactory,
    ContinuousNetCriticFactory,
)
from tianshou.highlevel.optim import AdamOptimizerFactory


def main(
    experiment_config: BasicExperimentConfig,
    logger_config: LoggerConfig,
    sampling_config: RLSamplingConfig,
    sac_config: SACAgentFactory.Config,
    hidden_sizes: Sequence[int] = (256, 256),
):
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    log_name = os.path.join(experiment_config.task, "sac", str(experiment_config.seed), now)
    logger_factory = DefaultLoggerFactory(logger_config)

    env_factory = MujocoEnvFactory(experiment_config, sampling_config)

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
