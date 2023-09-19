#!/usr/bin/env python3

import datetime
import os

from jsonargparse import CLI
from torch.distributions import Independent, Normal

from examples.mujoco.mujoco_env import MujocoEnvFactory
from tianshou.config import (
    BasicExperimentConfig,
    LoggerConfig,
    NNConfig,
    PGConfig,
    PPOConfig,
    RLAgentConfig,
    RLSamplingConfig,
)
from tianshou.highlevel.agent import PPOAgentFactory
from tianshou.highlevel.logger import DefaultLoggerFactory
from tianshou.highlevel.module import ContinuousActorProbFactory, ContinuousNetCriticFactory
from tianshou.highlevel.optim import AdamOptimizerFactory, LinearLRSchedulerFactory
from tianshou.highlevel.experiment import RLExperiment


def main(
    experiment_config: BasicExperimentConfig,
    logger_config: LoggerConfig,
    sampling_config: RLSamplingConfig,
    general_config: RLAgentConfig,
    pg_config: PGConfig,
    ppo_config: PPOConfig,
    nn_config: NNConfig,
):
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    log_name = os.path.join(experiment_config.task, "ppo", str(experiment_config.seed), now)
    logger_factory = DefaultLoggerFactory(logger_config)

    env_factory = MujocoEnvFactory(experiment_config, sampling_config)

    def dist_fn(*logits):
        return Independent(Normal(*logits), 1)

    actor_factory = ContinuousActorProbFactory(nn_config.hidden_sizes)
    critic_factory = ContinuousNetCriticFactory(nn_config.hidden_sizes)
    optim_factory = AdamOptimizerFactory(lr=nn_config.lr)
    lr_scheduler_factory = LinearLRSchedulerFactory(nn_config, sampling_config)
    agent_factory = PPOAgentFactory(general_config, pg_config, ppo_config, sampling_config, nn_config,
        actor_factory, critic_factory, optim_factory, dist_fn, lr_scheduler_factory)

    experiment = RLExperiment(experiment_config, logger_config, general_config, sampling_config,
        env_factory,
        logger_factory,
        agent_factory)

    experiment.run(log_name)


if __name__ == "__main__":
    CLI(main)
