#!/usr/bin/env python3

import datetime
import os
from collections.abc import Sequence
from dataclasses import dataclass

from jsonargparse import CLI
from torch.distributions import Independent, Normal

from examples.mujoco.mujoco_env import MujocoEnvFactory
from tianshou.highlevel.agent import PGConfig, PPOAgentFactory, PPOConfig, RLAgentConfig
from tianshou.highlevel.experiment import (
    RLExperiment,
    RLExperimentConfig,
    RLSamplingConfig,
)
from tianshou.highlevel.logger import DefaultLoggerFactory, LoggerConfig
from tianshou.highlevel.module import (
    ContinuousActorProbFactory,
    ContinuousNetCriticFactory,
)
from tianshou.highlevel.optim import AdamOptimizerFactory, LinearLRSchedulerFactory


@dataclass
class NNConfig:
    hidden_sizes: Sequence[int] = (64, 64)
    lr: float = 3e-4
    lr_decay: bool = True


def main(
    experiment_config: RLExperimentConfig,
    logger_config: LoggerConfig,
    sampling_config: RLSamplingConfig,
    general_config: RLAgentConfig,
    pg_config: PGConfig,
    ppo_config: PPOConfig,
    nn_config: NNConfig,
    task: str = "Ant-v4",
):
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    log_name = os.path.join(task, "ppo", str(experiment_config.seed), now)
    logger_factory = DefaultLoggerFactory(logger_config)

    env_factory = MujocoEnvFactory(task, experiment_config.seed, sampling_config)

    def dist_fn(*logits):
        return Independent(Normal(*logits), 1)

    actor_factory = ContinuousActorProbFactory(nn_config.hidden_sizes)
    critic_factory = ContinuousNetCriticFactory(nn_config.hidden_sizes)
    optim_factory = AdamOptimizerFactory()
    lr_scheduler_factory = LinearLRSchedulerFactory(sampling_config) if nn_config.lr_decay else None
    agent_factory = PPOAgentFactory(
        general_config,
        pg_config,
        ppo_config,
        sampling_config,
        actor_factory,
        critic_factory,
        optim_factory,
        dist_fn,
        nn_config.lr,
        lr_scheduler_factory,
    )

    experiment = RLExperiment(experiment_config, env_factory, logger_factory, agent_factory)

    experiment.run(log_name)


if __name__ == "__main__":
    CLI(main)
