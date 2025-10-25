#!/usr/bin/env python3

import os
from typing import Literal

from sensai.util import logging

from tianshou.env.atari.atari_network import (
    ActorFactoryAtariDQN,
)
from tianshou.env.atari.atari_wrapper import AtariEnvFactory, AtariEpochStopCallback
from tianshou.highlevel.config import OffPolicyTrainingConfig
from tianshou.highlevel.experiment import (
    DiscreteSACExperimentBuilder,
    ExperimentConfig,
)
from tianshou.highlevel.params.algorithm_params import DiscreteSACParams


def main(
    task: str = "PongNoFrameskip-v4",
    persistence_base_dir: str = "log",
    num_experiments: int = 1,
    experiment_launcher: Literal["sequential", "joblib"] = "sequential",
) -> None:
    """
    Train an agent using SAC on a specified Atari task, potentially running multiple experiments with different seeds
    and evaluating the results using rliable.

    :param task: the Atari task to train on.
    :param persistence_base_dir: the base directory for logging and saving experiment data,
        the task name will be appended to it.
    :param num_experiments: the number of experiments to run. The experiments differ exclusively in the seeds.
    :param experiment_launcher: the type of experiment launcher to use, only has an effect if `num_experiments>1`.
        You can use "joblib" for parallel execution of whole experiments.
    """
    persistence_base_dir = os.path.abspath(os.path.join(persistence_base_dir, task))
    experiment_config = ExperimentConfig(persistence_base_dir=persistence_base_dir, watch=False)

    training_config = OffPolicyTrainingConfig(
        max_epochs=100,
        epoch_num_steps=100000,
        update_step_num_gradient_steps_per_sample=0.1,
        batch_size=64,
        num_training_envs=10,
        num_test_envs=10,
        buffer_size=100000,
        collection_step_num_env_steps=10,
        replay_buffer_stack_num=4,
        replay_buffer_ignore_obs_next=True,
        replay_buffer_save_only_last_obs=True,
    )

    env_factory = AtariEnvFactory(task, 4, scale=False)

    experiment_builder = (
        DiscreteSACExperimentBuilder(env_factory, experiment_config, training_config)
        .with_sac_params(
            DiscreteSACParams(
                actor_lr=1e-5,
                critic1_lr=1e-5,
                critic2_lr=1e-5,
                gamma=0.99,
                tau=0.005,
                alpha=0.05,
                n_step_return_horizon=3,
            ),
        )
        .with_actor_factory(ActorFactoryAtariDQN(scale_obs=False, features_only=True))
        .with_common_critic_factory_use_actor()
        .with_epoch_stop_callback(AtariEpochStopCallback(task))
    )

    experiment_builder.build_and_run(num_experiments=num_experiments, launcher=experiment_launcher)


if __name__ == "__main__":
    result = logging.run_cli(main, level=logging.INFO)
