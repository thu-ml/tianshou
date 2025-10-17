#!/usr/bin/env python3

import os
from typing import Literal

from sensai.util import logging

from tianshou.env.atari.atari_network import (
    IntermediateModuleFactoryAtariDQN,
)
from tianshou.env.atari.atari_wrapper import AtariEnvFactory, AtariEpochStopCallback
from tianshou.highlevel.config import OffPolicyTrainingConfig
from tianshou.highlevel.experiment import (
    ExperimentConfig,
    IQNExperimentBuilder,
)
from tianshou.highlevel.params.algorithm_params import IQNParams
from tianshou.highlevel.trainer import (
    EpochTrainCallbackDQNEpsLinearDecay,
)


def main(
    task: str = "PongNoFrameskip-v4",
    persistence_base_dir: str = "log",
    num_experiments: int = 5,
    experiment_launcher: Literal["sequential", "joblib"] = "sequential",
) -> None:
    """
    Train an agent using IQN on a specified Atari task, potentially running multiple experiments with different seeds
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
        batch_size=32,
        num_train_envs=10,
        num_test_envs=10,
        buffer_size=100000,
        collection_step_num_env_steps=10,
        update_step_num_gradient_steps_per_sample=0.1,
        replay_buffer_stack_num=4,
        replay_buffer_ignore_obs_next=True,
        replay_buffer_save_only_last_obs=True,
    )

    env_factory = AtariEnvFactory(task, 4, scale=False)

    experiment_builder = (
        IQNExperimentBuilder(env_factory, experiment_config, training_config)
        .with_iqn_params(
            IQNParams(
                gamma=0.99,
                n_step_return_horizon=3,
                lr=0.0001,
                sample_size=32,
                online_sample_size=8,
                target_update_freq=500,
                target_sample_size=8,
                hidden_sizes=(512,),
                num_cosines=64,
                eps_training=1.0,
                eps_inference=0.005,
            ),
        )
        .with_preprocess_network_factory(IntermediateModuleFactoryAtariDQN(features_only=True))
        .with_epoch_train_callback(
            EpochTrainCallbackDQNEpsLinearDecay(1.0, 0.05),
        )
        .with_epoch_stop_callback(AtariEpochStopCallback(task))
    )

    experiment_builder.build_and_run(num_experiments=num_experiments, launcher=experiment_launcher)


if __name__ == "__main__":
    result = logging.run_cli(main, level=logging.INFO)
