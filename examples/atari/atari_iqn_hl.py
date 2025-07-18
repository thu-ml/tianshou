#!/usr/bin/env python3

import os
from collections.abc import Sequence

from sensai.util import logging
from sensai.util.logging import datetime_tag

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
    experiment_config: ExperimentConfig,
    task: str = "PongNoFrameskip-v4",
    scale_obs: bool = False,
    eps_test: float = 0.005,
    eps_train: float = 1.0,
    eps_train_final: float = 0.05,
    buffer_size: int = 100000,
    lr: float = 0.0001,
    gamma: float = 0.99,
    sample_size: int = 32,
    online_sample_size: int = 8,
    target_sample_size: int = 8,
    num_cosines: int = 64,
    hidden_sizes: Sequence[int] = (512,),
    n_step: int = 3,
    target_update_freq: int = 500,
    epoch: int = 100,
    epoch_num_steps: int = 100000,
    collection_step_num_env_steps: int = 10,
    update_per_step: float = 0.1,
    batch_size: int = 32,
    num_train_envs: int = 10,
    num_test_envs: int = 10,
    frames_stack: int = 4,
) -> None:
    log_name = os.path.join(task, "iqn", str(experiment_config.seed), datetime_tag())

    training_config = OffPolicyTrainingConfig(
        max_epochs=epoch,
        epoch_num_steps=epoch_num_steps,
        batch_size=batch_size,
        num_train_envs=num_train_envs,
        num_test_envs=num_test_envs,
        buffer_size=buffer_size,
        collection_step_num_env_steps=collection_step_num_env_steps,
        update_step_num_gradient_steps_per_sample=update_per_step,
        replay_buffer_stack_num=frames_stack,
        replay_buffer_ignore_obs_next=True,
        replay_buffer_save_only_last_obs=True,
    )

    env_factory = AtariEnvFactory(
        task,
        frames_stack,
        scale=scale_obs,
    )

    experiment = (
        IQNExperimentBuilder(env_factory, experiment_config, training_config)
        .with_iqn_params(
            IQNParams(
                gamma=gamma,
                n_step_return_horizon=n_step,
                lr=lr,
                sample_size=sample_size,
                online_sample_size=online_sample_size,
                target_update_freq=target_update_freq,
                target_sample_size=target_sample_size,
                hidden_sizes=hidden_sizes,
                num_cosines=num_cosines,
                eps_training=eps_train,
                eps_inference=eps_test,
            ),
        )
        .with_preprocess_network_factory(IntermediateModuleFactoryAtariDQN(features_only=True))
        .with_epoch_train_callback(
            EpochTrainCallbackDQNEpsLinearDecay(eps_train, eps_train_final),
        )
        .with_epoch_stop_callback(AtariEpochStopCallback(task))
        .build()
    )
    experiment.run(run_name=log_name)


if __name__ == "__main__":
    logging.run_cli(main)
