#!/usr/bin/env python3

import os
from collections.abc import Sequence

from examples.atari.atari_callbacks import (
    TestEpochCallbackDQNSetEps,
    TrainEpochCallbackNatureDQNEpsLinearDecay,
)
from examples.atari.atari_network import (
    IntermediateModuleFactoryAtariDQN,
)
from examples.atari.atari_wrapper import AtariEnvFactory, AtariStopCallback
from tianshou.highlevel.config import SamplingConfig
from tianshou.highlevel.experiment import (
    ExperimentConfig,
    IQNExperimentBuilder,
)
from tianshou.highlevel.params.policy_params import IQNParams
from tianshou.utils import logging
from tianshou.utils.logging import datetime_tag


def main(
    experiment_config: ExperimentConfig,
    task: str = "PongNoFrameskip-v4",
    scale_obs: int = 0,
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
    step_per_epoch: int = 100000,
    step_per_collect: int = 10,
    update_per_step: float = 0.1,
    batch_size: int = 32,
    training_num: int = 10,
    test_num: int = 10,
    frames_stack: int = 4,
    save_buffer_name: str | None = None,  # TODO support?
):
    log_name = os.path.join(task, "iqn", str(experiment_config.seed), datetime_tag())

    sampling_config = SamplingConfig(
        num_epochs=epoch,
        step_per_epoch=step_per_epoch,
        batch_size=batch_size,
        num_train_envs=training_num,
        num_test_envs=test_num,
        buffer_size=buffer_size,
        step_per_collect=step_per_collect,
        update_per_step=update_per_step,
        repeat_per_collect=None,
        replay_buffer_stack_num=frames_stack,
        replay_buffer_ignore_obs_next=True,
        replay_buffer_save_only_last_obs=True,
    )

    env_factory = AtariEnvFactory(task, experiment_config.seed, frames_stack, scale=scale_obs)

    experiment = (
        IQNExperimentBuilder(env_factory, experiment_config, sampling_config)
        .with_iqn_params(
            IQNParams(
                discount_factor=gamma,
                estimation_step=n_step,
                lr=lr,
                sample_size=sample_size,
                online_sample_size=online_sample_size,
                target_update_freq=target_update_freq,
                target_sample_size=target_sample_size,
                hidden_sizes=hidden_sizes,
                num_cosines=num_cosines,
            ),
        )
        .with_preprocess_network_factory(IntermediateModuleFactoryAtariDQN(features_only=True))
        .with_trainer_epoch_callback_train(
            TrainEpochCallbackNatureDQNEpsLinearDecay(eps_train, eps_train_final),
        )
        .with_trainer_epoch_callback_test(TestEpochCallbackDQNSetEps(eps_test))
        .with_trainer_stop_callback(AtariStopCallback(task))
        .build()
    )
    experiment.run(log_name)


if __name__ == "__main__":
    logging.run_cli(main)
