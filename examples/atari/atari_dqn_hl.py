#!/usr/bin/env python3

import os

from examples.atari.atari_callbacks import (
    TestEpochCallbackDQNSetEps,
    TrainEpochCallbackNatureDQNEpsLinearDecay,
)
from examples.atari.atari_network import (
    IntermediateModuleFactoryAtariDQN,
    IntermediateModuleFactoryAtariDQNFeatures,
)
from examples.atari.atari_wrapper import AtariEnvFactory, AtariStopCallback
from tianshou.highlevel.config import SamplingConfig
from tianshou.highlevel.experiment import (
    DQNExperimentBuilder,
    ExperimentConfig,
)
from tianshou.highlevel.params.policy_params import DQNParams
from tianshou.highlevel.params.policy_wrapper import (
    PolicyWrapperFactoryIntrinsicCuriosity,
)
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
    icm_lr_scale: float = 0.0,
    icm_reward_scale: float = 0.01,
    icm_forward_loss_weight: float = 0.2,
):
    log_name = os.path.join(task, "dqn", str(experiment_config.seed), datetime_tag())

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

    builder = (
        DQNExperimentBuilder(env_factory, experiment_config, sampling_config)
        .with_dqn_params(
            DQNParams(
                discount_factor=gamma,
                estimation_step=n_step,
                lr=lr,
                target_update_freq=target_update_freq,
            ),
        )
        .with_model_factory(IntermediateModuleFactoryAtariDQN())
        .with_trainer_epoch_callback_train(
            TrainEpochCallbackNatureDQNEpsLinearDecay(eps_train, eps_train_final),
        )
        .with_trainer_epoch_callback_test(TestEpochCallbackDQNSetEps(eps_test))
        .with_trainer_stop_callback(AtariStopCallback(task))
    )
    if icm_lr_scale > 0:
        builder.with_policy_wrapper_factory(
            PolicyWrapperFactoryIntrinsicCuriosity(
                feature_net_factory=IntermediateModuleFactoryAtariDQNFeatures(),
                hidden_sizes=[512],
                lr=lr,
                lr_scale=icm_lr_scale,
                reward_scale=icm_reward_scale,
                forward_loss_weight=icm_forward_loss_weight,
            ),
        )

    experiment = builder.build()
    experiment.run(log_name)


if __name__ == "__main__":
    logging.run_cli(main)
