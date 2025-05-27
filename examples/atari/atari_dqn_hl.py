#!/usr/bin/env python3

import os

from sensai.util import logging
from sensai.util.logging import datetime_tag

from tianshou.env.atari.atari_network import (
    IntermediateModuleFactoryAtariDQN,
    IntermediateModuleFactoryAtariDQNFeatures,
)
from tianshou.env.atari.atari_wrapper import AtariEnvFactory, AtariEpochStopCallback
from tianshou.highlevel.config import OffPolicyTrainingConfig
from tianshou.highlevel.experiment import (
    DQNExperimentBuilder,
    ExperimentConfig,
)
from tianshou.highlevel.params.algorithm_params import DQNParams
from tianshou.highlevel.params.algorithm_wrapper import (
    AlgorithmWrapperFactoryIntrinsicCuriosity,
)
from tianshou.highlevel.trainer import (
    EpochTestCallbackDQNSetEps,
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
    icm_lr_scale: float = 0.0,
    icm_reward_scale: float = 0.01,
    icm_forward_loss_weight: float = 0.2,
) -> None:
    log_name = os.path.join(task, "dqn", str(experiment_config.seed), datetime_tag())

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

    builder = (
        DQNExperimentBuilder(env_factory, experiment_config, training_config)
        .with_dqn_params(
            DQNParams(
                gamma=gamma,
                n_step_return_horizon=n_step,
                lr=lr,
                target_update_freq=target_update_freq,
            ),
        )
        .with_model_factory(IntermediateModuleFactoryAtariDQN())
        .with_epoch_train_callback(
            EpochTrainCallbackDQNEpsLinearDecay(eps_train, eps_train_final),
        )
        .with_epoch_test_callback(EpochTestCallbackDQNSetEps(eps_test))
        .with_epoch_stop_callback(AtariEpochStopCallback(task))
    )
    if icm_lr_scale > 0:
        builder.with_algorithm_wrapper_factory(
            AlgorithmWrapperFactoryIntrinsicCuriosity(
                feature_net_factory=IntermediateModuleFactoryAtariDQNFeatures(),
                hidden_sizes=[512],
                lr=lr,
                lr_scale=icm_lr_scale,
                reward_scale=icm_reward_scale,
                forward_loss_weight=icm_forward_loss_weight,
            ),
        )

    experiment = builder.build()
    experiment.run(run_name=log_name)


if __name__ == "__main__":
    logging.run_cli(main)
