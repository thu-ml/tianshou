#!/usr/bin/env python3

import os
from collections.abc import Sequence

from sensai.util import logging
from sensai.util.logging import datetime_tag

from examples.atari.atari_network import (
    ActorFactoryAtariDQN,
    IntermediateModuleFactoryAtariDQNFeatures,
)
from examples.atari.atari_wrapper import AtariEnvFactory, AtariEpochStopCallback
from tianshou.highlevel.config import SamplingConfig
from tianshou.highlevel.experiment import (
    DiscreteSACExperimentBuilder,
    ExperimentConfig,
)
from tianshou.highlevel.params.alpha import AutoAlphaFactoryDefault
from tianshou.highlevel.params.policy_params import DiscreteSACParams
from tianshou.highlevel.params.policy_wrapper import (
    PolicyWrapperFactoryIntrinsicCuriosity,
)


def main(
    experiment_config: ExperimentConfig,
    task: str = "PongNoFrameskip-v4",
    scale_obs: bool = False,
    buffer_size: int = 100000,
    actor_lr: float = 1e-5,
    critic_lr: float = 1e-5,
    gamma: float = 0.99,
    n_step: int = 3,
    tau: float = 0.005,
    alpha: float = 0.05,
    auto_alpha: bool = False,
    alpha_lr: float = 3e-4,
    epoch: int = 100,
    step_per_epoch: int = 100000,
    step_per_collect: int = 10,
    update_per_step: float = 0.1,
    batch_size: int = 64,
    hidden_sizes: Sequence[int] = (512,),
    training_num: int = 10,
    test_num: int = 10,
    frames_stack: int = 4,
    icm_lr_scale: float = 0.0,
    icm_reward_scale: float = 0.01,
    icm_forward_loss_weight: float = 0.2,
) -> None:
    log_name = os.path.join(task, "sac", str(experiment_config.seed), datetime_tag())

    sampling_config = SamplingConfig(
        num_epochs=epoch,
        step_per_epoch=step_per_epoch,
        update_per_step=update_per_step,
        batch_size=batch_size,
        num_train_envs=training_num,
        num_test_envs=test_num,
        buffer_size=buffer_size,
        step_per_collect=step_per_collect,
        repeat_per_collect=None,
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
        DiscreteSACExperimentBuilder(env_factory, experiment_config, sampling_config)
        .with_sac_params(
            DiscreteSACParams(
                actor_lr=actor_lr,
                critic1_lr=critic_lr,
                critic2_lr=critic_lr,
                gamma=gamma,
                tau=tau,
                alpha=AutoAlphaFactoryDefault(lr=alpha_lr, target_entropy_coefficient=0.98)
                if auto_alpha
                else alpha,
                estimation_step=n_step,
            ),
        )
        .with_actor_factory(ActorFactoryAtariDQN(scale_obs=False, features_only=True))
        .with_common_critic_factory_use_actor()
        .with_epoch_stop_callback(AtariEpochStopCallback(task))
    )
    if icm_lr_scale > 0:
        builder.with_policy_wrapper_factory(
            PolicyWrapperFactoryIntrinsicCuriosity(
                feature_net_factory=IntermediateModuleFactoryAtariDQNFeatures(),
                hidden_sizes=hidden_sizes,
                lr=actor_lr,
                lr_scale=icm_lr_scale,
                reward_scale=icm_reward_scale,
                forward_loss_weight=icm_forward_loss_weight,
            ),
        )
    experiment = builder.build()
    experiment.run(run_name=log_name)


if __name__ == "__main__":
    logging.run_cli(main)
