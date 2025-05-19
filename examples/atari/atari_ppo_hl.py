#!/usr/bin/env python3

import os
from collections.abc import Sequence

from sensai.util import logging
from sensai.util.logging import datetime_tag

from tianshou.env.atari.atari_network import (
    ActorFactoryAtariDQN,
    IntermediateModuleFactoryAtariDQNFeatures,
)
from tianshou.env.atari.atari_wrapper import AtariEnvFactory, AtariEpochStopCallback
from tianshou.highlevel.config import OnPolicyTrainingConfig
from tianshou.highlevel.experiment import (
    ExperimentConfig,
    PPOExperimentBuilder,
)
from tianshou.highlevel.params.algorithm_params import PPOParams
from tianshou.highlevel.params.lr_scheduler import LRSchedulerFactoryFactoryLinear
from tianshou.highlevel.params.policy_wrapper import (
    AlgorithmWrapperFactoryIntrinsicCuriosity,
)


def main(
    experiment_config: ExperimentConfig,
    task: str = "PongNoFrameskip-v4",
    scale_obs: bool = True,
    buffer_size: int = 100000,
    lr: float = 2.5e-4,
    gamma: float = 0.99,
    epoch: int = 100,
    epoch_num_steps: int = 100000,
    collection_step_num_env_steps: int = 1000,
    update_step_num_repetitions: int = 4,
    batch_size: int = 256,
    hidden_sizes: Sequence[int] = (512,),
    num_train_envs: int = 10,
    num_test_envs: int = 10,
    return_scaling: bool = False,
    vf_coef: float = 0.25,
    ent_coef: float = 0.01,
    gae_lambda: float = 0.95,
    lr_decay: bool = True,
    max_grad_norm: float = 0.5,
    eps_clip: float = 0.1,
    dual_clip: float | None = None,
    value_clip: bool = True,
    advantage_normalization: bool = True,
    recompute_adv: bool = False,
    frames_stack: int = 4,
    save_buffer_name: str | None = None,  # TODO add support in high-level API?
    icm_lr_scale: float = 0.0,
    icm_reward_scale: float = 0.01,
    icm_forward_loss_weight: float = 0.2,
) -> None:
    log_name = os.path.join(task, "ppo", str(experiment_config.seed), datetime_tag())

    training_config = OnPolicyTrainingConfig(
        max_epochs=epoch,
        epoch_num_steps=epoch_num_steps,
        batch_size=batch_size,
        num_train_envs=num_train_envs,
        num_test_envs=num_test_envs,
        buffer_size=buffer_size,
        collection_step_num_env_steps=collection_step_num_env_steps,
        update_step_num_repetitions=update_step_num_repetitions,
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
        PPOExperimentBuilder(env_factory, experiment_config, training_config)
        .with_ppo_params(
            PPOParams(
                gamma=gamma,
                gae_lambda=gae_lambda,
                return_scaling=return_scaling,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                value_clip=value_clip,
                advantage_normalization=advantage_normalization,
                eps_clip=eps_clip,
                dual_clip=dual_clip,
                recompute_advantage=recompute_adv,
                lr=lr,
                lr_scheduler=LRSchedulerFactoryFactoryLinear(training_config) if lr_decay else None,
            ),
        )
        .with_actor_factory(ActorFactoryAtariDQN(scale_obs=scale_obs, features_only=True))
        .with_critic_factory_use_actor()
        .with_epoch_stop_callback(AtariEpochStopCallback(task))
    )
    if icm_lr_scale > 0:
        builder.with_algorithm_wrapper_factory(
            AlgorithmWrapperFactoryIntrinsicCuriosity(
                feature_net_factory=IntermediateModuleFactoryAtariDQNFeatures(),
                hidden_sizes=hidden_sizes,
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
