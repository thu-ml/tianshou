#!/usr/bin/env python3

import os
from collections.abc import Sequence

from examples.atari.atari_network import (
    ActorFactoryAtariDQN,
    IntermediateModuleFactoryAtariDQNFeatures,
)
from examples.atari.atari_wrapper import AtariEnvFactory, AtariStopCallback
from tianshou.highlevel.config import SamplingConfig
from tianshou.highlevel.experiment import (
    ExperimentConfig,
    PPOExperimentBuilder,
)
from tianshou.highlevel.params.lr_scheduler import LRSchedulerFactoryLinear
from tianshou.highlevel.params.policy_params import PPOParams
from tianshou.highlevel.params.policy_wrapper import (
    PolicyWrapperFactoryIntrinsicCuriosity,
)
from tianshou.utils import logging
from tianshou.utils.logging import datetime_tag


def main(
    experiment_config: ExperimentConfig,
    task: str = "PongNoFrameskip-v4",
    scale_obs: bool = True,
    buffer_size: int = 100000,
    lr: float = 2.5e-4,
    gamma: float = 0.99,
    epoch: int = 100,
    step_per_epoch: int = 100000,
    step_per_collect: int = 1000,
    repeat_per_collect: int = 4,
    batch_size: int = 256,
    hidden_sizes: int | Sequence[int] = 512,
    training_num: int = 10,
    test_num: int = 10,
    rew_norm: bool = False,
    vf_coef: float = 0.25,
    ent_coef: float = 0.01,
    gae_lambda: float = 0.95,
    lr_decay: bool = True,
    max_grad_norm: float = 0.5,
    eps_clip: float = 0.1,
    dual_clip: float | None = None,
    value_clip: bool = True,
    norm_adv: bool = True,
    recompute_adv: bool = False,
    frames_stack: int = 4,
    save_buffer_name: str | None = None,  # TODO add support in high-level API?
    icm_lr_scale: float = 0.0,
    icm_reward_scale: float = 0.01,
    icm_forward_loss_weight: float = 0.2,
):
    log_name = os.path.join(task, "ppo", str(experiment_config.seed), datetime_tag())

    sampling_config = SamplingConfig(
        num_epochs=epoch,
        step_per_epoch=step_per_epoch,
        batch_size=batch_size,
        num_train_envs=training_num,
        num_test_envs=test_num,
        buffer_size=buffer_size,
        step_per_collect=step_per_collect,
        repeat_per_collect=repeat_per_collect,
        replay_buffer_stack_num=frames_stack,
        replay_buffer_ignore_obs_next=True,
        replay_buffer_save_only_last_obs=True,
    )

    env_factory = AtariEnvFactory(task, experiment_config.seed, frames_stack)

    builder = (
        PPOExperimentBuilder(env_factory, experiment_config, sampling_config)
        .with_ppo_params(
            PPOParams(
                discount_factor=gamma,
                gae_lambda=gae_lambda,
                reward_normalization=rew_norm,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                value_clip=value_clip,
                advantage_normalization=norm_adv,
                eps_clip=eps_clip,
                dual_clip=dual_clip,
                recompute_advantage=recompute_adv,
                lr=lr,
                lr_scheduler_factory=LRSchedulerFactoryLinear(sampling_config)
                if lr_decay
                else None,
            ),
        )
        .with_actor_factory(ActorFactoryAtariDQN(hidden_sizes, scale_obs, features_only=True))
        .with_critic_factory_use_actor()
        .with_trainer_stop_callback(AtariStopCallback(task))
    )
    if icm_lr_scale > 0:
        builder.with_policy_wrapper_factory(
            PolicyWrapperFactoryIntrinsicCuriosity(
                feature_net_factory=IntermediateModuleFactoryAtariDQNFeatures(),
                hidden_sizes=[hidden_sizes],
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
