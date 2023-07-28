#!/usr/bin/env python3

import os
import pprint

import torch
from jsonargparse import CLI
from torch import nn

from mujoco_env import make_mujoco_env
from tianshou.config import (
    BasicExperimentConfig,
    LoggerConfig,
    NNConfig,
    PGConfig,
    PPOConfig,
    RLAgentConfig,
    RLSamplingConfig,
)
from tianshou.config.utils import collect_configs
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import set_seed
from tianshou.utils.env import (
    get_continuous_env_info,
    get_train_test_collector,
    watch_agent,
)
from tianshou.utils.logger import get_logger_for_run
from tianshou.utils.lr_scheduler import get_linear_lr_schedular
from tianshou.utils.models import (
    fixed_std_normal,
    get_actor_critic,
    init_and_get_optim,
    resume_from_checkpoint,
)


def main(
    experiment_config: BasicExperimentConfig,
    logger_config: LoggerConfig,
    sampling_config: RLSamplingConfig,
    general_config: RLAgentConfig,
    pg_config: PGConfig,
    ppo_config: PPOConfig,
    nn_config: NNConfig,
):
    """
    Run the PPO test on the provided parameters.

    :param experiment_config: BasicExperimentConfig - not ML or RL specific
    :param logger_config: LoggerConfig
    :param sampling_config: SamplingConfig -
        sampling, epochs, parallelization, buffers, collectors, and batching.
    :param general_config: RLAgentConfig - general RL agent config
    :param pg_config: PGConfig: common to most policy gradient algorithms
    :param ppo_config: PPOConfig - PPO specific config
    :param nn_config: NNConfig - NN-training specific config

    :return: None
    """
    full_config = collect_configs(*locals().values())
    set_seed(experiment_config.seed)

    # create test and train envs, add env info to config
    env, train_envs, test_envs = make_mujoco_env(
        task=experiment_config.task,
        seed=experiment_config.seed,
        num_train_envs=sampling_config.num_train_envs,
        num_test_envs=sampling_config.num_test_envs,
        obs_norm=True,
        render_mode=experiment_config.render_mode,
    )

    # adding env_info to logged config
    state_shape, action_shape, max_action = get_continuous_env_info(env)
    full_config["env_info"] = {
        "state_shape": state_shape,
        "action_shape": action_shape,
        "max_action": max_action,
    }
    log_path, logger = get_logger_for_run(
        "ppo",
        experiment_config.task,
        logger_config,
        full_config,
        experiment_config.seed,
        experiment_config.resume_id,
    )

    # Setup NNs
    actor, critic = get_actor_critic(
        state_shape, nn_config.hidden_sizes, action_shape, experiment_config.device
    )
    optim = init_and_get_optim(actor, critic, nn_config.lr)

    lr_scheduler = None
    if nn_config.lr_decay:
        lr_scheduler = get_linear_lr_schedular(
            optim,
            sampling_config.step_per_epoch,
            sampling_config.step_per_collect,
            sampling_config.num_epochs,
        )

    policy = PPOPolicy(
        # nn-stuff
        actor,
        critic,
        optim,
        dist_fn=fixed_std_normal,
        lr_scheduler=lr_scheduler,
        # env-stuff
        action_space=train_envs.action_space,
        action_scaling=True,
        # general_config
        discount_factor=general_config.gamma,
        gae_lambda=general_config.gae_lambda,
        reward_normalization=general_config.rew_norm,
        action_bound_method=general_config.action_bound_method,
        # pg_config
        max_grad_norm=pg_config.max_grad_norm,
        vf_coef=pg_config.vf_coef,
        ent_coef=pg_config.ent_coef,
        # ppo_config
        eps_clip=ppo_config.eps_clip,
        value_clip=ppo_config.value_clip,
        dual_clip=ppo_config.dual_clip,
        advantage_normalization=ppo_config.norm_adv,
        recompute_advantage=ppo_config.recompute_adv,
    )

    if experiment_config.resume_path:
        resume_from_checkpoint(
            experiment_config.resume_path,
            policy,
            train_envs=train_envs,
            test_envs=test_envs,
            device=experiment_config.device,
        )

    train_collector, test_collector = get_train_test_collector(
        sampling_config.buffer_size,
        policy,
        train_envs,
        test_envs,
        start_timesteps=sampling_config.start_timesteps,
        start_timesteps_random=sampling_config.start_timesteps_random,
    )

    # TODO: test num is the number of test envs but used as episode_per_test
    #  here and in watch_agent
    if not experiment_config.watch:
        # RL training
        def save_best_fn(pol: nn.Module):
            state = {"model": pol.state_dict(), "obs_rms": train_envs.get_obs_rms()}
            torch.save(state, os.path.join(log_path, "policy.pth"))

        trainer = OnpolicyTrainer(
            policy,
            train_collector,
            test_collector,
            max_epoch=sampling_config.num_epochs,
            step_per_epoch=sampling_config.step_per_epoch,
            repeat_per_collect=sampling_config.repeat_per_collect,
            episode_per_test=sampling_config.num_test_episodes,
            batch_size=sampling_config.batch_size,
            step_per_collect=sampling_config.step_per_collect,
            save_best_fn=save_best_fn,
            logger=logger,
            test_in_train=False,
        )
        result = trainer.run()
        pprint.pprint(result)

    watch_agent(
        sampling_config.num_test_episodes_per_env,
        policy,
        test_collector,
        render=experiment_config.render,
    )


if __name__ == "__main__":
    CLI(main)
