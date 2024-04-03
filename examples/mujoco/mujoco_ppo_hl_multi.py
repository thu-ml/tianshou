#!/usr/bin/env python3

import os
import sys
from collections.abc import Sequence
from typing import Literal

import torch

from examples.mujoco.launcher import RegisteredExpLauncher
from examples.mujoco.mujoco_env import MujocoEnvFactory
from tianshou.highlevel.config import SamplingConfig
from tianshou.highlevel.env import VectorEnvType
from tianshou.highlevel.evaluation import RLiableExperimentResult
from tianshou.highlevel.experiment import (
    ExperimentConfig,
    PPOExperimentBuilder,
)
from tianshou.highlevel.params.dist_fn import (
    DistributionFunctionFactoryIndependentGaussians,
)
from tianshou.highlevel.params.lr_scheduler import LRSchedulerFactoryLinear
from tianshou.highlevel.params.policy_params import PPOParams
from tianshou.utils import logging
from tianshou.utils.logging import datetime_tag


def main(
    experiment_config: ExperimentConfig,
    task: str = "Ant-v4",
    num_experiments: int = 5,
    buffer_size: int = 4096,
    hidden_sizes: Sequence[int] = (64, 64),
    lr: float = 3e-4,
    gamma: float = 0.99,
    epoch: int = 100,
    step_per_epoch: int = 30000,
    step_per_collect: int = 2048,
    repeat_per_collect: int = 10,
    batch_size: int = 64,
    training_num: int = 10,
    test_num: int = 10,
    rew_norm: bool = True,
    vf_coef: float = 0.25,
    ent_coef: float = 0.0,
    gae_lambda: float = 0.95,
    bound_action_method: Literal["clip", "tanh"] | None = "clip",
    lr_decay: bool = True,
    max_grad_norm: float = 0.5,
    eps_clip: float = 0.2,
    dual_clip: float | None = None,
    value_clip: bool = False,
    norm_adv: bool = False,
    recompute_adv: bool = True,
    run_sequential: bool = False,
) -> str:
    """Use the high-level API of TianShou to evaluate the PPO algorithm on a MuJoCo environment with multiple seeds for
    a given configuration. The results for each run are stored in separate sub-folders. After the agents are trained,
    the results are evaluated using rliable API.
    """
    log_name = os.path.join("log", task, "ppo", datetime_tag())
    experiment_config.persistence_base_dir = log_name
    experiment_config.watch = False

    sampling_config = SamplingConfig(
        num_epochs=epoch,
        step_per_epoch=step_per_epoch,
        batch_size=batch_size,
        num_train_envs=training_num,
        num_test_envs=test_num,
        num_test_episodes=test_num,
        buffer_size=buffer_size,
        step_per_collect=step_per_collect,
        repeat_per_collect=repeat_per_collect,
    )

    env_factory = MujocoEnvFactory(
        task,
        train_seed=sampling_config.train_seed,
        test_seed=sampling_config.test_seed,
        obs_norm=True,
        venv_type=VectorEnvType.SUBPROC_SHARED_MEM_FORK_CONTEXT
        if sys.platform == "darwin"
        else VectorEnvType.SUBPROC_SHARED_MEM,
    )

    experiments = (
        PPOExperimentBuilder(env_factory, experiment_config, sampling_config)
        .with_ppo_params(
            PPOParams(
                discount_factor=gamma,
                gae_lambda=gae_lambda,
                action_bound_method=bound_action_method,
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
                dist_fn=DistributionFunctionFactoryIndependentGaussians(),
            ),
        )
        .with_actor_factory_default(hidden_sizes, torch.nn.Tanh, continuous_unbounded=True)
        .with_critic_factory_default(hidden_sizes, torch.nn.Tanh)
        .build_default_seeded_experiments(num_experiments)
    )

    if run_sequential:
        launcher = RegisteredExpLauncher.sequential.create_launcher()
    else:
        launcher = RegisteredExpLauncher.joblib.create_launcher()
    launcher.launch(experiments)

    return log_name


def eval_experiments(log_dir: str):
    """Evaluate the experiments in the given log directory using the rliable API."""
    rliable_result = RLiableExperimentResult.load_from_disk(log_dir)
    rliable_result.eval_results(save_figure=True)


if __name__ == "__main__":
    log_dir = logging.run_cli(main)
    eval_experiments(log_dir)
