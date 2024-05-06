#!/usr/bin/env python3
"""The high-level multi experiment script demonstrates how to use the high-level API of TianShou to train
a single configuration of an experiment (here a PPO agent on mujoco) with multiple non-intersecting seeds.
Thus, the experiment will be repeated `num_experiments` times.
For each repetition, a policy seed, train env seeds, and test env seeds are set that
are non-intersecting with the seeds of the other experiments.
Each experiment's results are stored in a separate subdirectory.

The final results are aggregated and turned into useful statistics with the rliable API.
The call to `eval_experiments` will load the results from the log directory and
create an interp-quantile mean plot for the returns as well as a performance profile plot.
These plots are saved in the log directory and displayed in the console.
"""

import os
import sys
from collections.abc import Sequence
from typing import Literal

import torch

from examples.mujoco.mujoco_env import MujocoEnvFactory
from tianshou.evaluation.launcher import RegisteredExpLauncher
from tianshou.evaluation.rliable_evaluation_hl import RLiableExperimentResult
from tianshou.highlevel.config import SamplingConfig
from tianshou.highlevel.experiment import (
    ExperimentConfig,
    PPOExperimentBuilder,
)
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.highlevel.params.dist_fn import (
    DistributionFunctionFactoryIndependentGaussians,
)
from tianshou.highlevel.params.lr_scheduler import LRSchedulerFactoryLinear
from tianshou.highlevel.params.policy_params import PPOParams
from tianshou.utils import logging
from tianshou.utils.logging import datetime_tag

log = logging.getLogger(__name__)


def main(
    num_experiments: int = 2,
    run_experiments_sequentially: bool = True,
) -> RLiableExperimentResult:
    """:param run_experiments_sequentially: if True, the experiments are run sequentially, otherwise in parallel.
        LIMITATIONS: currently, the parallel execution does not seem to work properly on linux.
        It might generally be undesired to run multiple experiments in parallel on the same machine,
        as a single experiment already uses all available CPU cores by default.
    :return: the directory where the results are stored
    """
    task = "Ant-v4"
    persistence_dir = os.path.abspath(os.path.join("log", task, "ppo", datetime_tag()))

    experiment_config = ExperimentConfig(persistence_base_dir=persistence_dir, watch=False)

    sampling_config = SamplingConfig(
        num_epochs=1,
        step_per_epoch=5000,
        batch_size=64,
        num_train_envs=10,
        num_test_envs=10,
        num_test_episodes=10,
        buffer_size=4096,
        step_per_collect=2048,
        repeat_per_collect=10,
    )

    env_factory = MujocoEnvFactory(
        task,
        train_seed=sampling_config.train_seed,
        test_seed=sampling_config.test_seed,
        obs_norm=True,
    )

    hidden_sizes = (64, 64)

    experiment_collection = (
        PPOExperimentBuilder(env_factory, experiment_config, sampling_config)
        .with_ppo_params(
            PPOParams(
                discount_factor=0.99,
                gae_lambda=0.95,
                action_bound_method="clip",
                reward_normalization=True,
                ent_coef=0.0,
                vf_coef=0.25,
                max_grad_norm=0.5,
                value_clip=False,
                advantage_normalization=False,
                eps_clip=0.2,
                dual_clip=None,
                recompute_advantage=True,
                lr=3e-4,
                lr_scheduler_factory=LRSchedulerFactoryLinear(sampling_config),
                dist_fn=DistributionFunctionFactoryIndependentGaussians(),
            ),
        )
        .with_actor_factory_default(hidden_sizes, torch.nn.Tanh, continuous_unbounded=True)
        .with_critic_factory_default(hidden_sizes, torch.nn.Tanh)
        .with_logger_factory(LoggerFactoryDefault("tensorboard"))
        .build_seeded_collection(num_experiments)
    )

    if run_experiments_sequentially:
        launcher = RegisteredExpLauncher.sequential.create_launcher()
    else:
        launcher = RegisteredExpLauncher.joblib.create_launcher()
    experiment_collection.run(launcher)

    rliable_result = RLiableExperimentResult.load_from_disk(persistence_dir)
    rliable_result.eval_results(show_plots=True, save_plots=True)
    return rliable_result


if __name__ == "__main__":
    result = logging.run_cli(main, level=logging.INFO)
