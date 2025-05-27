#!/usr/bin/env python3
"""The high-level multi experiment script demonstrates how to use the high-level API of TianShou to train
a single configuration of an experiment (here a PPO agent on mujoco) with multiple non-intersecting seeds.
Thus, the experiment will be repeated `num_experiments` times.
For each repetition, a policy seed, train env seeds, and test env seeds are set that
are non-intersecting with the seeds of the other experiments.
Each experiment's results are stored in a separate subdirectory.

The final results are aggregated and turned into useful statistics with the rliable package.
The call to `eval_experiments` will load the results from the log directory and
create an interp-quantile mean plot for the returns as well as a performance profile plot.
These plots are saved in the log directory and displayed in the console.
"""

import os
import warnings

import torch
from sensai.util import logging
from sensai.util.logging import datetime_tag

from examples.mujoco.mujoco_env import MujocoEnvFactory
from tianshou.evaluation.launcher import RegisteredExpLauncher
from tianshou.evaluation.rliable_evaluation_hl import RLiableExperimentResult
from tianshou.highlevel.config import SamplingConfig
from tianshou.highlevel.experiment import (
    ExperimentConfig,
    PPOExperimentBuilder,
)
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.highlevel.params.lr_scheduler import LRSchedulerFactoryLinear
from tianshou.highlevel.params.policy_params import PPOParams

log = logging.getLogger(__name__)


def main(
    num_experiments: int = 5,
    run_experiments_sequentially: bool = True,
    logger_type: str = "tensorboard",
) -> RLiableExperimentResult:
    """:param num_experiments: the number of experiments to run. The experiments differ exclusively in the seeds.
    :param run_experiments_sequentially: if True, the experiments are run sequentially, otherwise in parallel.
        If a single experiment is set to use all available CPU cores,
        it might be undesired to run multiple experiments in parallel on the same machine,
    :param logger_type: the type of logger to use. Currently, "wandb" and "tensorboard" are supported.
    :return: an object containing rliable-based evaluation results
    """
    if not run_experiments_sequentially and logger_type == "wandb":
        warnings.warn(
            "Parallel execution with wandb logger is still under development. Falling back to tensorboard.",
        )
        logger_type = "tensorboard"

    task = "Ant-v4"
    tag = datetime_tag()
    persistence_dir = os.path.abspath(os.path.join("log", task, "ppo", tag))

    experiment_config = ExperimentConfig(persistence_base_dir=persistence_dir, watch=False)

    sampling_config = SamplingConfig(
        num_epochs=1,
        step_per_epoch=5000,
        batch_size=64,
        num_train_envs=5,
        num_test_envs=5,
        num_test_episodes=5,
        buffer_size=4096,
        step_per_collect=2048,
        repeat_per_collect=1,
    )

    env_factory = MujocoEnvFactory(task, obs_norm=True)

    hidden_sizes = (64, 64)

    match logger_type:
        case "wandb":
            job_type = f"ppo/{tag}"
            logger_factory = LoggerFactoryDefault(
                logger_type="wandb",
                wandb_project="tianshou",
                group=task,
                job_type=job_type,
                save_interval=1,
            )
        case "tensorboard":
            logger_factory = LoggerFactoryDefault("tensorboard")
        case _:
            raise ValueError(f"Unknown logger type: {logger_type}")

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
            ),
        )
        .with_actor_factory_default(hidden_sizes, torch.nn.Tanh, continuous_unbounded=True)
        .with_critic_factory_default(hidden_sizes, torch.nn.Tanh)
        .with_logger_factory(logger_factory)
        .build_seeded_collection(num_experiments)
    )

    if run_experiments_sequentially:
        launcher = RegisteredExpLauncher.sequential.create_launcher()
    else:
        launcher = RegisteredExpLauncher.joblib.create_launcher()
    successful_experiment_stats = experiment_collection.run(launcher)
    log.info(f"Successfully completed {len(successful_experiment_stats)} experiments.")

    num_successful_experiments = len(successful_experiment_stats)
    for i, info_stats in enumerate(successful_experiment_stats, start=1):
        if info_stats is not None:
            log.info(f"Training stats for successful experiment {i}/{num_successful_experiments}:")
            log.info(info_stats.pprints_asdict())
        else:
            log.info(
                f"No training stats available for successful experiment {i}/{num_successful_experiments}.",
            )

    rliable_result = RLiableExperimentResult.load_from_disk(persistence_dir)
    rliable_result.eval_results(show_plots=True, save_plots=True)
    return rliable_result


if __name__ == "__main__":
    result = logging.run_cli(main, level=logging.INFO)
