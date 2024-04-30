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
from collections.abc import Sequence
from typing import Literal

import torch

from examples.mujoco.mujoco_env import MujocoEnvFactory
from tianshou.evaluation.launcher import RegisteredExpLauncher
from tianshou.evaluation.rliable_evaluation_hl import RLiableExperimentResult
from tianshou.highlevel.config import SamplingConfig
from tianshou.highlevel.env import VectorEnvType
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
    run_experiments_sequentially: bool = False,
) -> str:
    """Use the high-level API of TianShou to evaluate the PPO algorithm on a MuJoCo environment with multiple seeds for
    a given configuration. The results for each run are stored in separate sub-folders. After the agents are trained,
    the results are evaluated using the rliable API.

    :param experiment_config:
    :param task: a mujoco task name
    :param num_experiments: how many experiments to run with different seeds
    :param buffer_size:
    :param hidden_sizes:
    :param lr:
    :param gamma:
    :param epoch:
    :param step_per_epoch:
    :param step_per_collect:
    :param repeat_per_collect:
    :param batch_size:
    :param training_num:
    :param test_num:
    :param rew_norm:
    :param vf_coef:
    :param ent_coef:
    :param gae_lambda:
    :param bound_action_method:
    :param lr_decay:
    :param max_grad_norm:
    :param eps_clip:
    :param dual_clip:
    :param value_clip:
    :param norm_adv:
    :param recompute_adv:
    :param run_experiments_sequentially: if True, the experiments are run sequentially, otherwise in parallel.
        LIMITATIONS: currently, the parallel execution does not seem to work properly on linux.
        It might generally be undesired to run multiple experiments in parallel on the same machine,
        as a single experiment already uses all available CPU cores by default.
    :return: the directory where the results are stored
    """
    persistence_dir = os.path.abspath(os.path.join("log", task, "ppo", datetime_tag()))

    experiment_config.persistence_base_dir = persistence_dir
    log.info(f"Will save all experiment results to {persistence_dir}.")
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
        venv_type=VectorEnvType.SUBPROC_SHARED_MEM_FORK_CONTEXT,
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
        .with_logger_factory(LoggerFactoryDefault("tensorboard"))
        .build(num_experiments=num_experiments)
    )

    if run_experiments_sequentially:
        launcher = RegisteredExpLauncher.sequential.create_launcher()
    else:
        launcher = RegisteredExpLauncher.joblib.create_launcher()
    launcher.launch(experiments)

    return persistence_dir


def eval_experiments(log_dir: str) -> RLiableExperimentResult:
    """Evaluate the experiments in the given log directory using the rliable API."""
    rliable_result = RLiableExperimentResult.load_from_disk(log_dir)
    rliable_result.eval_results(show_plots=True, save_plots=True)
    return rliable_result


if __name__ == "__main__":
    log_dir = logging.run_cli(main, level=logging.INFO)
    assert isinstance(log_dir, str)  # for mypy
    evaluation_result = eval_experiments(log_dir)
