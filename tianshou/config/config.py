from dataclasses import dataclass
from typing import Literal

import torch
from jsonargparse import set_docstring_parse_options

set_docstring_parse_options(attribute_docstrings=True)


@dataclass
class BasicExperimentConfig:
    """Generic config for setting up the experiment, not RL or training specific."""

    seed: int = 42
    task: str = "Ant-v4"
    """Mujoco specific"""
    render: float | None = 0.0
    """Milliseconds between rendered frames; if None, no rendering"""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    resume_id: str | None = None
    """For restoring a model and running means of env-specifics from a checkpoint"""
    resume_path: str | None = None
    """For restoring a model and running means of env-specifics from a checkpoint"""
    watch: bool = False
    """If True, will not perform training and only watch the restored policy"""
    watch_num_episodes = 10


@dataclass
class LoggerConfig:
    """Logging config."""

    logdir: str = "log"
    logger: Literal["tensorboard", "wandb"] = "tensorboard"
    wandb_project: str = "mujoco.benchmark"
    """Only used if logger is wandb."""


@dataclass
class RLSamplingConfig:
    """Sampling, epochs, parallelization, buffers, collectors, and batching."""

    num_epochs: int = 100
    step_per_epoch: int = 30000
    batch_size: int = 64
    num_train_envs: int = 64
    num_test_envs: int = 10
    buffer_size: int = 4096
    step_per_collect: int = 2048
    repeat_per_collect: int = 10
    update_per_step: int = 1


@dataclass
class RLAgentConfig:
    """Config common to most RL algorithms."""

    gamma: float = 0.99
    """Discount factor"""
    gae_lambda: float = 0.95
    """For Generalized Advantage Estimate (equivalent to TD(lambda))"""
    action_bound_method: Literal["clip", "tanh"] | None = "clip"
    """How to map original actions in range (-inf, inf) to [-1, 1]"""
    rew_norm: bool = True
    """Whether to normalize rewards"""


@dataclass
class PGConfig:
    """Config of general policy-gradient algorithms."""

    ent_coef: float = 0.0
    vf_coef: float = 0.25
    max_grad_norm: float = 0.5


@dataclass
class PPOConfig:
    """PPO specific config."""

    value_clip: bool = False
    norm_adv: bool = False
    """Whether to normalize advantages"""
    eps_clip: float = 0.2
    dual_clip: float | None = None
    recompute_adv: bool = True
