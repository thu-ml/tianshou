import torch
from dataclasses import dataclass
from jsonargparse import set_docstring_parse_options
from typing import Literal, Optional, Sequence

set_docstring_parse_options(attribute_docstrings=True)


@dataclass
class BasicExperimentConfig:
    """Generic config for setting up the experiment, not RL or training specific."""

    seed: int = 42
    task: str = "Ant-v4"
    """Mujoco specific"""
    render: float = 0.01
    """Milliseconds between rendered frames"""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    resume_id: Optional[int] = None
    """For restoring a model and running means of env-specifics from a checkpoint"""
    resume_path: str = None
    """For restoring a model and running means of env-specifics from a checkpoint"""
    render_mode: Optional[Literal["human", "rgb_array"]] = None
    """Only affects train environments. None is recommended during training,
    but other modes may be useful when watch=True."""
    watch: bool = False
    """If True, will not perform training and only watch the restored policy"""


@dataclass
class LoggerConfig:
    """Logging config"""

    logdir: str = "log"
    logger: Literal["tensorboard", "wandb"] = "tensorboard"
    wandb_project: str = "mujoco.benchmark"
    """Only used if logger is wandb."""


@dataclass
class TrainerConfig:
    """Sampling, epochs, parallelization, buffers, collectors, and batching."""

    num_epochs: int = 100
    step_per_epoch: int = 30000
    batch_size: int = 1024
    num_train_envs: int = 64
    num_test_envs: int = 10
    """For watching a policy perform in the rendered env (e.g. with render_mode="human"), 
    `num_test_envs=1 `is recommended."""
    buffer_size: int = 20000
    start_timesteps: int = 0
    """If more than 0, will collect samples from the environment before training.
    Useful for prefilling replay buffers for off-policy algorithms."""
    start_timesteps_random: bool = True
    """Only used if start_timesteps > 0. If True, will collect samples from the
    environment using random actions. If False, will use the policy to collect
    samples."""
    step_per_collect: int = 2048
    repeat_per_collect: int = 10
    num_test_episodes_per_env: int = 1
    """
    By default, one episode is sampled from each test env for optimal parallelization.
    For visualization, it may be useful to decrease this and to set the num_test_envs to
    1
    """
    update_per_step: float = 1.0
    """The number of times the policy network would be updated per transition after 
    (step_per_collect) transitions are collected, e.g., if update_per_step set to 0.3, 
    and step_per_collect is 256 , policy will be updated round(256 * 0.3 = 76.8) = 77 
    times after 256 transitions are collected by the collector."""

    @property
    def num_test_episodes(self):
        return self.num_test_envs * self.num_test_episodes_per_env


@dataclass
class RLAgentConfig:
    """Config common to most RL algorithms"""

    gamma: float = 0.99
    """Discount factor"""
    gae_lambda: float = 0.95
    """For Generalized Advantage Estimate (equivalent to TD(lambda))"""
    action_bound_method: Optional[Literal["clip", "tanh"]] = "clip"
    """How to map original actions in range (-inf, inf) to [-1, 1]"""
    rew_norm: bool = True
    """Whether to normalize rewards"""


@dataclass
class PGConfig:
    """Config of general policy-gradient algorithms"""

    ent_coef: float = 0.0
    vf_coef: float = 0.25
    max_grad_norm: float = 0.5


@dataclass
class PPOConfig:
    """PPO specific config"""

    value_clip: bool = False
    norm_adv: bool = False
    """Whether to normalize advantages"""
    eps_clip: float = 0.2
    dual_clip: Optional[float] = None
    recompute_adv: bool = True


@dataclass
class NNConfig:
    hidden_sizes: Sequence[int] = (64, 64)
    lr: float = 3e-4
    lr_decay: bool = True
