import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.exploration import BaseNoise
from tianshou.highlevel.config import RLSamplingConfig
from tianshou.highlevel.env import Environments
from tianshou.highlevel.logger import Logger
from tianshou.highlevel.module import ActorFactory, CriticFactory, TDevice
from tianshou.highlevel.optim import LRSchedulerFactory, OptimizerFactory
from tianshou.policy import BasePolicy, PPOPolicy, SACPolicy
from tianshou.trainer import BaseTrainer, OffpolicyTrainer, OnpolicyTrainer
from tianshou.utils.net.common import ActorCritic

CHECKPOINT_DICT_KEY_MODEL = "model"
CHECKPOINT_DICT_KEY_OBS_RMS = "obs_rms"


class AgentFactory(ABC):
    def __init__(self, sampling_config: RLSamplingConfig):
        self.sampling_config = sampling_config

    def create_train_test_collector(self, policy: BasePolicy, envs: Environments):
        buffer_size = self.sampling_config.buffer_size
        train_envs = envs.train_envs
        if len(train_envs) > 1:
            buffer = VectorReplayBuffer(buffer_size, len(train_envs))
        else:
            buffer = ReplayBuffer(buffer_size)
        train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
        test_collector = Collector(policy, envs.test_envs)
        if self.sampling_config.start_timesteps > 0:
            train_collector.collect(n_step=self.sampling_config.start_timesteps, random=True)
        return train_collector, test_collector

    @abstractmethod
    def create_policy(self, envs: Environments, device: TDevice) -> BasePolicy:
        pass

    @staticmethod
    def _create_save_best_fn(envs: Environments, log_path: str) -> Callable:
        def save_best_fn(pol: torch.nn.Module) -> None:
            state = {
                CHECKPOINT_DICT_KEY_MODEL: pol.state_dict(),
                CHECKPOINT_DICT_KEY_OBS_RMS: envs.train_envs.get_obs_rms(),
            }
            torch.save(state, os.path.join(log_path, "policy.pth"))

        return save_best_fn

    @staticmethod
    def load_checkpoint(policy: torch.nn.Module, path, envs: Environments, device: TDevice):
        ckpt = torch.load(path, map_location=device)
        policy.load_state_dict(ckpt[CHECKPOINT_DICT_KEY_MODEL])
        if envs.train_envs:
            envs.train_envs.set_obs_rms(ckpt[CHECKPOINT_DICT_KEY_OBS_RMS])
        if envs.test_envs:
            envs.test_envs.set_obs_rms(ckpt[CHECKPOINT_DICT_KEY_OBS_RMS])
        print("Loaded agent and obs. running means from: ", path)  # TODO logging

    @abstractmethod
    def create_trainer(
        self,
        policy: BasePolicy,
        train_collector: Collector,
        test_collector: Collector,
        envs: Environments,
        logger: Logger,
    ) -> BaseTrainer:
        pass


class OnpolicyAgentFactory(AgentFactory, ABC):
    def create_trainer(
        self,
        policy: BasePolicy,
        train_collector: Collector,
        test_collector: Collector,
        envs: Environments,
        logger: Logger,
    ) -> OnpolicyTrainer:
        sampling_config = self.sampling_config
        return OnpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=sampling_config.num_epochs,
            step_per_epoch=sampling_config.step_per_epoch,
            repeat_per_collect=sampling_config.repeat_per_collect,
            episode_per_test=sampling_config.num_test_envs,
            batch_size=sampling_config.batch_size,
            step_per_collect=sampling_config.step_per_collect,
            save_best_fn=self._create_save_best_fn(envs, logger.log_path),
            logger=logger.logger,
            test_in_train=False,
        )


class OffpolicyAgentFactory(AgentFactory, ABC):
    def create_trainer(
        self,
        policy: BasePolicy,
        train_collector: Collector,
        test_collector: Collector,
        envs: Environments,
        logger: Logger,
    ) -> OffpolicyTrainer:
        sampling_config = self.sampling_config
        return OffpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=sampling_config.num_epochs,
            step_per_epoch=sampling_config.step_per_epoch,
            step_per_collect=sampling_config.step_per_collect,
            episode_per_test=sampling_config.num_test_envs,
            batch_size=sampling_config.batch_size,
            save_best_fn=self._create_save_best_fn(envs, logger.log_path),
            logger=logger.logger,
            update_per_step=sampling_config.update_per_step,
            test_in_train=False,
        )


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
class PGConfig(RLAgentConfig):
    """Config of general policy-gradient algorithms."""

    ent_coef: float = 0.0
    vf_coef: float = 0.25
    max_grad_norm: float = 0.5


@dataclass
class PPOConfig(PGConfig):
    """PPO specific config."""

    value_clip: bool = False
    norm_adv: bool = False
    """Whether to normalize advantages"""
    eps_clip: float = 0.2
    dual_clip: float | None = None
    recompute_adv: bool = True
    dist_fn: Callable = None
    lr: float = 1e-3
    lr_scheduler_factory: LRSchedulerFactory | None = None


class PPOAgentFactory(OnpolicyAgentFactory):
    def __init__(
        self,
        config: PPOConfig,
        sampling_config: RLSamplingConfig,
        actor_factory: ActorFactory,
        critic_factory: CriticFactory,
        optimizer_factory: OptimizerFactory,
    ):
        super().__init__(sampling_config)
        self.optimizer_factory = optimizer_factory
        self.critic_factory = critic_factory
        self.actor_factory = actor_factory
        self.config = config

    def create_policy(self, envs: Environments, device: TDevice) -> PPOPolicy:
        actor = self.actor_factory.create_module(envs, device)
        critic = self.critic_factory.create_module(envs, device, use_action=False)
        actor_critic = ActorCritic(actor, critic)
        optim = self.optimizer_factory.create_optimizer(actor_critic, self.config.lr)
        if self.config.lr_scheduler_factory is not None:
            lr_scheduler = self.config.lr_scheduler_factory.create_scheduler(optim)
        else:
            lr_scheduler = None
        return PPOPolicy(
            # nn-stuff
            actor,
            critic,
            optim,
            dist_fn=self.config.dist_fn,
            lr_scheduler=lr_scheduler,
            # env-stuff
            action_space=envs.get_action_space(),
            action_scaling=True,
            # general_config
            discount_factor=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            reward_normalization=self.config.rew_norm,
            action_bound_method=self.config.action_bound_method,
            # pg_config
            max_grad_norm=self.config.max_grad_norm,
            vf_coef=self.config.vf_coef,
            ent_coef=self.config.ent_coef,
            # ppo_config
            eps_clip=self.config.eps_clip,
            value_clip=self.config.value_clip,
            dual_clip=self.config.dual_clip,
            advantage_normalization=self.config.norm_adv,
            recompute_advantage=self.config.recompute_adv,
        )


class AutoAlphaFactory(ABC):
    @abstractmethod
    def create_auto_alpha(
        self,
        envs: Environments,
        optim_factory: OptimizerFactory,
        device: TDevice,
    ):
        pass


class DefaultAutoAlphaFactory(AutoAlphaFactory):  # TODO better name?
    def __init__(self, lr: float = 3e-4):
        self.lr = lr

    def create_auto_alpha(
        self,
        envs: Environments,
        optim_factory: OptimizerFactory,
        device: TDevice,
    ) -> tuple[float, torch.Tensor, torch.optim.Optimizer]:
        target_entropy = -np.prod(envs.get_action_shape())
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=self.lr)
        return target_entropy, log_alpha, alpha_optim


@dataclass
class SACConfig:
    tau: float = 0.005
    gamma: float = 0.99
    alpha: float | tuple[float, torch.Tensor, torch.optim.Optimizer] | AutoAlphaFactory = 0.2
    reward_normalization: bool = False
    estimation_step: int = 1
    deterministic_eval: bool = True
    actor_lr: float = 1e-3
    critic1_lr: float = 1e-3
    critic2_lr: float = 1e-3


class SACAgentFactory(OffpolicyAgentFactory):
    def __init__(
        self,
        config: SACConfig,
        sampling_config: RLSamplingConfig,
        actor_factory: ActorFactory,
        critic1_factory: CriticFactory,
        critic2_factory: CriticFactory,
        optim_factory: OptimizerFactory,
        exploration_noise: BaseNoise | None = None,
    ):
        super().__init__(sampling_config)
        self.critic2_factory = critic2_factory
        self.critic1_factory = critic1_factory
        self.actor_factory = actor_factory
        self.exploration_noise = exploration_noise
        self.optim_factory = optim_factory
        self.config = config

    def create_policy(self, envs: Environments, device: TDevice) -> BasePolicy:
        actor = self.actor_factory.create_module(envs, device)
        critic1 = self.critic1_factory.create_module(envs, device, use_action=True)
        critic2 = self.critic2_factory.create_module(envs, device, use_action=True)
        actor_optim = self.optim_factory.create_optimizer(actor, lr=self.config.actor_lr)
        critic1_optim = self.optim_factory.create_optimizer(critic1, lr=self.config.critic1_lr)
        critic2_optim = self.optim_factory.create_optimizer(critic2, lr=self.config.critic2_lr)
        if isinstance(self.config.alpha, AutoAlphaFactory):
            alpha = self.config.alpha.create_auto_alpha(envs, self.optim_factory, device)
        else:
            alpha = self.config.alpha
        return SACPolicy(
            actor,
            actor_optim,
            critic1,
            critic1_optim,
            critic2,
            critic2_optim,
            tau=self.config.tau,
            gamma=self.config.gamma,
            alpha=alpha,
            estimation_step=self.config.estimation_step,
            action_space=envs.get_action_space(),
            deterministic_eval=self.config.deterministic_eval,
            exploration_noise=self.exploration_noise,
        )
