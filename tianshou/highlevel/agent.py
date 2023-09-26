import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.highlevel.config import RLSamplingConfig
from tianshou.highlevel.env import Environments
from tianshou.highlevel.logger import Logger
from tianshou.highlevel.module import (
    ActorCriticModuleOpt,
    ActorFactory,
    ActorModuleOptFactory,
    CriticFactory,
    CriticModuleOptFactory,
    ModuleOpt,
    TDevice,
)
from tianshou.highlevel.optim import OptimizerFactory
from tianshou.highlevel.params.alpha import AutoAlphaFactory
from tianshou.highlevel.params.env_param import FloatEnvParamFactory
from tianshou.highlevel.params.lr_scheduler import LRSchedulerFactory
from tianshou.highlevel.params.noise import NoiseFactory
from tianshou.highlevel.params.policy_params import (
    ParamTransformer,
    PPOParams,
    SACParams,
    TD3Params,
)
from tianshou.policy import BasePolicy, PPOPolicy, SACPolicy, TD3Policy
from tianshou.policy.modelfree.pg import TDistParams
from tianshou.trainer import BaseTrainer, OffpolicyTrainer, OnpolicyTrainer
from tianshou.utils import MultipleLRSchedulers
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


class ParamTransformerDrop(ParamTransformer):
    def __init__(self, *keys: str):
        self.keys = keys

    def transform(self, kwargs: dict[str, Any]) -> None:
        for k in self.keys:
            del kwargs[k]


class ParamTransformerLRScheduler(ParamTransformer):
    def __init__(self, optim: torch.optim.Optimizer):
        self.optim = optim

    def transform(self, kwargs: dict[str, Any]) -> None:
        factory: LRSchedulerFactory | None = self.get(kwargs, "lr_scheduler_factory", drop=True)
        kwargs["lr_scheduler"] = (
            factory.create_scheduler(self.optim) if factory is not None else None
        )


class _ActorMixin:
    def __init__(self, actor_factory: ActorFactory, optim_factory: OptimizerFactory):
        self.actor_module_opt_factory = ActorModuleOptFactory(actor_factory, optim_factory)

    def create_actor_module_opt(self, envs: Environments, device: TDevice, lr: float) -> ModuleOpt:
        return self.actor_module_opt_factory.create_module_opt(envs, device, lr)


class _ActorCriticMixin:
    """Mixin for agents that use an ActorCritic module with a single optimizer."""

    def __init__(
        self,
        actor_factory: ActorFactory,
        critic_factory: CriticFactory,
        optim_factory: OptimizerFactory,
        critic_use_action: bool,
    ):
        self.actor_factory = actor_factory
        self.critic_factory = critic_factory
        self.optim_factory = optim_factory
        self.critic_use_action = critic_use_action

    def create_actor_critic_module_opt(
        self,
        envs: Environments,
        device: TDevice,
        lr: float,
    ) -> ActorCriticModuleOpt:
        actor = self.actor_factory.create_module(envs, device)
        critic = self.critic_factory.create_module(envs, device, use_action=self.critic_use_action)
        actor_critic = ActorCritic(actor, critic)
        optim = self.optim_factory.create_optimizer(actor_critic, lr)
        return ActorCriticModuleOpt(actor_critic, optim)


class _ActorAndCriticMixin(_ActorMixin):
    def __init__(
        self,
        actor_factory: ActorFactory,
        critic_factory: CriticFactory,
        optim_factory: OptimizerFactory,
        critic_use_action: bool,
    ):
        super().__init__(actor_factory, optim_factory)
        self.critic_module_opt_factory = CriticModuleOptFactory(
            critic_factory,
            optim_factory,
            critic_use_action,
        )

    def create_critic_module_opt(self, envs: Environments, device: TDevice, lr: float) -> ModuleOpt:
        return self.critic_module_opt_factory.create_module_opt(envs, device, lr)


class _ActorAndDualCriticsMixin(_ActorAndCriticMixin):
    def __init__(
        self,
        actor_factory: ActorFactory,
        critic_factory: CriticFactory,
        critic2_factory: CriticFactory,
        optim_factory: OptimizerFactory,
        critic_use_action: bool,
    ):
        super().__init__(actor_factory, critic_factory, optim_factory, critic_use_action)
        self.critic2_module_opt_factory = CriticModuleOptFactory(
            critic2_factory,
            optim_factory,
            critic_use_action,
        )

    def create_critic2_module_opt(
        self,
        envs: Environments,
        device: TDevice,
        lr: float,
    ) -> ModuleOpt:
        return self.critic2_module_opt_factory.create_module_opt(envs, device, lr)


class PPOAgentFactory(OnpolicyAgentFactory, _ActorCriticMixin):
    def __init__(
        self,
        params: PPOParams,
        sampling_config: RLSamplingConfig,
        actor_factory: ActorFactory,
        critic_factory: CriticFactory,
        optimizer_factory: OptimizerFactory,
        dist_fn: Callable[[TDistParams], torch.distributions.Distribution],
    ):
        super().__init__(sampling_config)
        _ActorCriticMixin.__init__(
            self,
            actor_factory,
            critic_factory,
            optimizer_factory,
            critic_use_action=False,
        )
        self.params = params
        self.dist_fn = dist_fn

    def create_policy(self, envs: Environments, device: TDevice) -> PPOPolicy:
        actor_critic = self.create_actor_critic_module_opt(envs, device, self.params.lr)
        kwargs = self.params.create_kwargs(
            ParamTransformerDrop("lr"),
            ParamTransformerLRScheduler(actor_critic.optim),
        )
        return PPOPolicy(
            actor=actor_critic.actor,
            critic=actor_critic.critic,
            optim=actor_critic.optim,
            dist_fn=self.dist_fn,
            action_space=envs.get_action_space(),
            **kwargs,
        )


class ParamTransformerAlpha(ParamTransformer):
    def __init__(self, envs: Environments, optim_factory: OptimizerFactory, device: TDevice):
        self.envs = envs
        self.optim_factory = optim_factory
        self.device = device

    def transform(self, kwargs: dict[str, Any]) -> None:
        key = "alpha"
        alpha = self.get(kwargs, key)
        if isinstance(alpha, AutoAlphaFactory):
            kwargs[key] = alpha.create_auto_alpha(self.envs, self.optim_factory, self.device)


class ParamTransformerMultiLRScheduler(ParamTransformer):
    def __init__(self, optim_key_list: list[tuple[torch.optim.Optimizer, str]]):
        self.optim_key_list = optim_key_list

    def transform(self, kwargs: dict[str, Any]) -> None:
        lr_schedulers = []
        for optim, lr_scheduler_factory_key in self.optim_key_list:
            lr_scheduler_factory: LRSchedulerFactory | None = self.get(
                kwargs,
                lr_scheduler_factory_key,
                drop=True,
            )
            if lr_scheduler_factory is not None:
                lr_schedulers.append(lr_scheduler_factory.create_scheduler(optim))
        match len(lr_schedulers):
            case 0:
                lr_scheduler = None
            case 1:
                lr_scheduler = lr_schedulers[0]
            case _:
                lr_scheduler = MultipleLRSchedulers(*lr_schedulers)
        kwargs["lr_scheduler"] = lr_scheduler


class SACAgentFactory(OffpolicyAgentFactory, _ActorAndDualCriticsMixin):
    def __init__(
        self,
        params: SACParams,
        sampling_config: RLSamplingConfig,
        actor_factory: ActorFactory,
        critic1_factory: CriticFactory,
        critic2_factory: CriticFactory,
        optim_factory: OptimizerFactory,
    ):
        super().__init__(sampling_config)
        _ActorAndDualCriticsMixin.__init__(
            self,
            actor_factory,
            critic1_factory,
            critic2_factory,
            optim_factory,
            critic_use_action=True,
        )
        self.params = params
        self.optim_factory = optim_factory

    def create_policy(self, envs: Environments, device: TDevice) -> BasePolicy:
        actor = self.create_actor_module_opt(envs, device, self.params.actor_lr)
        critic1 = self.create_critic_module_opt(envs, device, self.params.critic1_lr)
        critic2 = self.create_critic2_module_opt(envs, device, self.params.critic2_lr)
        kwargs = self.params.create_kwargs(
            ParamTransformerDrop("actor_lr", "critic1_lr", "critic2_lr"),
            ParamTransformerMultiLRScheduler(
                [
                    (actor.optim, "actor_lr_scheduler_factory"),
                    (critic1.optim, "critic1_lr_scheduler_factory"),
                    (critic2.optim, "critic2_lr_scheduler_factory"),
                ],
            ),
            ParamTransformerAlpha(envs, optim_factory=self.optim_factory, device=device),
        )
        return SACPolicy(
            actor=actor.module,
            actor_optim=actor.optim,
            critic=critic1.module,
            critic_optim=critic1.optim,
            critic2=critic2.module,
            critic2_optim=critic2.optim,
            action_space=envs.get_action_space(),
            observation_space=envs.get_observation_space(),
            **kwargs,
        )


class ParamTransformerNoiseFactory(ParamTransformer):
    def __init__(self, key: str, envs: Environments):
        self.key = key
        self.envs = envs

    def transform(self, kwargs: dict[str, Any]) -> None:
        value = kwargs[self.key]
        if isinstance(value, NoiseFactory):
            kwargs[self.key] = value.create_noise(self.envs)


class ParamTransformerFloatEnvParamFactory(ParamTransformer):
    def __init__(self, key: str, envs: Environments):
        self.key = key
        self.envs = envs

    def transform(self, kwargs: dict[str, Any]) -> None:
        value = kwargs[self.key]
        if isinstance(value, FloatEnvParamFactory):
            kwargs[self.key] = value.create_param(self.envs)


class TD3AgentFactory(OffpolicyAgentFactory, _ActorAndDualCriticsMixin):
    def __init__(
        self,
        params: TD3Params,
        sampling_config: RLSamplingConfig,
        actor_factory: ActorFactory,
        critic1_factory: CriticFactory,
        critic2_factory: CriticFactory,
        optim_factory: OptimizerFactory,
    ):
        super().__init__(sampling_config)
        _ActorAndDualCriticsMixin.__init__(
            self,
            actor_factory,
            critic1_factory,
            critic2_factory,
            optim_factory,
            critic_use_action=True,
        )
        self.params = params
        self.optim_factory = optim_factory

    def create_policy(self, envs: Environments, device: TDevice) -> BasePolicy:
        actor = self.create_actor_module_opt(envs, device, self.params.actor_lr)
        critic1 = self.create_critic_module_opt(envs, device, self.params.critic1_lr)
        critic2 = self.create_critic2_module_opt(envs, device, self.params.critic2_lr)
        kwargs = self.params.create_kwargs(
            ParamTransformerDrop("actor_lr", "critic1_lr", "critic2_lr"),
            ParamTransformerMultiLRScheduler(
                [
                    (actor.optim, "actor_lr_scheduler_factory"),
                    (critic1.optim, "critic1_lr_scheduler_factory"),
                    (critic2.optim, "critic2_lr_scheduler_factory"),
                ],
            ),
            ParamTransformerNoiseFactory("exploration_noise", envs),
            ParamTransformerFloatEnvParamFactory("policy_noise", envs),
            ParamTransformerFloatEnvParamFactory("noise_clip", envs),
        )
        return TD3Policy(
            actor=actor.module,
            actor_optim=actor.optim,
            critic=critic1.module,
            critic_optim=critic1.optim,
            critic2=critic2.module,
            critic2_optim=critic2.optim,
            action_space=envs.get_action_space(),
            observation_space=envs.get_observation_space(),
            **kwargs,
        )
