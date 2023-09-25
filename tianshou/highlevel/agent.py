import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Dict, Any, List, Tuple

import torch

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.exploration import BaseNoise
from tianshou.highlevel.config import RLSamplingConfig
from tianshou.highlevel.env import Environments
from tianshou.highlevel.logger import Logger
from tianshou.highlevel.module import ActorFactory, CriticFactory, TDevice
from tianshou.highlevel.optim import OptimizerFactory
from tianshou.highlevel.params.alpha import AutoAlphaFactory
from tianshou.highlevel.params.lr_scheduler import LRSchedulerFactory
from tianshou.highlevel.params.policy_params import PPOParams, ParamTransformer, SACParams
from tianshou.policy import BasePolicy, PPOPolicy, SACPolicy
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

    def transform(self, kwargs: Dict[str, Any]) -> None:
        for k in self.keys:
            del kwargs[k]


class ParamTransformerLRScheduler(ParamTransformer):
    def __init__(self, optim: torch.optim.Optimizer):
        self.optim = optim

    def transform(self, kwargs: Dict[str, Any]) -> None:
        factory: LRSchedulerFactory | None = self.get(kwargs, "lr_scheduler_factory", drop=True)
        kwargs["lr_scheduler"] = factory.create_scheduler(self.optim) if factory is not None else None


class PPOAgentFactory(OnpolicyAgentFactory):
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
        self.optimizer_factory = optimizer_factory
        self.critic_factory = critic_factory
        self.actor_factory = actor_factory
        self.config = params
        self.dist_fn = dist_fn

    def create_policy(self, envs: Environments, device: TDevice) -> PPOPolicy:
        actor = self.actor_factory.create_module(envs, device)
        critic = self.critic_factory.create_module(envs, device, use_action=False)
        actor_critic = ActorCritic(actor, critic)
        optim = self.optimizer_factory.create_optimizer(actor_critic, self.config.lr)
        kwargs = self.config.create_kwargs(
            ParamTransformerDrop("lr"),
            ParamTransformerLRScheduler(optim))
        return PPOPolicy(
            actor=actor,
            critic=critic,
            optim=optim,
            dist_fn=self.dist_fn,
            action_space=envs.get_action_space(),
            **kwargs
        )


class ParamTransformerAlpha(ParamTransformer):
    def __init__(self, envs: Environments, optim_factory: OptimizerFactory, device: TDevice):
        self.envs = envs
        self.optim_factory = optim_factory
        self.device = device

    def transform(self, kwargs: Dict[str, Any]) -> None:
        key = "alpha"
        alpha = self.get(kwargs, key)
        if isinstance(alpha, AutoAlphaFactory):
            kwargs[key] = alpha.create_auto_alpha(self.envs, self.optim_factory, self.device)


class ParamTransformerMultiLRScheduler(ParamTransformer):
    def __init__(self, optim_key_list: List[Tuple[torch.optim.Optimizer, str]]):
        self.optim_key_list = optim_key_list

    def transform(self, kwargs: Dict[str, Any]) -> None:
        lr_schedulers = []
        for optim, lr_scheduler_factory_key in self.optim_key_list:
            lr_scheduler_factory: LRSchedulerFactory | None = self.get(kwargs, lr_scheduler_factory_key, drop=True)
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


class SACAgentFactory(OffpolicyAgentFactory):
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
        self.critic2_factory = critic2_factory
        self.critic1_factory = critic1_factory
        self.actor_factory = actor_factory
        self.optim_factory = optim_factory
        self.params = params

    def create_policy(self, envs: Environments, device: TDevice) -> BasePolicy:
        actor = self.actor_factory.create_module(envs, device)
        critic1 = self.critic1_factory.create_module(envs, device, use_action=True)
        critic2 = self.critic2_factory.create_module(envs, device, use_action=True)
        actor_optim = self.optim_factory.create_optimizer(actor, lr=self.params.actor_lr)
        critic1_optim = self.optim_factory.create_optimizer(critic1, lr=self.params.critic1_lr)
        critic2_optim = self.optim_factory.create_optimizer(critic2, lr=self.params.critic2_lr)
        kwargs = self.params.create_kwargs(
            ParamTransformerDrop("actor_lr", "critic1_lr", "critic2_lr"),
            ParamTransformerMultiLRScheduler([
                (actor_optim, "actor_lr_scheduler_factory"),
                (critic1_optim, "critic1_lr_scheduler_factory"),
                (critic2_optim, "critic2_lr_scheduler_factory")]
            ),
            ParamTransformerAlpha(envs, optim_factory=self.optim_factory, device=device))
        return SACPolicy(
            actor=actor,
            actor_optim=actor_optim,
            critic=critic1,
            critic_optim=critic1_optim,
            critic2=critic2,
            critic2_optim=critic2_optim,
            action_space=envs.get_action_space(),
            observation_space=envs.get_observation_space(),
            **kwargs
        )
