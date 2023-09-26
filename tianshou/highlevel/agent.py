import os
from abc import ABC, abstractmethod
from collections.abc import Callable

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
from tianshou.highlevel.params.policy_params import (
    ParamTransformerData,
    PPOParams,
    SACParams,
    TD3Params,
)
from tianshou.policy import BasePolicy, PPOPolicy, SACPolicy, TD3Policy
from tianshou.policy.modelfree.pg import TDistParams
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
            ParamTransformerData(
                envs=envs,
                device=device,
                optim_factory=self.optim_factory,
                optim=actor_critic.optim,
            ),
        )
        return PPOPolicy(
            actor=actor_critic.actor,
            critic=actor_critic.critic,
            optim=actor_critic.optim,
            dist_fn=self.dist_fn,
            action_space=envs.get_action_space(),
            **kwargs,
        )


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
            ParamTransformerData(
                envs=envs,
                device=device,
                optim_factory=self.optim_factory,
                actor=actor,
                critic1=critic1,
                critic2=critic2,
            ),
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
            ParamTransformerData(
                envs=envs,
                device=device,
                optim_factory=self.optim_factory,
                actor=actor,
                critic1=critic1,
                critic2=critic2,
            ),
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
