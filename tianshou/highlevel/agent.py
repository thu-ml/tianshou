import os
from abc import abstractmethod, ABC
from typing import Callable

import torch

from tianshou.config import RLSamplingConfig, PGConfig, PPOConfig, RLAgentConfig, NNConfig
from tianshou.data import VectorReplayBuffer, ReplayBuffer, Collector
from tianshou.highlevel.env import Environments
from tianshou.highlevel.logger import Logger
from tianshou.highlevel.module import ActorFactory, CriticFactory, TDevice
from tianshou.highlevel.optim import OptimizerFactory, LRSchedulerFactory
from tianshou.policy import BasePolicy, PPOPolicy
from tianshou.trainer import BaseTrainer, OnpolicyTrainer
from tianshou.utils.net.common import ActorCritic


CHECKPOINT_DICT_KEY_MODEL = "model"
CHECKPOINT_DICT_KEY_OBS_RMS = "obs_rms"


class AgentFactory(ABC):
    @abstractmethod
    def create_policy(self, envs: Environments, device: TDevice) -> BasePolicy:
        pass

    @staticmethod
    def _create_save_best_fn(envs: Environments, log_path: str) -> Callable:
        def save_best_fn(pol: torch.nn.Module):
            state = {"model": pol.state_dict(), "obs_rms": envs.train_envs.get_obs_rms()}
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
    def create_train_test_collector(self,
            policy: BasePolicy,
            envs: Environments):
        pass

    @abstractmethod
    def create_trainer(self, policy: BasePolicy, train_collector: Collector, test_collector: Collector,
            envs: Environments, logger: Logger) -> BaseTrainer:
        pass


class OnpolicyAgentFactory(AgentFactory, ABC):
    def __init__(self, sampling_config: RLSamplingConfig):
        self.sampling_config = sampling_config

    def create_train_test_collector(self,
            policy: BasePolicy,
            envs: Environments):
        buffer_size = self.sampling_config.buffer_size
        train_envs = envs.train_envs
        if len(train_envs) > 1:
            buffer = VectorReplayBuffer(buffer_size, len(train_envs))
        else:
            buffer = ReplayBuffer(buffer_size)
        train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
        test_collector = Collector(policy, envs.test_envs)
        return train_collector, test_collector

    def create_trainer(self, policy: BasePolicy, train_collector: Collector, test_collector: Collector,
            envs: Environments, logger: Logger) -> OnpolicyTrainer:
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


class PPOAgentFactory(OnpolicyAgentFactory):
    def __init__(self, general_config: RLAgentConfig,
            pg_config: PGConfig,
            ppo_config: PPOConfig,
            sampling_config: RLSamplingConfig,
            nn_config: NNConfig,
            actor_factory: ActorFactory,
            critic_factory: CriticFactory,
            optimizer_factory: OptimizerFactory,
            dist_fn,
            lr_scheduler_factory: LRSchedulerFactory):
        super().__init__(sampling_config)
        self.optimizer_factory = optimizer_factory
        self.critic_factory = critic_factory
        self.actor_factory = actor_factory
        self.ppo_config = ppo_config
        self.pg_config = pg_config
        self.general_config = general_config
        self.lr_scheduler_factory = lr_scheduler_factory
        self.dist_fn = dist_fn
        self.nn_config = nn_config

    def create_policy(self, envs: Environments, device: TDevice) -> PPOPolicy:
        actor = self.actor_factory.create_module(envs, device)
        critic = self.critic_factory.create_module(envs, device)
        actor_critic = ActorCritic(actor, critic)
        optim = self.optimizer_factory.create_optimizer(actor_critic)
        lr_scheduler = self.lr_scheduler_factory.create_scheduler(optim)
        return PPOPolicy(
            # nn-stuff
            actor,
            critic,
            optim,
            dist_fn=self.dist_fn,
            lr_scheduler=lr_scheduler,
            # env-stuff
            action_space=envs.get_action_space(),
            action_scaling=True,
            # general_config
            discount_factor=self.general_config.gamma,
            gae_lambda=self.general_config.gae_lambda,
            reward_normalization=self.general_config.rew_norm,
            action_bound_method=self.general_config.action_bound_method,
            # pg_config
            max_grad_norm=self.pg_config.max_grad_norm,
            vf_coef=self.pg_config.vf_coef,
            ent_coef=self.pg_config.ent_coef,
            # ppo_config
            eps_clip=self.ppo_config.eps_clip,
            value_clip=self.ppo_config.value_clip,
            dual_clip=self.ppo_config.dual_clip,
            advantage_normalization=self.ppo_config.norm_adv,
            recompute_advantage=self.ppo_config.recompute_adv,
        )
