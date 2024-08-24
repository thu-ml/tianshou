import logging
import typing
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, cast

import gymnasium
from sensai.util.string import ToStringMixin

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.data.collector import BaseCollector, CollectStats
from tianshou.highlevel.config import SamplingConfig
from tianshou.highlevel.env import Environments
from tianshou.highlevel.module.actor import (
    ActorFactory,
)
from tianshou.highlevel.module.core import (
    ModuleFactory,
    TDevice,
)
from tianshou.highlevel.module.critic import CriticEnsembleFactory, CriticFactory
from tianshou.highlevel.module.module_opt import (
    ActorCriticOpt,
)
from tianshou.highlevel.optim import OptimizerFactory
from tianshou.highlevel.params.policy_params import (
    A2CParams,
    DDPGParams,
    DiscreteSACParams,
    DQNParams,
    IQNParams,
    NPGParams,
    Params,
    ParamsMixinActorAndDualCritics,
    ParamsMixinLearningRateWithScheduler,
    ParamTransformerData,
    PGParams,
    PPOParams,
    REDQParams,
    SACParams,
    TD3Params,
    TRPOParams,
)
from tianshou.highlevel.params.policy_wrapper import PolicyWrapperFactory
from tianshou.highlevel.persistence import PolicyPersistence
from tianshou.highlevel.trainer import TrainerCallbacks, TrainingContext
from tianshou.highlevel.world import World
from tianshou.policy import (
    A2CPolicy,
    BasePolicy,
    DDPGPolicy,
    DiscreteSACPolicy,
    DQNPolicy,
    IQNPolicy,
    NPGPolicy,
    PGPolicy,
    PPOPolicy,
    REDQPolicy,
    SACPolicy,
    TD3Policy,
    TRPOPolicy,
)
from tianshou.policy.base import RandomActionPolicy
from tianshou.trainer import BaseTrainer, OffpolicyTrainer, OnpolicyTrainer
from tianshou.utils.net.common import ActorCritic

CHECKPOINT_DICT_KEY_MODEL = "model"
CHECKPOINT_DICT_KEY_OBS_RMS = "obs_rms"
TParams = TypeVar("TParams", bound=Params)
TActorCriticParams = TypeVar(
    "TActorCriticParams",
    bound=Params | ParamsMixinLearningRateWithScheduler,
)
TActorDualCriticsParams = TypeVar(
    "TActorDualCriticsParams",
    bound=Params | ParamsMixinActorAndDualCritics,
)
TDiscreteCriticOnlyParams = TypeVar(
    "TDiscreteCriticOnlyParams",
    bound=Params | ParamsMixinLearningRateWithScheduler,
)
TPolicy = TypeVar("TPolicy", bound=BasePolicy)
log = logging.getLogger(__name__)


class AgentFactory(ABC, ToStringMixin):
    """Factory for the creation of an agent's policy, its trainer as well as collectors."""

    def __init__(self, sampling_config: SamplingConfig, optim_factory: OptimizerFactory):
        self.sampling_config = sampling_config
        self.optim_factory = optim_factory
        self.policy_wrapper_factory: PolicyWrapperFactory | None = None
        self.trainer_callbacks: TrainerCallbacks = TrainerCallbacks()

    def create_train_test_collector(
        self,
        policy: BasePolicy,
        envs: Environments,
        reset_collectors: bool = True,
    ) -> tuple[BaseCollector, BaseCollector]:
        """:param policy:
        :param envs:
        :param reset_collectors: Whether to reset the collectors before returning them.
            Setting to True means that the envs will be reset as well.
        :return:
        """
        buffer_size = self.sampling_config.buffer_size
        train_envs = envs.train_envs
        buffer: ReplayBuffer
        if len(train_envs) > 1:
            buffer = VectorReplayBuffer(
                buffer_size,
                len(train_envs),
                stack_num=self.sampling_config.replay_buffer_stack_num,
                save_only_last_obs=self.sampling_config.replay_buffer_save_only_last_obs,
                ignore_obs_next=self.sampling_config.replay_buffer_ignore_obs_next,
            )
        else:
            buffer = ReplayBuffer(
                buffer_size,
                stack_num=self.sampling_config.replay_buffer_stack_num,
                save_only_last_obs=self.sampling_config.replay_buffer_save_only_last_obs,
                ignore_obs_next=self.sampling_config.replay_buffer_ignore_obs_next,
            )
        train_collector = Collector[CollectStats](
            policy,
            train_envs,
            buffer,
            exploration_noise=True,
        )
        test_collector = Collector[CollectStats](policy, envs.test_envs)
        if reset_collectors:
            train_collector.reset()
            test_collector.reset()
        return train_collector, test_collector

    def set_policy_wrapper_factory(
        self,
        policy_wrapper_factory: PolicyWrapperFactory | None,
    ) -> None:
        self.policy_wrapper_factory = policy_wrapper_factory

    def set_trainer_callbacks(self, callbacks: TrainerCallbacks) -> None:
        self.trainer_callbacks = callbacks

    @abstractmethod
    def _create_policy(self, envs: Environments, device: TDevice) -> BasePolicy:
        pass

    def create_policy(self, envs: Environments, device: TDevice) -> BasePolicy:
        policy = self._create_policy(envs, device)
        if self.policy_wrapper_factory is not None:
            policy = self.policy_wrapper_factory.create_wrapped_policy(
                policy,
                envs,
                self.optim_factory,
                device,
            )
        return policy

    @abstractmethod
    def create_trainer(self, world: World, policy_persistence: PolicyPersistence) -> BaseTrainer:
        pass


class OnPolicyAgentFactory(AgentFactory, ABC):
    def create_trainer(
        self,
        world: World,
        policy_persistence: PolicyPersistence,
    ) -> OnpolicyTrainer:
        sampling_config = self.sampling_config
        callbacks = self.trainer_callbacks
        context = TrainingContext(world.policy, world.envs, world.logger)
        train_fn = (
            callbacks.epoch_train_callback.get_trainer_fn(context)
            if callbacks.epoch_train_callback
            else None
        )
        test_fn = (
            callbacks.epoch_test_callback.get_trainer_fn(context)
            if callbacks.epoch_test_callback
            else None
        )
        stop_fn = (
            callbacks.epoch_stop_callback.get_trainer_fn(context)
            if callbacks.epoch_stop_callback
            else None
        )
        return OnpolicyTrainer(
            policy=world.policy,
            train_collector=world.train_collector,
            test_collector=world.test_collector,
            max_epoch=sampling_config.num_epochs,
            step_per_epoch=sampling_config.step_per_epoch,
            repeat_per_collect=sampling_config.repeat_per_collect,
            episode_per_test=sampling_config.num_test_episodes,
            batch_size=sampling_config.batch_size,
            step_per_collect=sampling_config.step_per_collect,
            save_best_fn=policy_persistence.get_save_best_fn(world),
            save_checkpoint_fn=policy_persistence.get_save_checkpoint_fn(world),
            logger=world.logger,
            test_in_train=False,
            train_fn=train_fn,
            test_fn=test_fn,
            stop_fn=stop_fn,
            verbose=False,
        )


class OffPolicyAgentFactory(AgentFactory, ABC):
    def create_trainer(
        self,
        world: World,
        policy_persistence: PolicyPersistence,
    ) -> OffpolicyTrainer:
        sampling_config = self.sampling_config
        callbacks = self.trainer_callbacks
        context = TrainingContext(world.policy, world.envs, world.logger)
        train_fn = (
            callbacks.epoch_train_callback.get_trainer_fn(context)
            if callbacks.epoch_train_callback
            else None
        )
        test_fn = (
            callbacks.epoch_test_callback.get_trainer_fn(context)
            if callbacks.epoch_test_callback
            else None
        )
        stop_fn = (
            callbacks.epoch_stop_callback.get_trainer_fn(context)
            if callbacks.epoch_stop_callback
            else None
        )
        return OffpolicyTrainer(
            policy=world.policy,
            train_collector=world.train_collector,
            test_collector=world.test_collector,
            max_epoch=sampling_config.num_epochs,
            step_per_epoch=sampling_config.step_per_epoch,
            step_per_collect=sampling_config.step_per_collect,
            episode_per_test=sampling_config.num_test_episodes,
            batch_size=sampling_config.batch_size,
            save_best_fn=policy_persistence.get_save_best_fn(world),
            logger=world.logger,
            update_per_step=sampling_config.update_per_step,
            test_in_train=False,
            train_fn=train_fn,
            test_fn=test_fn,
            stop_fn=stop_fn,
            verbose=False,
        )


class RandomActionAgentFactory(OnPolicyAgentFactory):
    def _create_policy(self, envs: Environments, device: TDevice) -> RandomActionPolicy:
        return RandomActionPolicy(envs.get_action_space())


class PGAgentFactory(OnPolicyAgentFactory):
    def __init__(
        self,
        params: PGParams,
        sampling_config: SamplingConfig,
        actor_factory: ActorFactory,
        optim_factory: OptimizerFactory,
    ):
        super().__init__(sampling_config, optim_factory)
        self.params = params
        self.actor_factory = actor_factory
        self.optim_factory = optim_factory

    def _create_policy(self, envs: Environments, device: TDevice) -> PGPolicy:
        actor = self.actor_factory.create_module_opt(
            envs,
            device,
            self.optim_factory,
            self.params.lr,
        )
        kwargs = self.params.create_kwargs(
            ParamTransformerData(
                envs=envs,
                device=device,
                optim=actor.optim,
                optim_factory=self.optim_factory,
            ),
        )
        dist_fn = self.actor_factory.create_dist_fn(envs)
        assert dist_fn is not None
        return PGPolicy(
            actor=actor.module,
            optim=actor.optim,
            action_space=envs.get_action_space(),
            observation_space=envs.get_observation_space(),
            dist_fn=dist_fn,
            **kwargs,
        )


class ActorCriticAgentFactory(
    Generic[TActorCriticParams, TPolicy],
    OnPolicyAgentFactory,
    ABC,
):
    def __init__(
        self,
        params: TActorCriticParams,
        sampling_config: SamplingConfig,
        actor_factory: ActorFactory,
        critic_factory: CriticFactory,
        optimizer_factory: OptimizerFactory,
    ):
        super().__init__(sampling_config, optim_factory=optimizer_factory)
        self.params = params
        self.actor_factory = actor_factory
        self.critic_factory = critic_factory
        self.optim_factory = optimizer_factory
        self.critic_use_action = False

    @abstractmethod
    def _get_policy_class(self) -> type[TPolicy]:
        pass

    def create_actor_critic_module_opt(
        self,
        envs: Environments,
        device: TDevice,
        lr: float,
    ) -> ActorCriticOpt:
        actor = self.actor_factory.create_module(envs, device)
        critic = self.critic_factory.create_module(envs, device, use_action=self.critic_use_action)
        actor_critic = ActorCritic(actor, critic)
        optim = self.optim_factory.create_optimizer(actor_critic, lr)
        return ActorCriticOpt(actor_critic, optim)

    @typing.no_type_check
    def _create_kwargs(self, envs: Environments, device: TDevice) -> dict[str, Any]:
        actor_critic = self.create_actor_critic_module_opt(envs, device, self.params.lr)
        kwargs = self.params.create_kwargs(
            ParamTransformerData(
                envs=envs,
                device=device,
                optim_factory=self.optim_factory,
                optim=actor_critic.optim,
            ),
        )
        kwargs["actor"] = actor_critic.actor
        kwargs["critic"] = actor_critic.critic
        kwargs["optim"] = actor_critic.optim
        kwargs["action_space"] = envs.get_action_space()
        kwargs["dist_fn"] = self.actor_factory.create_dist_fn(envs)
        return kwargs

    def _create_policy(self, envs: Environments, device: TDevice) -> TPolicy:
        policy_class = self._get_policy_class()
        return policy_class(**self._create_kwargs(envs, device))


class A2CAgentFactory(ActorCriticAgentFactory[A2CParams, A2CPolicy]):
    def _get_policy_class(self) -> type[A2CPolicy]:
        return A2CPolicy


class PPOAgentFactory(ActorCriticAgentFactory[PPOParams, PPOPolicy]):
    def _get_policy_class(self) -> type[PPOPolicy]:
        return PPOPolicy


class NPGAgentFactory(ActorCriticAgentFactory[NPGParams, NPGPolicy]):
    def _get_policy_class(self) -> type[NPGPolicy]:
        return NPGPolicy


class TRPOAgentFactory(ActorCriticAgentFactory[TRPOParams, TRPOPolicy]):
    def _get_policy_class(self) -> type[TRPOPolicy]:
        return TRPOPolicy


class DiscreteCriticOnlyAgentFactory(
    OffPolicyAgentFactory,
    Generic[TDiscreteCriticOnlyParams, TPolicy],
):
    def __init__(
        self,
        params: TDiscreteCriticOnlyParams,
        sampling_config: SamplingConfig,
        model_factory: ModuleFactory,
        optim_factory: OptimizerFactory,
    ):
        super().__init__(sampling_config, optim_factory)
        self.params = params
        self.model_factory = model_factory
        self.optim_factory = optim_factory

    @abstractmethod
    def _get_policy_class(self) -> type[TPolicy]:
        pass

    @typing.no_type_check
    def _create_policy(self, envs: Environments, device: TDevice) -> TPolicy:
        model = self.model_factory.create_module(envs, device)
        optim = self.optim_factory.create_optimizer(model, self.params.lr)
        kwargs = self.params.create_kwargs(
            ParamTransformerData(
                envs=envs,
                device=device,
                optim=optim,
                optim_factory=self.optim_factory,
            ),
        )
        envs.get_type().assert_discrete(self)
        action_space = cast(gymnasium.spaces.Discrete, envs.get_action_space())
        policy_class = self._get_policy_class()
        return policy_class(
            model=model,
            optim=optim,
            action_space=action_space,
            observation_space=envs.get_observation_space(),
            **kwargs,
        )


class DQNAgentFactory(DiscreteCriticOnlyAgentFactory[DQNParams, DQNPolicy]):
    def _get_policy_class(self) -> type[DQNPolicy]:
        return DQNPolicy


class IQNAgentFactory(DiscreteCriticOnlyAgentFactory[IQNParams, IQNPolicy]):
    def _get_policy_class(self) -> type[IQNPolicy]:
        return IQNPolicy


class DDPGAgentFactory(OffPolicyAgentFactory):
    def __init__(
        self,
        params: DDPGParams,
        sampling_config: SamplingConfig,
        actor_factory: ActorFactory,
        critic_factory: CriticFactory,
        optim_factory: OptimizerFactory,
    ):
        super().__init__(sampling_config, optim_factory)
        self.critic_factory = critic_factory
        self.actor_factory = actor_factory
        self.params = params
        self.optim_factory = optim_factory

    def _create_policy(self, envs: Environments, device: TDevice) -> BasePolicy:
        actor = self.actor_factory.create_module_opt(
            envs,
            device,
            self.optim_factory,
            self.params.actor_lr,
        )
        critic = self.critic_factory.create_module_opt(
            envs,
            device,
            True,
            self.optim_factory,
            self.params.critic_lr,
        )
        kwargs = self.params.create_kwargs(
            ParamTransformerData(
                envs=envs,
                device=device,
                optim_factory=self.optim_factory,
                actor=actor,
                critic1=critic,
            ),
        )
        return DDPGPolicy(
            actor=actor.module,
            actor_optim=actor.optim,
            critic=critic.module,
            critic_optim=critic.optim,
            action_space=envs.get_action_space(),
            observation_space=envs.get_observation_space(),
            **kwargs,
        )


class REDQAgentFactory(OffPolicyAgentFactory):
    def __init__(
        self,
        params: REDQParams,
        sampling_config: SamplingConfig,
        actor_factory: ActorFactory,
        critic_ensemble_factory: CriticEnsembleFactory,
        optim_factory: OptimizerFactory,
    ):
        super().__init__(sampling_config, optim_factory)
        self.critic_ensemble_factory = critic_ensemble_factory
        self.actor_factory = actor_factory
        self.params = params
        self.optim_factory = optim_factory

    def _create_policy(self, envs: Environments, device: TDevice) -> BasePolicy:
        envs.get_type().assert_continuous(self)
        actor = self.actor_factory.create_module_opt(
            envs,
            device,
            self.optim_factory,
            self.params.actor_lr,
        )
        critic_ensemble = self.critic_ensemble_factory.create_module_opt(
            envs,
            device,
            self.params.ensemble_size,
            True,
            self.optim_factory,
            self.params.critic_lr,
        )
        kwargs = self.params.create_kwargs(
            ParamTransformerData(
                envs=envs,
                device=device,
                optim_factory=self.optim_factory,
                actor=actor,
                critic1=critic_ensemble,
            ),
        )
        action_space = cast(gymnasium.spaces.Box, envs.get_action_space())
        return REDQPolicy(
            actor=actor.module,
            actor_optim=actor.optim,
            critic=critic_ensemble.module,
            critic_optim=critic_ensemble.optim,
            action_space=action_space,
            observation_space=envs.get_observation_space(),
            **kwargs,
        )


class ActorDualCriticsAgentFactory(
    OffPolicyAgentFactory,
    Generic[TActorDualCriticsParams, TPolicy],
    ABC,
):
    def __init__(
        self,
        params: TActorDualCriticsParams,
        sampling_config: SamplingConfig,
        actor_factory: ActorFactory,
        critic1_factory: CriticFactory,
        critic2_factory: CriticFactory,
        optim_factory: OptimizerFactory,
    ):
        super().__init__(sampling_config, optim_factory)
        self.params = params
        self.actor_factory = actor_factory
        self.critic1_factory = critic1_factory
        self.critic2_factory = critic2_factory
        self.optim_factory = optim_factory

    @abstractmethod
    def _get_policy_class(self) -> type[TPolicy]:
        pass

    def _get_discrete_last_size_use_action_shape(self) -> bool:
        return True

    @staticmethod
    def _get_critic_use_action(envs: Environments) -> bool:
        return envs.get_type().is_continuous()

    @typing.no_type_check
    def _create_policy(self, envs: Environments, device: TDevice) -> TPolicy:
        actor = self.actor_factory.create_module_opt(
            envs,
            device,
            self.optim_factory,
            self.params.actor_lr,
        )
        use_action_shape = self._get_discrete_last_size_use_action_shape()
        critic_use_action = self._get_critic_use_action(envs)
        critic1 = self.critic1_factory.create_module_opt(
            envs,
            device,
            critic_use_action,
            self.optim_factory,
            self.params.critic1_lr,
            discrete_last_size_use_action_shape=use_action_shape,
        )
        critic2 = self.critic2_factory.create_module_opt(
            envs,
            device,
            critic_use_action,
            self.optim_factory,
            self.params.critic2_lr,
            discrete_last_size_use_action_shape=use_action_shape,
        )
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
        policy_class = self._get_policy_class()
        return policy_class(
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


class SACAgentFactory(ActorDualCriticsAgentFactory[SACParams, SACPolicy]):
    def _get_policy_class(self) -> type[SACPolicy]:
        return SACPolicy


class DiscreteSACAgentFactory(ActorDualCriticsAgentFactory[DiscreteSACParams, DiscreteSACPolicy]):
    def _get_policy_class(self) -> type[DiscreteSACPolicy]:
        return DiscreteSACPolicy


class TD3AgentFactory(ActorDualCriticsAgentFactory[TD3Params, TD3Policy]):
    def _get_policy_class(self) -> type[TD3Policy]:
        return TD3Policy
