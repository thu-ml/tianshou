import logging
import typing
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, cast

import gymnasium
import torch
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
    A2C,
    DDPG,
    NPG,
    PPO,
    TD3,
    TRPO,
    Algorithm,
    DeepQLearning,
    DiscreteSACPolicy,
    IQNPolicy,
    REDQPolicy,
    Reinforce,
    SACPolicy,
)
from tianshou.policy.base import (
    OffPolicyAlgorithm,
    OnPolicyAlgorithm,
    Policy,
    RandomActionPolicy,
)
from tianshou.policy.modelfree.ddpg import DDPGPolicy
from tianshou.policy.modelfree.dqn import DQNPolicy
from tianshou.policy.modelfree.pg import ActorPolicy
from tianshou.trainer import BaseTrainer, OffpolicyTrainer, OnpolicyTrainer
from tianshou.trainer.base import OffPolicyTrainingConfig, OnPolicyTrainingConfig
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
TAlgorithm = TypeVar("TAlgorithm", bound=Algorithm)
TPolicy = TypeVar("TPolicy", bound=Policy)
log = logging.getLogger(__name__)


class AlgorithmFactory(ABC, ToStringMixin):
    """Factory for the creation of an :class:`Algorithm` instance, its policy, trainer as well as collectors."""

    def __init__(self, sampling_config: SamplingConfig, optim_factory: OptimizerFactory):
        self.sampling_config = sampling_config
        self.optim_factory = optim_factory
        self.policy_wrapper_factory: PolicyWrapperFactory | None = None
        self.trainer_callbacks: TrainerCallbacks = TrainerCallbacks()

    def create_train_test_collector(
        self,
        policy: Algorithm,
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

    @staticmethod
    def _create_policy(
        constructor: type[TPolicy], params_dict: dict, policy_params: list[str], **kwargs
    ) -> TPolicy:
        params = {p: params_dict.pop(p) for p in policy_params}
        return constructor(**params, **kwargs)

    @abstractmethod
    def _create_algorithm(self, envs: Environments, device: TDevice) -> Algorithm:
        pass

    def create_algorithm(self, envs: Environments, device: TDevice) -> Algorithm:
        policy = self._create_algorithm(envs, device)
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


class OnPolicyAlgorithmFactory(AlgorithmFactory, ABC):
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
        algorithm = cast(OnPolicyAlgorithm, world.policy)
        return algorithm.create_trainer(
            OnPolicyTrainingConfig(
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
        )


class OffPolicyAlgorithmFactory(AlgorithmFactory, ABC):
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
        algorithm = cast(OffPolicyAlgorithm, world.policy)
        return algorithm.create_trainer(
            OffPolicyTrainingConfig(
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
        )


class RandomActionAlgorithmFactory(OnPolicyAlgorithmFactory):
    def _create_algorithm(self, envs: Environments, device: TDevice) -> RandomActionPolicy:
        return RandomActionPolicy(envs.get_action_space())


class ReinforceAlgorithmFactory(OnPolicyAlgorithmFactory):
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

    def _create_algorithm(self, envs: Environments, device: TDevice) -> Reinforce:
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
        policy = self._create_policy(
            ActorPolicy,
            kwargs,
            ["action_scaling", "action_bound_method", "deterministic_eval"],
            actor=actor.module,
            dist_fn=dist_fn,
            action_space=envs.get_action_space(),
            observation_space=envs.get_observation_space(),
        )
        return Reinforce(
            policy=policy,
            optim=actor.optim,
            **kwargs,
        )


class ActorCriticAlgorithmFactory(
    Generic[TActorCriticParams, TAlgorithm],
    OnPolicyAlgorithmFactory,
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
    def _get_algorithm_class(self) -> type[TAlgorithm]:
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
        kwargs["observation_space"] = envs.get_observation_space()
        kwargs["dist_fn"] = self.actor_factory.create_dist_fn(envs)
        return kwargs

    def _create_algorithm(self, envs: Environments, device: TDevice) -> TAlgorithm:
        params = self._create_kwargs(envs, device)
        policy = self._create_policy(
            ActorPolicy,
            params,
            [
                "actor",
                "dist_fn",
                "action_space",
                "deterministic_eval",
                "observation_space",
                "action_scaling",
                "action_bound_method",
            ],
        )
        algorithm_class = self._get_algorithm_class()
        return algorithm_class(policy=policy, **params)


class A2CAlgorithmFactory(ActorCriticAlgorithmFactory[A2CParams, A2C]):
    def _get_algorithm_class(self) -> type[A2C]:
        return A2C


class PPOAlgorithmFactory(ActorCriticAlgorithmFactory[PPOParams, PPO]):
    def _get_algorithm_class(self) -> type[PPO]:
        return PPO


class NPGAlgorithmFactory(ActorCriticAlgorithmFactory[NPGParams, NPG]):
    def _get_algorithm_class(self) -> type[NPG]:
        return NPG


class TRPOAlgorithmFactory(ActorCriticAlgorithmFactory[TRPOParams, TRPO]):
    def _get_algorithm_class(self) -> type[TRPO]:
        return TRPO


class DiscreteCriticOnlyAlgorithmFactory(
    OffPolicyAlgorithmFactory,
    Generic[TDiscreteCriticOnlyParams, TAlgorithm],
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
    def _get_algorithm_class(self) -> type[TAlgorithm]:
        pass

    @abstractmethod
    def _create_discrete_critic_only_policy(
        self,
        model: torch.nn.Module,
        params: dict,
        action_space: gymnasium.spaces.Discrete,
        observation_space: gymnasium.spaces.Space,
    ) -> TPolicy:
        pass

    @typing.no_type_check
    def _create_algorithm(self, envs: Environments, device: TDevice) -> TAlgorithm:
        model = self.model_factory.create_module(envs, device)
        optim = self.optim_factory.create_optimizer(model, self.params.lr)
        params_dict = self.params.create_kwargs(
            ParamTransformerData(
                envs=envs,
                device=device,
                optim=optim,
                optim_factory=self.optim_factory,
            ),
        )
        envs.get_type().assert_discrete(self)
        action_space = cast(gymnasium.spaces.Discrete, envs.get_action_space())
        policy = self._create_discrete_critic_only_policy(
            model, params_dict, action_space, envs.get_observation_space()
        )
        algorithm_class = self._get_algorithm_class()
        return algorithm_class(
            policy=policy,
            optim=optim,
            **params_dict,
        )


class DeepQLearningAlgorithmFactory(DiscreteCriticOnlyAlgorithmFactory[DQNParams, DeepQLearning]):
    def _create_discrete_critic_only_policy(
        self,
        model: torch.nn.Module,
        params: dict,
        action_space: gymnasium.spaces.Discrete,
        observation_space: gymnasium.spaces.Space,
    ) -> TPolicy:
        return self._create_policy(
            constructor=DQNPolicy,
            params_dict=params,
            policy_params=[],
            model=model,
            action_space=action_space,
            observation_space=observation_space,
        )

    def _get_algorithm_class(self) -> type[DeepQLearning]:
        return DeepQLearning


class IQNAlgorithmFactory(DiscreteCriticOnlyAlgorithmFactory[IQNParams, IQNPolicy]):
    def _get_algorithm_class(self) -> type[IQNPolicy]:
        return IQNPolicy


class DDPGAlgorithmFactory(OffPolicyAlgorithmFactory):
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

    def _create_algorithm(self, envs: Environments, device: TDevice) -> Algorithm:
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
        policy = self._create_policy(
            DDPGPolicy,
            kwargs,
            ["action_scaling", "action_bound_method"],
            actor=actor.module,
            action_space=envs.get_action_space(),
            observation_space=envs.get_observation_space(),
        )
        return DDPG(
            policy=policy,
            policy_optim=actor.optim,
            critic=critic.module,
            critic_optim=critic.optim,
            **kwargs,
        )


class REDQAlgorithmFactory(OffPolicyAlgorithmFactory):
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

    def _create_algorithm(self, envs: Environments, device: TDevice) -> Algorithm:
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


class ActorDualCriticsAlgorithmFactory(
    OffPolicyAlgorithmFactory,
    Generic[TActorDualCriticsParams, TAlgorithm],
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
    def _get_algorithm_class(self) -> type[TAlgorithm]:
        pass

    def _get_discrete_last_size_use_action_shape(self) -> bool:
        return True

    @staticmethod
    def _get_critic_use_action(envs: Environments) -> bool:
        return envs.get_type().is_continuous()

    @typing.no_type_check
    def _create_algorithm(self, envs: Environments, device: TDevice) -> TAlgorithm:
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
        policy_class = self._get_algorithm_class()
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


class SACAlgorithmFactory(ActorDualCriticsAlgorithmFactory[SACParams, SACPolicy]):
    def _get_algorithm_class(self) -> type[SACPolicy]:
        return SACPolicy


class DiscreteSACAlgorithmFactory(
    ActorDualCriticsAlgorithmFactory[DiscreteSACParams, DiscreteSACPolicy]
):
    def _get_algorithm_class(self) -> type[DiscreteSACPolicy]:
        return DiscreteSACPolicy


class TD3AlgorithmFactory(ActorDualCriticsAlgorithmFactory[TD3Params, TD3]):
    def _get_algorithm_class(self) -> type[TD3]:
        return TD3
