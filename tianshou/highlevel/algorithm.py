import logging
import typing
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, cast

import gymnasium
import torch
from sensai.util.string import ToStringMixin

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.data.collector import BaseCollector, CollectStats
from tianshou.highlevel.config import (
    OffPolicyTrainingConfig,
    OnPolicyTrainingConfig,
    TrainingConfig,
)
from tianshou.highlevel.env import Environments
from tianshou.highlevel.module.actor import (
    ActorFactory,
)
from tianshou.highlevel.module.core import (
    ModuleFactory,
    TDevice,
)
from tianshou.highlevel.module.critic import CriticEnsembleFactory, CriticFactory
from tianshou.highlevel.optim import OptimizerFactoryFactory
from tianshou.highlevel.params.policy_params import (
    A2CParams,
    DDPGParams,
    DiscreteSACParams,
    DQNParams,
    IQNParams,
    NPGParams,
    Params,
    ParamsMixinActorAndDualCritics,
    ParamsMixinSingleModel,
    ParamTransformerData,
    PPOParams,
    REDQParams,
    ReinforceParams,
    SACParams,
    TD3Params,
    TRPOParams,
)
from tianshou.highlevel.params.policy_wrapper import AlgorithmWrapperFactory
from tianshou.highlevel.persistence import PolicyPersistence
from tianshou.highlevel.trainer import TrainerCallbacks, TrainingContext
from tianshou.highlevel.world import World
from tianshou.policy import (
    A2C,
    DDPG,
    DQN,
    IQN,
    NPG,
    PPO,
    REDQ,
    SAC,
    TD3,
    TRPO,
    Algorithm,
    DiscreteSAC,
    Reinforce,
)
from tianshou.policy.base import (
    OffPolicyAlgorithm,
    OnPolicyAlgorithm,
    Policy,
)
from tianshou.policy.modelfree.ddpg import ContinuousDeterministicPolicy
from tianshou.policy.modelfree.discrete_sac import DiscreteSACPolicy
from tianshou.policy.modelfree.dqn import DiscreteQLearningPolicy
from tianshou.policy.modelfree.iqn import IQNPolicy
from tianshou.policy.modelfree.pg import ActorPolicyProbabilistic
from tianshou.policy.modelfree.redq import REDQPolicy
from tianshou.policy.modelfree.sac import SACPolicy
from tianshou.trainer import OffPolicyTrainer, OnPolicyTrainer, Trainer
from tianshou.trainer.base import OffPolicyTrainerParams, OnPolicyTrainerParams
from tianshou.utils.net.discrete import DiscreteActor

CHECKPOINT_DICT_KEY_MODEL = "model"
CHECKPOINT_DICT_KEY_OBS_RMS = "obs_rms"
TParams = TypeVar("TParams", bound=Params)
TActorCriticParams = TypeVar(
    "TActorCriticParams",
    bound=Params | ParamsMixinSingleModel,
)
TActorDualCriticsParams = TypeVar(
    "TActorDualCriticsParams",
    bound=Params | ParamsMixinActorAndDualCritics,
)
TDiscreteCriticOnlyParams = TypeVar(
    "TDiscreteCriticOnlyParams",
    bound=Params | ParamsMixinSingleModel,
)
TAlgorithm = TypeVar("TAlgorithm", bound=Algorithm)
TPolicy = TypeVar("TPolicy", bound=Policy)
TTrainingConfig = TypeVar("TTrainingConfig", bound=TrainingConfig)
log = logging.getLogger(__name__)


class AlgorithmFactory(ABC, ToStringMixin, Generic[TTrainingConfig]):
    """Factory for the creation of an :class:`Algorithm` instance, its policy, trainer as well as collectors."""

    def __init__(self, training_config: TTrainingConfig, optim_factory: OptimizerFactoryFactory):
        self.training_config = training_config
        self.optim_factory = optim_factory
        self.algorithm_wrapper_factory: AlgorithmWrapperFactory | None = None
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
        buffer_size = self.training_config.buffer_size
        train_envs = envs.train_envs
        buffer: ReplayBuffer
        if len(train_envs) > 1:
            buffer = VectorReplayBuffer(
                buffer_size,
                len(train_envs),
                stack_num=self.training_config.replay_buffer_stack_num,
                save_only_last_obs=self.training_config.replay_buffer_save_only_last_obs,
                ignore_obs_next=self.training_config.replay_buffer_ignore_obs_next,
            )
        else:
            buffer = ReplayBuffer(
                buffer_size,
                stack_num=self.training_config.replay_buffer_stack_num,
                save_only_last_obs=self.training_config.replay_buffer_save_only_last_obs,
                ignore_obs_next=self.training_config.replay_buffer_ignore_obs_next,
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
        policy_wrapper_factory: AlgorithmWrapperFactory | None,
    ) -> None:
        self.algorithm_wrapper_factory = policy_wrapper_factory

    def set_trainer_callbacks(self, callbacks: TrainerCallbacks) -> None:
        self.trainer_callbacks = callbacks

    @staticmethod
    def _create_policy_from_args(
        constructor: type[TPolicy], params_dict: dict, policy_params: list[str], **kwargs: Any
    ) -> TPolicy:
        params = {p: params_dict.pop(p) for p in policy_params}
        return constructor(**params, **kwargs)

    @abstractmethod
    def _create_algorithm(self, envs: Environments, device: TDevice) -> Algorithm:
        pass

    def create_algorithm(self, envs: Environments, device: TDevice) -> Algorithm:
        policy = self._create_algorithm(envs, device)
        if self.algorithm_wrapper_factory is not None:
            policy = self.algorithm_wrapper_factory.create_wrapped_algorithm(
                policy,
                envs,
                self.optim_factory,
                device,
            )
        return policy

    @abstractmethod
    def create_trainer(self, world: World, policy_persistence: PolicyPersistence) -> Trainer:
        pass


class OnPolicyAlgorithmFactory(AlgorithmFactory[OnPolicyTrainingConfig], ABC):
    def create_trainer(
        self,
        world: World,
        policy_persistence: PolicyPersistence,
    ) -> OnPolicyTrainer:
        training_config = self.training_config
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
        assert world.train_collector is not None
        return algorithm.create_trainer(
            OnPolicyTrainerParams(
                train_collector=world.train_collector,
                test_collector=world.test_collector,
                max_epoch=training_config.num_epochs,
                step_per_epoch=training_config.step_per_epoch,
                repeat_per_collect=training_config.repeat_per_collect,
                episode_per_test=training_config.num_test_episodes,
                batch_size=training_config.batch_size,
                step_per_collect=training_config.step_per_collect,
                save_best_fn=policy_persistence.get_save_best_fn(world),
                save_checkpoint_fn=policy_persistence.get_save_checkpoint_fn(world),
                logger=world.logger,
                test_in_train=training_config.test_in_train,
                train_fn=train_fn,
                test_fn=test_fn,
                stop_fn=stop_fn,
                verbose=False,
            )
        )


class OffPolicyAlgorithmFactory(AlgorithmFactory[OffPolicyTrainingConfig], ABC):
    def create_trainer(
        self,
        world: World,
        policy_persistence: PolicyPersistence,
    ) -> OffPolicyTrainer:
        training_config = self.training_config
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
        assert world.train_collector is not None
        return algorithm.create_trainer(
            OffPolicyTrainerParams(
                train_collector=world.train_collector,
                test_collector=world.test_collector,
                max_epoch=training_config.num_epochs,
                step_per_epoch=training_config.step_per_epoch,
                step_per_collect=training_config.step_per_collect,
                episode_per_test=training_config.num_test_episodes,
                batch_size=training_config.batch_size,
                save_best_fn=policy_persistence.get_save_best_fn(world),
                logger=world.logger,
                update_per_step=training_config.update_per_step,
                test_in_train=training_config.test_in_train,
                train_fn=train_fn,
                test_fn=test_fn,
                stop_fn=stop_fn,
                verbose=False,
            )
        )


class ReinforceAlgorithmFactory(OnPolicyAlgorithmFactory):
    def __init__(
        self,
        params: ReinforceParams,
        training_config: OnPolicyTrainingConfig,
        actor_factory: ActorFactory,
        optim_factory: OptimizerFactoryFactory,
    ):
        super().__init__(training_config, optim_factory)
        self.params = params
        self.actor_factory = actor_factory
        self.optim_factory = optim_factory

    def _create_algorithm(self, envs: Environments, device: TDevice) -> Reinforce:
        actor = self.actor_factory.create_module(envs, device)
        kwargs = self.params.create_kwargs(
            ParamTransformerData(
                envs=envs,
                device=device,
                optim_factory_default=self.optim_factory,
            ),
        )
        dist_fn = self.actor_factory.create_dist_fn(envs)
        assert dist_fn is not None
        policy = self._create_policy_from_args(
            ActorPolicyProbabilistic,
            kwargs,
            ["action_scaling", "action_bound_method", "deterministic_eval"],
            actor=actor,
            dist_fn=dist_fn,
            action_space=envs.get_action_space(),
            observation_space=envs.get_observation_space(),
        )
        return Reinforce(
            policy=policy,
            **kwargs,
        )


class ActorCriticOnPolicyAlgorithmFactory(
    OnPolicyAlgorithmFactory,
    Generic[TActorCriticParams, TAlgorithm],
):
    def __init__(
        self,
        params: TActorCriticParams,
        training_config: OnPolicyTrainingConfig,
        actor_factory: ActorFactory,
        critic_factory: CriticFactory,
        optimizer_factory: OptimizerFactoryFactory,
    ):
        super().__init__(training_config, optim_factory=optimizer_factory)
        self.params = params
        self.actor_factory = actor_factory
        self.critic_factory = critic_factory
        self.optim_factory = optimizer_factory
        self.critic_use_action = False

    @abstractmethod
    def _get_algorithm_class(self) -> type[TAlgorithm]:
        pass

    @typing.no_type_check
    def _create_kwargs(self, envs: Environments, device: TDevice) -> dict[str, Any]:
        actor = self.actor_factory.create_module(envs, device)
        critic = self.critic_factory.create_module(envs, device, use_action=self.critic_use_action)
        kwargs = self.params.create_kwargs(
            ParamTransformerData(
                envs=envs,
                device=device,
                optim_factory_default=self.optim_factory,
            ),
        )
        kwargs["actor"] = actor
        kwargs["critic"] = critic
        kwargs["action_space"] = envs.get_action_space()
        kwargs["observation_space"] = envs.get_observation_space()
        kwargs["dist_fn"] = self.actor_factory.create_dist_fn(envs)
        return kwargs

    def _create_algorithm(self, envs: Environments, device: TDevice) -> TAlgorithm:
        params = self._create_kwargs(envs, device)
        policy = self._create_policy_from_args(
            ActorPolicyProbabilistic,
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


class A2CAlgorithmFactory(ActorCriticOnPolicyAlgorithmFactory[A2CParams, A2C]):
    def _get_algorithm_class(self) -> type[A2C]:
        return A2C


class PPOAlgorithmFactory(ActorCriticOnPolicyAlgorithmFactory[PPOParams, PPO]):
    def _get_algorithm_class(self) -> type[PPO]:
        return PPO


class NPGAlgorithmFactory(ActorCriticOnPolicyAlgorithmFactory[NPGParams, NPG]):
    def _get_algorithm_class(self) -> type[NPG]:
        return NPG


class TRPOAlgorithmFactory(ActorCriticOnPolicyAlgorithmFactory[TRPOParams, TRPO]):
    def _get_algorithm_class(self) -> type[TRPO]:
        return TRPO


class DiscreteCriticOnlyOffPolicyAlgorithmFactory(
    OffPolicyAlgorithmFactory,
    Generic[TDiscreteCriticOnlyParams, TAlgorithm],
):
    def __init__(
        self,
        params: TDiscreteCriticOnlyParams,
        training_config: OffPolicyTrainingConfig,
        model_factory: ModuleFactory,
        optim_factory: OptimizerFactoryFactory,
    ):
        super().__init__(training_config, optim_factory)
        self.params = params
        self.model_factory = model_factory
        self.optim_factory = optim_factory

    @abstractmethod
    def _get_algorithm_class(self) -> type[TAlgorithm]:
        pass

    @abstractmethod
    def _create_policy(
        self,
        model: torch.nn.Module,
        params: dict,
        action_space: gymnasium.spaces.Discrete,
        observation_space: gymnasium.spaces.Space,
    ) -> Policy:
        pass

    @typing.no_type_check
    def _create_algorithm(self, envs: Environments, device: TDevice) -> TAlgorithm:
        model = self.model_factory.create_module(envs, device)
        params_dict = self.params.create_kwargs(
            ParamTransformerData(
                envs=envs,
                device=device,
                optim_factory_default=self.optim_factory,
            ),
        )
        envs.get_type().assert_discrete(self)
        action_space = cast(gymnasium.spaces.Discrete, envs.get_action_space())
        policy = self._create_policy(model, params_dict, action_space, envs.get_observation_space())
        algorithm_class = self._get_algorithm_class()
        return algorithm_class(
            policy=policy,
            **params_dict,
        )


class DQNAlgorithmFactory(DiscreteCriticOnlyOffPolicyAlgorithmFactory[DQNParams, DQN]):
    def _create_policy(
        self,
        model: torch.nn.Module,
        params: dict,
        action_space: gymnasium.spaces.Discrete,
        observation_space: gymnasium.spaces.Space,
    ) -> Policy:
        return self._create_policy_from_args(
            constructor=DiscreteQLearningPolicy,
            params_dict=params,
            policy_params=["eps_training", "eps_inference"],
            model=model,
            action_space=action_space,
            observation_space=observation_space,
        )

    def _get_algorithm_class(self) -> type[DQN]:
        return DQN


class IQNAlgorithmFactory(DiscreteCriticOnlyOffPolicyAlgorithmFactory[IQNParams, IQN]):
    def _create_policy(
        self,
        model: torch.nn.Module,
        params: dict,
        action_space: gymnasium.spaces.Discrete,
        observation_space: gymnasium.spaces.Space,
    ) -> Policy:
        pass
        return self._create_policy_from_args(
            IQNPolicy,
            params,
            [
                "sample_size",
                "online_sample_size",
                "target_sample_size",
                "eps_training",
                "eps_inference",
            ],
            model=model,
            action_space=action_space,
            observation_space=observation_space,
        )

    def _get_algorithm_class(self) -> type[IQN]:
        return IQN


class DDPGAlgorithmFactory(OffPolicyAlgorithmFactory):
    def __init__(
        self,
        params: DDPGParams,
        training_config: OffPolicyTrainingConfig,
        actor_factory: ActorFactory,
        critic_factory: CriticFactory,
        optim_factory: OptimizerFactoryFactory,
    ):
        super().__init__(training_config, optim_factory)
        self.critic_factory = critic_factory
        self.actor_factory = actor_factory
        self.params = params
        self.optim_factory = optim_factory

    def _create_algorithm(self, envs: Environments, device: TDevice) -> Algorithm:
        actor = self.actor_factory.create_module(envs, device)
        critic = self.critic_factory.create_module(
            envs,
            device,
            True,
        )
        kwargs = self.params.create_kwargs(
            ParamTransformerData(
                envs=envs,
                device=device,
                optim_factory_default=self.optim_factory,
            ),
        )
        policy = self._create_policy_from_args(
            ContinuousDeterministicPolicy,
            kwargs,
            ["exploration_noise", "action_scaling", "action_bound_method"],
            actor=actor,
            action_space=envs.get_action_space(),
            observation_space=envs.get_observation_space(),
        )
        return DDPG(
            policy=policy,
            critic=critic,
            **kwargs,
        )


class REDQAlgorithmFactory(OffPolicyAlgorithmFactory):
    def __init__(
        self,
        params: REDQParams,
        training_config: OffPolicyTrainingConfig,
        actor_factory: ActorFactory,
        critic_ensemble_factory: CriticEnsembleFactory,
        optim_factory: OptimizerFactoryFactory,
    ):
        super().__init__(training_config, optim_factory)
        self.critic_ensemble_factory = critic_ensemble_factory
        self.actor_factory = actor_factory
        self.params = params
        self.optim_factory = optim_factory

    def _create_algorithm(self, envs: Environments, device: TDevice) -> Algorithm:
        envs.get_type().assert_continuous(self)
        actor = self.actor_factory.create_module(
            envs,
            device,
        )
        critic_ensemble = self.critic_ensemble_factory.create_module(
            envs,
            device,
            self.params.ensemble_size,
            True,
        )
        kwargs = self.params.create_kwargs(
            ParamTransformerData(
                envs=envs,
                device=device,
                optim_factory_default=self.optim_factory,
            ),
        )
        action_space = cast(gymnasium.spaces.Box, envs.get_action_space())
        policy = self._create_policy_from_args(
            REDQPolicy,
            kwargs,
            ["exploration_noise", "deterministic_eval", "action_scaling", "action_bound_method"],
            actor=actor,
            action_space=action_space,
            observation_space=envs.get_observation_space(),
        )
        return REDQ(
            policy=policy,
            critic=critic_ensemble,
            **kwargs,
        )


class ActorDualCriticsOffPolicyAlgorithmFactory(
    OffPolicyAlgorithmFactory,
    Generic[TActorDualCriticsParams, TAlgorithm, TPolicy],
):
    def __init__(
        self,
        params: TActorDualCriticsParams,
        training_config: OffPolicyTrainingConfig,
        actor_factory: ActorFactory,
        critic1_factory: CriticFactory,
        critic2_factory: CriticFactory,
        optim_factory: OptimizerFactoryFactory,
    ):
        super().__init__(training_config, optim_factory)
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

    @abstractmethod
    def _create_policy(
        self, actor: torch.nn.Module | DiscreteActor, envs: Environments, params: dict
    ) -> TPolicy:
        pass

    @typing.no_type_check
    def _create_algorithm(self, envs: Environments, device: TDevice) -> TAlgorithm:
        actor = self.actor_factory.create_module(envs, device)
        use_action_shape = self._get_discrete_last_size_use_action_shape()
        critic_use_action = self._get_critic_use_action(envs)
        critic1 = self.critic1_factory.create_module(
            envs,
            device,
            use_action=critic_use_action,
            discrete_last_size_use_action_shape=use_action_shape,
        )
        critic2 = self.critic2_factory.create_module(
            envs,
            device,
            use_action=critic_use_action,
            discrete_last_size_use_action_shape=use_action_shape,
        )
        kwargs = self.params.create_kwargs(
            ParamTransformerData(
                envs=envs,
                device=device,
                optim_factory_default=self.optim_factory,
            ),
        )
        policy = self._create_policy(actor, envs, kwargs)
        algorithm_class = self._get_algorithm_class()
        return algorithm_class(
            policy=policy,
            critic=critic1,
            critic2=critic2,
            **kwargs,
        )


class SACAlgorithmFactory(ActorDualCriticsOffPolicyAlgorithmFactory[SACParams, SAC, SACPolicy]):
    def _create_policy(
        self, actor: torch.nn.Module | DiscreteActor, envs: Environments, params: dict
    ) -> SACPolicy:
        return self._create_policy_from_args(
            SACPolicy,
            params,
            ["exploration_noise", "deterministic_eval", "action_scaling", "action_bound_method"],
            actor=actor,
            action_space=envs.get_action_space(),
            observation_space=envs.get_observation_space(),
        )

    def _get_algorithm_class(self) -> type[SAC]:
        return SAC


class DiscreteSACAlgorithmFactory(
    ActorDualCriticsOffPolicyAlgorithmFactory[DiscreteSACParams, DiscreteSAC, DiscreteSACPolicy]
):
    def _create_policy(
        self, actor: torch.nn.Module | DiscreteActor, envs: Environments, params: dict
    ) -> DiscreteSACPolicy:
        return self._create_policy_from_args(
            DiscreteSACPolicy,
            params,
            ["deterministic_eval"],
            actor=actor,
            action_space=envs.get_action_space(),
            observation_space=envs.get_observation_space(),
        )

    def _get_algorithm_class(self) -> type[DiscreteSAC]:
        return DiscreteSAC


class TD3AlgorithmFactory(
    ActorDualCriticsOffPolicyAlgorithmFactory[TD3Params, TD3, ContinuousDeterministicPolicy]
):
    def _create_policy(
        self, actor: torch.nn.Module | DiscreteActor, envs: Environments, params: dict
    ) -> ContinuousDeterministicPolicy:
        return self._create_policy_from_args(
            ContinuousDeterministicPolicy,
            params,
            ["exploration_noise", "action_scaling", "action_bound_method"],
            actor=actor,
            action_space=envs.get_action_space(),
            observation_space=envs.get_observation_space(),
        )

    def _get_algorithm_class(self) -> type[TD3]:
        return TD3
