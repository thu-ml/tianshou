import os
import logging
from abc import abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pprint import pprint
from typing import Generic, Self, TypeVar

import numpy as np
import torch

from tianshou.data import Collector
from tianshou.highlevel.agent import (
    A2CAgentFactory,
    AgentFactory,
    DDPGAgentFactory,
    DiscreteSACAgentFactory,
    DQNAgentFactory,
    IQNAgentFactory,
    NPGAgentFactory,
    PGAgentFactory,
    PPOAgentFactory,
    REDQAgentFactory,
    SACAgentFactory,
    TD3AgentFactory,
    TRPOAgentFactory,
)
from tianshou.highlevel.config import SamplingConfig
from tianshou.highlevel.env import EnvFactory, Environments
from tianshou.highlevel.logger import DefaultLoggerFactory, LoggerFactory
from tianshou.highlevel.module.actor import (
    ActorFactory,
    ActorFactoryDefault,
    ActorFactoryTransientStorageDecorator,
    ActorFuture,
    ActorFutureProviderProtocol,
    ContinuousActorType,
    IntermediateModuleFactoryFromActorFactory,
)
from tianshou.highlevel.module.core import (
    ImplicitQuantileNetworkFactory,
    IntermediateModuleFactory,
    TDevice,
)
from tianshou.highlevel.module.critic import (
    CriticEnsembleFactory,
    CriticEnsembleFactoryDefault,
    CriticFactory,
    CriticFactoryDefault,
    CriticFactoryReuseActor,
)
from tianshou.highlevel.optim import OptimizerFactory, OptimizerFactoryAdam
from tianshou.highlevel.params.policy_params import (
    A2CParams,
    DDPGParams,
    DiscreteSACParams,
    DQNParams,
    IQNParams,
    NPGParams,
    PGParams,
    PPOParams,
    REDQParams,
    SACParams,
    TD3Params,
    TRPOParams,
)
from tianshou.highlevel.params.policy_wrapper import PolicyWrapperFactory
from tianshou.highlevel.persistence import PersistableConfigProtocol, PolicyPersistence, PersistenceGroup
from tianshou.highlevel.trainer import (
    TrainerCallbacks,
    TrainerEpochCallbackTest,
    TrainerEpochCallbackTrain,
    TrainerStopCallback,
)
from tianshou.highlevel.world import World
from tianshou.policy import BasePolicy
from tianshou.trainer import BaseTrainer
from tianshou.utils.logging import datetime_tag
from tianshou.utils.string import ToStringMixin

log = logging.getLogger(__name__)
TPolicy = TypeVar("TPolicy", bound=BasePolicy)
TTrainer = TypeVar("TTrainer", bound=BaseTrainer)


@dataclass
class ExperimentConfig:
    """Generic config for setting up the experiment, not RL or training specific."""

    seed: int = 42
    render: float | None = 0.0
    """Milliseconds between rendered frames; if None, no rendering"""
    device: TDevice = "cuda" if torch.cuda.is_available() else "cpu"
    """The torch device to use"""
    policy_restore_directory: str | None = None
    """Directory from which to load the policy neural network parameters (saved in a previous run)"""
    train: bool = True
    """Whether to perform training"""
    watch: bool = True
    """Whether to watch agent performance (after training)"""
    watch_num_episodes = 10
    """Number of episodes for which to watch performance (if watch is enabled)"""
    watch_render: float = 0.0
    """Milliseconds between rendered frames when watching agent performance (if watch is enabled)"""


class Experiment(Generic[TPolicy, TTrainer], ToStringMixin):
    def __init__(
        self,
        config: ExperimentConfig,
        env_factory: EnvFactory | Callable[[PersistableConfigProtocol | None], Environments],
        agent_factory: AgentFactory,
        logger_factory: LoggerFactory | None = None,
        env_config: PersistableConfigProtocol | None = None,
    ):
        if logger_factory is None:
            logger_factory = DefaultLoggerFactory()
        self.config = config
        self.env_factory = env_factory
        self.agent_factory = agent_factory
        self.logger_factory = logger_factory
        self.env_config = env_config

    def _set_seed(self) -> None:
        seed = self.config.seed
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _build_config_dict(self) -> dict:
        return {
            "experiment": self.pprints()
        }

    def run(self, experiment_name: str | None = None, logger_run_id: str | None = None) -> None:
        """:param experiment_name: the experiment name, which corresponds to the directory (within the logging
            directory) where all results associated with the experiment will be saved.
            The name may contain path separators (os.path.sep, used by os.path.join), in which case
            a nested directory structure will be created.
            If None, use a name containing the current date and time.
        :param logger_run_id: Run identifier to use for logger initialization/resumption (applies when
            using wandb, in particular).
        :return:
        """
        if experiment_name is None:
            experiment_name = datetime_tag()

        log.info(f"Working directory: {os.getcwd()}")

        self._set_seed()

        # create environments
        envs = self.env_factory(self.env_config)
        log.info(f"Created {envs}")

        # initialize persistence
        additional_persistence = PersistenceGroup(*envs.persistence)
        policy_persistence = PolicyPersistence(additional_persistence)

        # initialize logger
        full_config = self._build_config_dict()
        full_config.update(envs.info())
        logger = self.logger_factory.create_logger(
            log_name=experiment_name,
            run_id=logger_run_id,
            config_dict=full_config,
        )

        policy = self.agent_factory.create_policy(envs, self.config.device)

        train_collector, test_collector = self.agent_factory.create_train_test_collector(
            policy,
            envs,
        )

        world = World(
            envs=envs,
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            logger=logger,
            restore_directory=self.config.policy_restore_directory
        )

        if self.config.policy_restore_directory:
            policy_persistence.restore(
                policy,
                world,
                self.config.device,
            )

        if self.config.train:
            trainer = self.agent_factory.create_trainer(world, policy_persistence)
            world.trainer = trainer

            result = trainer.run()
            pprint(result)  # TODO logging

        if self.config.watch:
            self._watch_agent(
                self.config.watch_num_episodes,
                policy,
                test_collector,
                self.config.watch_render,
            )

        # TODO return result

    @staticmethod
    def _watch_agent(
        num_episodes: int,
        policy: BasePolicy,
        test_collector: Collector,
        render: float,
    ) -> None:
        policy.eval()
        test_collector.reset()
        result = test_collector.collect(n_episode=num_episodes, render=render)
        print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')


class ExperimentBuilder:
    def __init__(
        self,
        env_factory: EnvFactory,
        experiment_config: ExperimentConfig | None = None,
        sampling_config: SamplingConfig | None = None,
    ):
        if experiment_config is None:
            experiment_config = ExperimentConfig()
        if sampling_config is None:
            sampling_config = SamplingConfig()
        self._config = experiment_config
        self._env_factory = env_factory
        self._sampling_config = sampling_config
        self._logger_factory: LoggerFactory | None = None
        self._optim_factory: OptimizerFactory | None = None
        self._env_config: PersistableConfigProtocol | None = None
        self._policy_wrapper_factory: PolicyWrapperFactory | None = None
        self._trainer_callbacks: TrainerCallbacks = TrainerCallbacks()

    def with_env_config(self, config: PersistableConfigProtocol) -> Self:
        self._env_config = config
        return self

    def with_logger_factory(self, logger_factory: LoggerFactory) -> Self:
        self._logger_factory = logger_factory
        return self

    def with_policy_wrapper_factory(self, policy_wrapper_factory: PolicyWrapperFactory) -> Self:
        self._policy_wrapper_factory = policy_wrapper_factory
        return self

    def with_optim_factory(self, optim_factory: OptimizerFactory) -> Self:
        self._optim_factory = optim_factory
        return self

    def with_optim_factory_default(
        self,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0,
    ) -> Self:
        """Configures the use of the default optimizer, Adam, with the given parameters.

        :param betas: coefficients used for computing running averages of gradient and its square
        :param eps: term added to the denominator to improve numerical stability
        :param weight_decay: weight decay (L2 penalty)
        :return: the builder
        """
        self._optim_factory = OptimizerFactoryAdam(betas=betas, eps=eps, weight_decay=weight_decay)
        return self

    def with_trainer_epoch_callback_train(self, callback: TrainerEpochCallbackTrain) -> Self:
        self._trainer_callbacks.epoch_callback_train = callback
        return self

    def with_trainer_epoch_callback_test(self, callback: TrainerEpochCallbackTest) -> Self:
        self._trainer_callbacks.epoch_callback_test = callback
        return self

    def with_trainer_stop_callback(self, callback: TrainerStopCallback) -> Self:
        self._trainer_callbacks.stop_callback = callback
        return self

    @abstractmethod
    def _create_agent_factory(self) -> AgentFactory:
        pass

    def _get_optim_factory(self) -> OptimizerFactory:
        if self._optim_factory is None:
            return OptimizerFactoryAdam()
        else:
            return self._optim_factory

    def build(self) -> Experiment:
        agent_factory = self._create_agent_factory()
        agent_factory.set_trainer_callbacks(self._trainer_callbacks)
        if self._policy_wrapper_factory:
            agent_factory.set_policy_wrapper_factory(self._policy_wrapper_factory)
        experiment: Experiment = Experiment(
            self._config,
            self._env_factory,
            agent_factory,
            self._logger_factory,
            env_config=self._env_config,
        )
        log.info(f"Created experiment:\n{experiment.pprints()}")
        return experiment


class _BuilderMixinActorFactory(ActorFutureProviderProtocol):
    def __init__(self, continuous_actor_type: ContinuousActorType):
        self._continuous_actor_type = continuous_actor_type
        self._actor_future = ActorFuture()
        self._actor_factory: ActorFactory | None = None

    def with_actor_factory(self, actor_factory: ActorFactory) -> Self:
        self._actor_factory = actor_factory
        return self

    def _with_actor_factory_default(
        self,
        hidden_sizes: Sequence[int],
        continuous_unbounded: bool = False,
        continuous_conditioned_sigma: bool = False,
    ) -> Self:
        self._actor_factory = ActorFactoryDefault(
            self._continuous_actor_type,
            hidden_sizes,
            continuous_unbounded=continuous_unbounded,
            continuous_conditioned_sigma=continuous_conditioned_sigma,
        )
        return self

    def get_actor_future(self) -> ActorFuture:
        return self._actor_future

    def _get_actor_factory(self) -> ActorFactory:
        actor_factory: ActorFactory
        if self._actor_factory is None:
            actor_factory = ActorFactoryDefault(self._continuous_actor_type)
        else:
            actor_factory = self._actor_factory
        return ActorFactoryTransientStorageDecorator(actor_factory, self._actor_future)


class _BuilderMixinActorFactory_ContinuousGaussian(_BuilderMixinActorFactory):
    """Specialization of the actor mixin where, in the continuous case, the actor component outputs
    Gaussian distribution parameters.
    """

    def __init__(self) -> None:
        super().__init__(ContinuousActorType.GAUSSIAN)

    def with_actor_factory_default(
        self,
        hidden_sizes: Sequence[int],
        continuous_unbounded: bool = False,
        continuous_conditioned_sigma: bool = False,
    ) -> Self:
        return super()._with_actor_factory_default(
            hidden_sizes,
            continuous_unbounded=continuous_unbounded,
            continuous_conditioned_sigma=continuous_conditioned_sigma,
        )


class _BuilderMixinActorFactory_ContinuousDeterministic(_BuilderMixinActorFactory):
    """Specialization of the actor mixin where, in the continuous case, the actor uses a deterministic policy."""

    def __init__(self) -> None:
        super().__init__(ContinuousActorType.DETERMINISTIC)

    def with_actor_factory_default(self, hidden_sizes: Sequence[int]) -> Self:
        return super()._with_actor_factory_default(hidden_sizes)


class _BuilderMixinCriticsFactory:
    def __init__(self, num_critics: int, actor_future_provider: ActorFutureProviderProtocol):
        self._actor_future_provider = actor_future_provider
        self._critic_factories: list[CriticFactory | None] = [None] * num_critics

    def _with_critic_factory(self, idx: int, critic_factory: CriticFactory) -> Self:
        self._critic_factories[idx] = critic_factory
        return self

    def _with_critic_factory_default(self, idx: int, hidden_sizes: Sequence[int]) -> Self:
        self._critic_factories[idx] = CriticFactoryDefault(hidden_sizes)
        return self

    def _with_critic_factory_use_actor(self, idx: int) -> Self:
        self._critic_factories[idx] = CriticFactoryReuseActor(
            self._actor_future_provider.get_actor_future(),
        )
        return self

    def _get_critic_factory(self, idx: int) -> CriticFactory:
        factory = self._critic_factories[idx]
        if factory is None:
            return CriticFactoryDefault()
        else:
            return factory


class _BuilderMixinSingleCriticFactory(_BuilderMixinCriticsFactory):
    def __init__(self, actor_future_provider: ActorFutureProviderProtocol = None) -> None:
        super().__init__(1, actor_future_provider)

    def with_critic_factory(self, critic_factory: CriticFactory) -> Self:
        self._with_critic_factory(0, critic_factory)
        return self

    def with_critic_factory_default(
        self,
        hidden_sizes: Sequence[int] = CriticFactoryDefault.DEFAULT_HIDDEN_SIZES,
    ) -> Self:
        self._with_critic_factory_default(0, hidden_sizes)
        return self


class _BuilderMixinSingleCriticCanUseActorFactory(_BuilderMixinSingleCriticFactory):
    def __init__(self, actor_future_provider: ActorFutureProviderProtocol) -> None:
        super().__init__(actor_future_provider)

    def with_critic_factory_use_actor(self) -> Self:
        """Makes the critic use the same network as the actor."""
        return self._with_critic_factory_use_actor(0)


class _BuilderMixinDualCriticFactory(_BuilderMixinCriticsFactory):
    def __init__(self, actor_future_provider: ActorFutureProviderProtocol) -> None:
        super().__init__(2, actor_future_provider)

    def with_common_critic_factory(self, critic_factory: CriticFactory) -> Self:
        for i in range(len(self._critic_factories)):
            self._with_critic_factory(i, critic_factory)
        return self

    def with_common_critic_factory_default(
        self,
        hidden_sizes: Sequence[int] = CriticFactoryDefault.DEFAULT_HIDDEN_SIZES,
    ) -> Self:
        for i in range(len(self._critic_factories)):
            self._with_critic_factory_default(i, hidden_sizes)
        return self

    def with_common_critic_factory_use_actor(self) -> Self:
        """Makes all critics use the same network as the actor."""
        for i in range(len(self._critic_factories)):
            self._with_critic_factory_use_actor(i)
        return self

    def with_critic1_factory(self, critic_factory: CriticFactory) -> Self:
        self._with_critic_factory(0, critic_factory)
        return self

    def with_critic1_factory_default(
        self,
        hidden_sizes: Sequence[int] = CriticFactoryDefault.DEFAULT_HIDDEN_SIZES,
    ) -> Self:
        self._with_critic_factory_default(0, hidden_sizes)
        return self

    def with_critic1_factory_use_actor(self) -> Self:
        """Makes the critic use the same network as the actor."""
        return self._with_critic_factory_use_actor(0)

    def with_critic2_factory(self, critic_factory: CriticFactory) -> Self:
        self._with_critic_factory(1, critic_factory)
        return self

    def with_critic2_factory_default(
        self,
        hidden_sizes: Sequence[int] = CriticFactoryDefault.DEFAULT_HIDDEN_SIZES,
    ) -> Self:
        self._with_critic_factory_default(0, hidden_sizes)
        return self

    def with_critic2_factory_use_actor(self) -> Self:
        """Makes the second critic use the same network as the actor."""
        return self._with_critic_factory_use_actor(1)


class _BuilderMixinCriticEnsembleFactory:
    def __init__(self) -> None:
        self.critic_ensemble_factory: CriticEnsembleFactory | None = None

    def with_critic_ensemble_factory(self, factory: CriticEnsembleFactory) -> Self:
        self.critic_ensemble_factory = factory
        return self

    def with_critic_ensemble_factory_default(
        self,
        hidden_sizes: Sequence[int] = CriticFactoryDefault.DEFAULT_HIDDEN_SIZES,
    ) -> Self:
        self.critic_ensemble_factory = CriticEnsembleFactoryDefault(hidden_sizes)
        return self

    def _get_critic_ensemble_factory(self):
        if self.critic_ensemble_factory is None:
            return CriticEnsembleFactoryDefault()
        else:
            return self.critic_ensemble_factory


class PGExperimentBuilder(
    ExperimentBuilder,
    _BuilderMixinActorFactory_ContinuousGaussian,
):
    def __init__(
        self,
        env_factory: EnvFactory,
        experiment_config: ExperimentConfig | None = None,
        sampling_config: SamplingConfig | None = None,
    ):
        super().__init__(env_factory, experiment_config, sampling_config)
        _BuilderMixinActorFactory_ContinuousGaussian.__init__(self)
        self._params: PGParams = PGParams()
        self._env_config = None

    def with_pg_params(self, params: PGParams) -> Self:
        self._params = params
        return self

    def _create_agent_factory(self) -> AgentFactory:
        return PGAgentFactory(
            self._params,
            self._sampling_config,
            self._get_actor_factory(),
            self._get_optim_factory(),
        )


class A2CExperimentBuilder(
    ExperimentBuilder,
    _BuilderMixinActorFactory_ContinuousGaussian,
    _BuilderMixinSingleCriticCanUseActorFactory,
):
    def __init__(
        self,
        env_factory: EnvFactory,
        experiment_config: ExperimentConfig | None = None,
        sampling_config: SamplingConfig | None = None,
    ):
        super().__init__(env_factory, experiment_config, sampling_config)
        _BuilderMixinActorFactory_ContinuousGaussian.__init__(self)
        _BuilderMixinSingleCriticCanUseActorFactory.__init__(self, self)
        self._params: A2CParams = A2CParams()
        self._env_config = None

    def with_a2c_params(self, params: A2CParams) -> Self:
        self._params = params
        return self

    def _create_agent_factory(self) -> AgentFactory:
        return A2CAgentFactory(
            self._params,
            self._sampling_config,
            self._get_actor_factory(),
            self._get_critic_factory(0),
            self._get_optim_factory(),
        )


class PPOExperimentBuilder(
    ExperimentBuilder,
    _BuilderMixinActorFactory_ContinuousGaussian,
    _BuilderMixinSingleCriticCanUseActorFactory,
):
    def __init__(
        self,
        env_factory: EnvFactory,
        experiment_config: ExperimentConfig | None = None,
        sampling_config: SamplingConfig | None = None,
    ):
        super().__init__(env_factory, experiment_config, sampling_config)
        _BuilderMixinActorFactory_ContinuousGaussian.__init__(self)
        _BuilderMixinSingleCriticCanUseActorFactory.__init__(self, self)
        self._params: PPOParams = PPOParams()

    def with_ppo_params(self, params: PPOParams) -> Self:
        self._params = params
        return self

    def _create_agent_factory(self) -> AgentFactory:
        return PPOAgentFactory(
            self._params,
            self._sampling_config,
            self._get_actor_factory(),
            self._get_critic_factory(0),
            self._get_optim_factory(),
        )


class NPGExperimentBuilder(
    ExperimentBuilder,
    _BuilderMixinActorFactory_ContinuousGaussian,
    _BuilderMixinSingleCriticCanUseActorFactory,
):
    def __init__(
        self,
        env_factory: EnvFactory,
        experiment_config: ExperimentConfig | None = None,
        sampling_config: SamplingConfig | None = None,
    ):
        super().__init__(env_factory, experiment_config, sampling_config)
        _BuilderMixinActorFactory_ContinuousGaussian.__init__(self)
        _BuilderMixinSingleCriticCanUseActorFactory.__init__(self, self)
        self._params: NPGParams = NPGParams()

    def with_npg_params(self, params: NPGParams) -> Self:
        self._params = params
        return self

    def _create_agent_factory(self) -> AgentFactory:
        return NPGAgentFactory(
            self._params,
            self._sampling_config,
            self._get_actor_factory(),
            self._get_critic_factory(0),
            self._get_optim_factory(),
        )


class TRPOExperimentBuilder(
    ExperimentBuilder,
    _BuilderMixinActorFactory_ContinuousGaussian,
    _BuilderMixinSingleCriticCanUseActorFactory,
):
    def __init__(
        self,
        env_factory: EnvFactory,
        experiment_config: ExperimentConfig | None = None,
        sampling_config: SamplingConfig | None = None,
    ):
        super().__init__(env_factory, experiment_config, sampling_config)
        _BuilderMixinActorFactory_ContinuousGaussian.__init__(self)
        _BuilderMixinSingleCriticCanUseActorFactory.__init__(self, self)
        self._params: TRPOParams = TRPOParams()

    def with_trpo_params(self, params: TRPOParams) -> Self:
        self._params = params
        return self

    def _create_agent_factory(self) -> AgentFactory:
        return TRPOAgentFactory(
            self._params,
            self._sampling_config,
            self._get_actor_factory(),
            self._get_critic_factory(0),
            self._get_optim_factory(),
        )


class DQNExperimentBuilder(
    ExperimentBuilder,
    _BuilderMixinActorFactory,
):
    def __init__(
        self,
        env_factory: EnvFactory,
        experiment_config: ExperimentConfig | None = None,
        sampling_config: SamplingConfig | None = None,
    ):
        super().__init__(env_factory, experiment_config, sampling_config)
        _BuilderMixinActorFactory.__init__(self, ContinuousActorType.UNSUPPORTED)
        self._params: DQNParams = DQNParams()

    def with_dqn_params(self, params: DQNParams) -> Self:
        self._params = params
        return self

    def _create_agent_factory(self) -> AgentFactory:
        return DQNAgentFactory(
            self._params,
            self._sampling_config,
            self._get_actor_factory(),
            self._get_optim_factory(),
        )


class IQNExperimentBuilder(ExperimentBuilder):
    def __init__(
        self,
        env_factory: EnvFactory,
        experiment_config: ExperimentConfig | None = None,
        sampling_config: SamplingConfig | None = None,
    ):
        super().__init__(env_factory, experiment_config, sampling_config)
        self._params: IQNParams = IQNParams()
        self._preprocess_network_factory = IntermediateModuleFactoryFromActorFactory(
            ActorFactoryDefault(ContinuousActorType.UNSUPPORTED),
        )

    def with_iqn_params(self, params: IQNParams) -> Self:
        self._params = params
        return self

    def with_preprocess_network_factory(self, module_factory: IntermediateModuleFactory) -> Self:
        self._preprocess_network_factory = module_factory
        return self

    def _create_agent_factory(self) -> AgentFactory:
        model_factory = ImplicitQuantileNetworkFactory(
            self._preprocess_network_factory,
            hidden_sizes=self._params.hidden_sizes,
            num_cosines=self._params.num_cosines,
        )
        return IQNAgentFactory(
            self._params,
            self._sampling_config,
            model_factory,
            self._get_optim_factory(),
        )


class DDPGExperimentBuilder(
    ExperimentBuilder,
    _BuilderMixinActorFactory_ContinuousDeterministic,
    _BuilderMixinSingleCriticCanUseActorFactory,
):
    def __init__(
        self,
        env_factory: EnvFactory,
        experiment_config: ExperimentConfig | None = None,
        sampling_config: SamplingConfig | None = None,
    ):
        super().__init__(env_factory, experiment_config, sampling_config)
        _BuilderMixinActorFactory_ContinuousDeterministic.__init__(self)
        _BuilderMixinSingleCriticCanUseActorFactory.__init__(self, self)
        self._params: DDPGParams = DDPGParams()

    def with_ddpg_params(self, params: DDPGParams) -> Self:
        self._params = params
        return self

    def _create_agent_factory(self) -> AgentFactory:
        return DDPGAgentFactory(
            self._params,
            self._sampling_config,
            self._get_actor_factory(),
            self._get_critic_factory(0),
            self._get_optim_factory(),
        )


class REDQExperimentBuilder(
    ExperimentBuilder,
    _BuilderMixinActorFactory_ContinuousGaussian,
    _BuilderMixinCriticEnsembleFactory,
):
    def __init__(
        self,
        env_factory: EnvFactory,
        experiment_config: ExperimentConfig | None = None,
        sampling_config: SamplingConfig | None = None,
    ):
        super().__init__(env_factory, experiment_config, sampling_config)
        _BuilderMixinActorFactory_ContinuousGaussian.__init__(self)
        _BuilderMixinCriticEnsembleFactory.__init__(self)
        self._params: REDQParams = REDQParams()

    def with_redq_params(self, params: REDQParams) -> Self:
        self._params = params
        return self

    def _create_agent_factory(self) -> AgentFactory:
        return REDQAgentFactory(
            self._params,
            self._sampling_config,
            self._get_actor_factory(),
            self._get_critic_ensemble_factory(),
            self._get_optim_factory(),
        )


class SACExperimentBuilder(
    ExperimentBuilder,
    _BuilderMixinActorFactory_ContinuousGaussian,
    _BuilderMixinDualCriticFactory,
):
    def __init__(
        self,
        env_factory: EnvFactory,
        experiment_config: ExperimentConfig | None = None,
        sampling_config: SamplingConfig | None = None,
    ):
        super().__init__(env_factory, experiment_config, sampling_config)
        _BuilderMixinActorFactory_ContinuousGaussian.__init__(self)
        _BuilderMixinDualCriticFactory.__init__(self, self)
        self._params: SACParams = SACParams()

    def with_sac_params(self, params: SACParams) -> Self:
        self._params = params
        return self

    def _create_agent_factory(self) -> AgentFactory:
        return SACAgentFactory(
            self._params,
            self._sampling_config,
            self._get_actor_factory(),
            self._get_critic_factory(0),
            self._get_critic_factory(1),
            self._get_optim_factory(),
        )


class DiscreteSACExperimentBuilder(
    ExperimentBuilder,
    _BuilderMixinActorFactory,
    _BuilderMixinDualCriticFactory,
):
    def __init__(
        self,
        env_factory: EnvFactory,
        experiment_config: ExperimentConfig | None = None,
        sampling_config: SamplingConfig | None = None,
    ):
        super().__init__(env_factory, experiment_config, sampling_config)
        _BuilderMixinActorFactory.__init__(self, ContinuousActorType.UNSUPPORTED)
        _BuilderMixinDualCriticFactory.__init__(self, self)
        self._params: DiscreteSACParams = DiscreteSACParams()

    def with_sac_params(self, params: DiscreteSACParams) -> Self:
        self._params = params
        return self

    def _create_agent_factory(self) -> AgentFactory:
        return DiscreteSACAgentFactory(
            self._params,
            self._sampling_config,
            self._get_actor_factory(),
            self._get_critic_factory(0),
            self._get_critic_factory(1),
            self._get_optim_factory(),
        )


class TD3ExperimentBuilder(
    ExperimentBuilder,
    _BuilderMixinActorFactory_ContinuousDeterministic,
    _BuilderMixinDualCriticFactory,
):
    def __init__(
        self,
        env_factory: EnvFactory,
        experiment_config: ExperimentConfig | None = None,
        sampling_config: SamplingConfig | None = None,
    ):
        super().__init__(env_factory, experiment_config, sampling_config)
        _BuilderMixinActorFactory_ContinuousDeterministic.__init__(self)
        _BuilderMixinDualCriticFactory.__init__(self, self)
        self._params: TD3Params = TD3Params()

    def with_td3_params(self, params: TD3Params) -> Self:
        self._params = params
        return self

    def _create_agent_factory(self) -> AgentFactory:
        return TD3AgentFactory(
            self._params,
            self._sampling_config,
            self._get_actor_factory(),
            self._get_critic_factory(0),
            self._get_critic_factory(1),
            self._get_optim_factory(),
        )
