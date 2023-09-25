from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pprint import pprint
from typing import Generic, TypeVar, Callable

import numpy as np
import torch

from tianshou.data import Collector
from tianshou.highlevel.agent import AgentFactory, PPOAgentFactory, SACAgentFactory
from tianshou.highlevel.config import RLSamplingConfig
from tianshou.highlevel.env import EnvFactory
from tianshou.highlevel.logger import DefaultLoggerFactory, LoggerFactory
from tianshou.highlevel.module import (
    ActorFactory,
    CriticFactory,
    DefaultActorFactory,
    DefaultCriticFactory,
)
from tianshou.highlevel.optim import AdamOptimizerFactory, OptimizerFactory
from tianshou.highlevel.params.policy_params import PPOParams, SACParams
from tianshou.policy import BasePolicy
from tianshou.policy.modelfree.pg import TDistParams
from tianshou.trainer import BaseTrainer

TPolicy = TypeVar("TPolicy", bound=BasePolicy)
TTrainer = TypeVar("TTrainer", bound=BaseTrainer)


@dataclass
class RLExperimentConfig:
    """Generic config for setting up the experiment, not RL or training specific."""

    seed: int = 42
    render: float | None = 0.0
    """Milliseconds between rendered frames; if None, no rendering"""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    resume_id: str | None = None
    """For restoring a model and running means of env-specifics from a checkpoint"""
    resume_path: str | None = None
    """For restoring a model and running means of env-specifics from a checkpoint"""
    watch: bool = False
    """If True, will not perform training and only watch the restored policy"""
    watch_num_episodes = 10


class RLExperiment(Generic[TPolicy, TTrainer]):
    def __init__(
        self,
        config: RLExperimentConfig,
        env_factory: EnvFactory,
        agent_factory: AgentFactory,
        logger_factory: LoggerFactory | None = None,
    ):
        if logger_factory is None:
            logger_factory = DefaultLoggerFactory()
        self.config = config
        self.env_factory = env_factory
        self.agent_factory = agent_factory
        self.logger_factory = logger_factory

    def _set_seed(self) -> None:
        seed = self.config.seed
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _build_config_dict(self) -> dict:
        return {
            # TODO
        }

    def run(self, log_name: str) -> None:
        self._set_seed()

        envs = self.env_factory.create_envs()

        full_config = self._build_config_dict()
        full_config.update(envs.info())

        run_id = self.config.resume_id
        logger = self.logger_factory.create_logger(
            log_name=log_name,
            run_id=run_id,
            config_dict=full_config,
        )

        policy = self.agent_factory.create_policy(envs, self.config.device)
        if self.config.resume_path:
            self.agent_factory.load_checkpoint(
                policy,
                self.config.resume_path,
                envs,
                self.config.device,
            )

        train_collector, test_collector = self.agent_factory.create_train_test_collector(
            policy,
            envs,
        )

        if not self.config.watch:
            trainer = self.agent_factory.create_trainer(
                policy,
                train_collector,
                test_collector,
                envs,
                logger,
            )
            result = trainer.run()
            pprint(result)  # TODO logging

        self._watch_agent(
            self.config.watch_num_episodes,
            policy,
            test_collector,
            self.config.render,
        )

    @staticmethod
    def _watch_agent(num_episodes, policy: BasePolicy, test_collector: Collector, render) -> None:
        policy.eval()
        test_collector.reset()
        result = test_collector.collect(n_episode=num_episodes, render=render)
        print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')


TBuilder = TypeVar("TBuilder", bound="RLExperimentBuilder")


class RLExperimentBuilder:
    def __init__(
        self,
        experiment_config: RLExperimentConfig,
        env_factory: EnvFactory,
        sampling_config: RLSamplingConfig,
    ):
        self._config = experiment_config
        self._env_factory = env_factory
        self._sampling_config = sampling_config
        self._logger_factory: LoggerFactory | None = None
        self._optim_factory: OptimizerFactory | None = None

    def with_logger_factory(self: TBuilder, logger_factory: LoggerFactory) -> TBuilder:
        self._logger_factory = logger_factory
        return self

    def with_optim_factory(self: TBuilder, optim_factory: OptimizerFactory) -> TBuilder:
        self._optim_factory = optim_factory
        return self

    def with_optim_factory_default(
        self: TBuilder, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
    ) -> TBuilder:
        """Configures the use of the default optimizer, Adam, with the given parameters.

        :param betas: coefficients used for computing running averages of gradient and its square
        :param eps: term added to the denominator to improve numerical stability
        :param weight_decay: weight decay (L2 penalty)
        :return: the builder
        """
        self._optim_factory = AdamOptimizerFactory(betas=betas, eps=eps, weight_decay=weight_decay)
        return self

    @abstractmethod
    def _create_agent_factory(self) -> AgentFactory:
        pass

    def _get_optim_factory(self) -> OptimizerFactory:
        if self._optim_factory is None:
            return AdamOptimizerFactory()
        else:
            return self._optim_factory

    def build(self) -> RLExperiment:
        return RLExperiment(
            self._config, self._env_factory, self._create_agent_factory(), self._logger_factory,
        )


class _BuilderMixinActorFactory:
    def __init__(self):
        self._actor_factory: ActorFactory | None = None

    def with_actor_factory(self: TBuilder, actor_factory: ActorFactory) -> TBuilder:
        self: TBuilder | _BuilderMixinActorFactory
        self._actor_factory = actor_factory
        return self

    def with_actor_factory_default(
        self: TBuilder,
        hidden_sizes: Sequence[int],
        continuous_unbounded=False,
        continuous_conditioned_sigma=False,
    ) -> TBuilder:
        self: TBuilder | _BuilderMixinActorFactory
        self._actor_factory = DefaultActorFactory(
            hidden_sizes,
            continuous_unbounded=continuous_unbounded,
            continuous_conditioned_sigma=continuous_conditioned_sigma,
        )
        return self

    def _get_actor_factory(self):
        if self._actor_factory is None:
            return DefaultActorFactory()
        else:
            return self._actor_factory


class _BuilderMixinCriticsFactory:
    def __init__(self, num_critics: int):
        self._critic_factories: list[CriticFactory | None] = [None] * num_critics

    def _with_critic_factory(self, idx: int, critic_factory: CriticFactory):
        self._critic_factories[idx] = critic_factory
        return self

    def _with_critic_factory_default(self, idx: int, hidden_sizes: Sequence[int]):
        self._critic_factories[idx] = DefaultCriticFactory(hidden_sizes)
        return self

    def _get_critic_factory(self, idx: int):
        factory = self._critic_factories[idx]
        if factory is None:
            return DefaultCriticFactory()
        else:
            return factory


class _BuilderMixinSingleCriticFactory(_BuilderMixinCriticsFactory):
    def __init__(self):
        super().__init__(1)

    def with_critic_factory(self: TBuilder, critic_factory: CriticFactory) -> TBuilder:
        self: TBuilder | "_BuilderMixinSingleCriticFactory"
        self._with_critic_factory(0, critic_factory)
        return self

    def with_critic_factory_default(
        self: TBuilder, hidden_sizes: Sequence[int] = DefaultCriticFactory.DEFAULT_HIDDEN_SIZES,
    ) -> TBuilder:
        self: TBuilder | "_BuilderMixinSingleCriticFactory"
        self._with_critic_factory_default(0, hidden_sizes)
        return self


class _BuilderMixinDualCriticFactory(_BuilderMixinCriticsFactory):
    def __init__(self):
        super().__init__(2)

    def with_common_critic_factory(self: TBuilder, critic_factory: CriticFactory) -> TBuilder:
        self: TBuilder | "_BuilderMixinDualCriticFactory"
        for i in range(len(self._critic_factories)):
            self._with_critic_factory(i, critic_factory)
        return self

    def with_common_critic_factory_default(
        self, hidden_sizes: Sequence[int] = DefaultCriticFactory.DEFAULT_HIDDEN_SIZES,
    ) -> TBuilder:
        self: TBuilder | "_BuilderMixinDualCriticFactory"
        for i in range(len(self._critic_factories)):
            self._with_critic_factory_default(i, hidden_sizes)
        return self

    def with_critic1_factory(self: TBuilder, critic_factory: CriticFactory) -> TBuilder:
        self: TBuilder | "_BuilderMixinDualCriticFactory"
        self._with_critic_factory(0, critic_factory)
        return self

    def with_critic1_factory_default(
        self, hidden_sizes: Sequence[int] = DefaultCriticFactory.DEFAULT_HIDDEN_SIZES,
    ) -> TBuilder:
        self: TBuilder | "_BuilderMixinDualCriticFactory"
        self._with_critic_factory_default(0, hidden_sizes)
        return self

    def with_critic2_factory(self: TBuilder, critic_factory: CriticFactory) -> TBuilder:
        self: TBuilder | "_BuilderMixinDualCriticFactory"
        self._with_critic_factory(1, critic_factory)
        return self

    def with_critic2_factory_default(
        self, hidden_sizes: Sequence[int] = DefaultCriticFactory.DEFAULT_HIDDEN_SIZES,
    ) -> TBuilder:
        self: TBuilder | "_BuilderMixinDualCriticFactory"
        self._with_critic_factory_default(0, hidden_sizes)
        return self


class PPOExperimentBuilder(
    RLExperimentBuilder, _BuilderMixinActorFactory, _BuilderMixinSingleCriticFactory,
):
    def __init__(
        self,
        experiment_config: RLExperimentConfig,
        env_factory: EnvFactory,
        sampling_config: RLSamplingConfig,
        dist_fn: Callable[[TDistParams], torch.distributions.Distribution],
    ):
        super().__init__(experiment_config, env_factory, sampling_config)
        _BuilderMixinActorFactory.__init__(self)
        _BuilderMixinSingleCriticFactory.__init__(self)
        self._params: PPOParams = PPOParams()
        self._dist_fn = dist_fn

    def with_ppo_params(self, params: PPOParams) -> "PPOExperimentBuilder":
        self._params = params
        return self

    @abstractmethod
    def _create_agent_factory(self) -> AgentFactory:
        return PPOAgentFactory(
            self._params,
            self._sampling_config,
            self._get_actor_factory(),
            self._get_critic_factory(0),
            self._get_optim_factory(),
            self._dist_fn
        )


class SACExperimentBuilder(
    RLExperimentBuilder, _BuilderMixinActorFactory, _BuilderMixinDualCriticFactory,
):
    def __init__(
        self,
        experiment_config: RLExperimentConfig,
        env_factory: EnvFactory,
        sampling_config: RLSamplingConfig,
    ):
        super().__init__(experiment_config, env_factory, sampling_config)
        _BuilderMixinActorFactory.__init__(self)
        _BuilderMixinDualCriticFactory.__init__(self)
        self._params: SACParams = SACParams()

    def with_sac_params(self, params: SACParams) -> "SACExperimentBuilder":
        self._params = params
        return self

    def _create_agent_factory(self) -> AgentFactory:
        return SACAgentFactory(self._params, self._sampling_config, self._get_actor_factory(),
            self._get_critic_factory(0), self._get_critic_factory(1), self._get_optim_factory())
