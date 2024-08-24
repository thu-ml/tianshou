"""The experiment module provides high-level interfaces for setting up and running reinforcement learning experiments.

The main entry points are:

* :class:`ExperimentConfig`: a dataclass for configuring the experiment. The configuration is
  different from RL specific configuration (such as policy and trainer parameters)
  and only pertains to configuration that is common to all experiments.
* :class:`Experiment`: represents a reinforcement learning experiment.
  It is composed of configuration and factory objects, is lightweight and serializable.
  An instance of `Experiment` is usually saved as a pickle file after an experiment is executed.
* :class:`ExperimentBuilder`: a helper class for creating experiments. It contains a lot of defaults
  and allows for easy customization of the experiment setup.
* :class:`ExperimentCollection`: a shallow wrapper around a list of experiments providing a
  simple interface for running them with a launcher. Useful for running multiple experiments in parallel, in
  particular, for the important case of running experiments that only differ in their random seeds.

Various implementations of the `ExperimentBuilder` are provided for each of the algorithms supported by Tianshou.
"""

import os
import pickle
from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import suppress
from copy import deepcopy
from dataclasses import asdict, dataclass
from pprint import pformat
from typing import TYPE_CHECKING, Any, Self, Union, cast

if TYPE_CHECKING:
    from tianshou.evaluation.launcher import ExpLauncher, RegisteredExpLauncher

import numpy as np
import torch
from sensai.util import logging
from sensai.util.logging import datetime_tag
from sensai.util.string import ToStringMixin

from tianshou.data import BaseCollector, Collector, CollectStats, InfoStats
from tianshou.env import BaseVectorEnv
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
    RandomActionAgentFactory,
    REDQAgentFactory,
    SACAgentFactory,
    TD3AgentFactory,
    TRPOAgentFactory,
)
from tianshou.highlevel.config import SamplingConfig
from tianshou.highlevel.env import EnvFactory
from tianshou.highlevel.logger import LoggerFactory, LoggerFactoryDefault, TLogger
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
    TDevice,
)
from tianshou.highlevel.module.critic import (
    CriticEnsembleFactory,
    CriticEnsembleFactoryDefault,
    CriticFactory,
    CriticFactoryDefault,
    CriticFactoryReuseActor,
)
from tianshou.highlevel.module.intermediate import IntermediateModuleFactory
from tianshou.highlevel.module.special import ImplicitQuantileNetworkFactory
from tianshou.highlevel.optim import (
    OptimizerFactory,
    OptimizerFactoryAdam,
)
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
from tianshou.highlevel.persistence import (
    PersistenceGroup,
    PolicyPersistence,
)
from tianshou.highlevel.trainer import (
    EpochStopCallback,
    EpochTestCallback,
    EpochTrainCallback,
    TrainerCallbacks,
)
from tianshou.highlevel.world import World
from tianshou.policy import BasePolicy
from tianshou.utils import LazyLogger
from tianshou.utils.net.common import ModuleType
from tianshou.utils.print import DataclassPPrintMixin
from tianshou.utils.warning import deprecation

log = logging.getLogger(__name__)


@dataclass
class ExperimentConfig(ToStringMixin, DataclassPPrintMixin):
    """Generic config for setting up the experiment, not RL or training specific."""

    seed: int = 42
    """The random seed with which to initialize random number generators."""
    device: TDevice = "cuda" if torch.cuda.is_available() else "cpu"
    """The torch device to use"""
    policy_restore_directory: str | None = None
    """Directory from which to load the policy neural network parameters (persistence directory of a previous run)"""
    train: bool = True
    """Whether to perform training"""
    watch: bool = True
    """Whether to watch agent performance (after training)"""
    watch_num_episodes: int = 10
    """Number of episodes for which to watch performance (if `watch` is enabled)"""
    watch_render: float = 0.0
    """Milliseconds between rendered frames when watching agent performance (if `watch` is enabled)"""
    persistence_base_dir: str = "log"
    """Base directory in which experiment data is to be stored. Every experiment run will create a subdirectory
    in this directory based on the run's experiment name"""
    persistence_enabled: bool = True
    """Whether persistence is enabled, allowing files to be stored"""
    log_file_enabled: bool = True
    """Whether to write to a log file; has no effect if `persistence_enabled` is False.
    Disable this if you have externally configured log file generation."""
    policy_persistence_mode: PolicyPersistence.Mode = PolicyPersistence.Mode.POLICY
    """Controls the way in which the policy is persisted"""


@dataclass
class ExperimentResult:
    """Contains the results of an experiment."""

    world: World
    """The `World` contains all the essential instances of the experiment.
    Can also be created via `Experiment.create_experiment_world` for more custom setups, see docstring there.

    Note: it is typically not serializable, so it is not stored in the experiment pickle, and shouldn't be
    sent across processes, meaning also that `ExperimentResult` itself is typically not serializable.
    """
    trainer_result: InfoStats | None
    """dataclass of results as returned by the trainer (if any)"""


class Experiment(ToStringMixin, DataclassPPrintMixin):
    """Represents a reinforcement learning experiment.

    An experiment is composed only of configuration and factory objects, which themselves
    should be designed to contain only configuration. Therefore, experiments can easily
    be stored/pickled and later restored without any problems.

    The main entry points are:

    1. :meth:`run`: runs the experiment and returns the results
    2. :meth:`create_experiment_world`: creates the world object for the experiment, which contains all relevant instances.
        Useful for setting up the experiment and running it in a more custom way.

    The methods :meth:`save` and :meth:`from_directory` can be used to store and restore experiments.
    """

    LOG_FILENAME = "log.txt"
    EXPERIMENT_PICKLE_FILENAME = "experiment.pkl"

    def __init__(
        self,
        config: ExperimentConfig,
        env_factory: EnvFactory,
        agent_factory: AgentFactory,
        sampling_config: SamplingConfig,
        name: str,
        logger_factory: LoggerFactory | None = None,
    ):
        if logger_factory is None:
            logger_factory = LoggerFactoryDefault()
        self.config = config
        self.sampling_config = sampling_config
        self.env_factory = env_factory
        self.agent_factory = agent_factory
        self.logger_factory = logger_factory
        self.name = name

    @classmethod
    def from_directory(cls, directory: str, restore_policy: bool = True) -> "Experiment":
        """Restores an experiment from a previously stored pickle.

        :param directory: persistence directory of a previous run, in which a pickled experiment is found
        :param restore_policy: whether the experiment shall be configured to restore the policy that was
            persisted in the given directory
        """
        with open(os.path.join(directory, cls.EXPERIMENT_PICKLE_FILENAME), "rb") as f:
            experiment: Experiment = pickle.load(f)
        if restore_policy:
            experiment.config.policy_restore_directory = directory
        return experiment

    def get_seeding_info_as_str(self) -> str:
        """Returns information on the seeds used in the experiment as a string.

        This can be useful for creating unique experiment names based on seeds, e.g.
        A typical example is to do `experiment.name = f"{experiment.name}_{experiment.get_seeding_info_as_str()}"`.
        """
        return "_".join(
            [
                f"exp_seed={self.config.seed}",
                f"train_seed={self.sampling_config.train_seed}",
                f"test_seed={self.sampling_config.test_seed}",
            ],
        )

    def _set_seed(self) -> None:
        seed = self.config.seed
        log.info(f"Setting random seed {seed}")
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _build_config_dict(self) -> dict:
        return {"experiment": self.pprints()}

    def save(self, directory: str) -> None:
        path = os.path.join(directory, self.EXPERIMENT_PICKLE_FILENAME)
        log.info(
            f"Saving serialized experiment in {path}; can be restored via Experiment.from_directory('{directory}')",
        )
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def create_experiment_world(
        self,
        override_experiment_name: str | None = None,
        logger_run_id: str | None = None,
        raise_error_on_dirname_collision: bool = True,
        reset_collectors: bool = True,
    ) -> World:
        """Creates the world object for the experiment.

        The world object contains all relevant instances for the experiment,
        such as environments, policy, collectors, etc.
        This method is the main entrypoint for users who don't want to use `run` directly. A common use case
        is that some configuration or custom logic should happen before the training loop starts, but one
        still wants to use the convenience of high-level interfaces for setting up the experiment.

        :param override_experiment_name: pass to override the experiment name in the resulting `World`.
            Affects the name of the persistence directory and logger configuration. If None, the experiment's
            name will be used.
            The name may contain path separators (i.e. `os.path.sep`, as used by `os.path.join`), in which case
            a nested directory structure will be created.
        :param logger_run_id: Run identifier to use for logger initialization/resumption.
        :param raise_error_on_dirname_collision: whether to raise an error on collisions when creating the
            persistence directory. Only takes effect if persistence is enabled. Set to `False` e.g., when continuing
            a previously executed experiment with the same `persistence_base_dir` and name.
        :param reset_collectors: whether to reset the collectors before training starts.
            Setting to `False` can be useful when continuing training from a previous run with restored collectors,
            or for adding custom logic before training starts.
        """
        if override_experiment_name is not None:
            exp_name = override_experiment_name
        else:
            exp_name = self.name

        # initialize persistence directory
        use_persistence = self.config.persistence_enabled
        persistence_dir = os.path.join(self.config.persistence_base_dir, exp_name)
        if use_persistence:
            os.makedirs(persistence_dir, exist_ok=not raise_error_on_dirname_collision)

        with logging.FileLoggerContext(
            os.path.join(persistence_dir, self.LOG_FILENAME),
            enabled=use_persistence and self.config.log_file_enabled,
        ):
            # log initial information
            log.info(f"Preparing experiment world (name='{exp_name}'):\n{self.pprints()}")
            log.info(f"Working directory: {os.getcwd()}")

            self._set_seed()

            # create environments
            envs = self.env_factory.create_envs(
                self.sampling_config.num_train_envs,
                self.sampling_config.num_test_envs,
                create_watch_env=self.config.watch,
            )
            log.info(f"Created {envs}")

            # initialize persistence
            additional_persistence = PersistenceGroup(*envs.persistence, enabled=use_persistence)
            policy_persistence = PolicyPersistence(
                additional_persistence,
                enabled=use_persistence,
                mode=self.config.policy_persistence_mode,
            )
            if use_persistence:
                log.info(f"Persistence directory: {os.path.abspath(persistence_dir)}")
                self.save(persistence_dir)

            # initialize logger
            full_config = self._build_config_dict()
            full_config.update(envs.info())
            full_config["experiment_config"] = asdict(self.config)
            full_config["sampling_config"] = asdict(self.sampling_config)
            with suppress(AttributeError):
                full_config["policy_params"] = asdict(self.agent_factory.params)

            logger: TLogger
            if use_persistence:
                logger = self.logger_factory.create_logger(
                    log_dir=persistence_dir,
                    experiment_name=exp_name,
                    run_id=logger_run_id,
                    config_dict=full_config,
                )
            else:
                logger = LazyLogger()

            # create policy and collectors
            log.info("Creating policy")
            policy = self.agent_factory.create_policy(envs, self.config.device)

            log.info("Creating collectors")
            train_collector: BaseCollector | None = None
            test_collector: BaseCollector | None = None
            if self.config.train:
                train_collector, test_collector = self.agent_factory.create_train_test_collector(
                    policy,
                    envs,
                    reset_collectors=reset_collectors,
                )

            # create context object with all relevant instances (except trainer; added later)
            world = World(
                envs=envs,
                policy=policy,
                train_collector=train_collector,
                test_collector=test_collector,
                logger=logger,
                persist_directory=persistence_dir,
                restore_directory=self.config.policy_restore_directory,
            )

            # restore policy parameters if applicable
            if self.config.policy_restore_directory:
                policy_persistence.restore(
                    policy,
                    world,
                    self.config.device,
                )

            if self.config.train:
                trainer = self.agent_factory.create_trainer(world, policy_persistence)
                world.trainer = trainer

        return world

    def run(
        self,
        run_name: str | None = None,
        logger_run_id: str | None = None,
        raise_error_on_dirname_collision: bool = True,
        **kwargs: dict[str, Any],
    ) -> ExperimentResult:
        """Run the experiment and return the results.

        :param run_name: Defines a name for this run of the experiment, which determines
            the subdirectory (within the persistence base directory) where all results will be saved.
            If None, the experiment's name will be used.
            The name may contain path separators (i.e. `os.path.sep`, as used by `os.path.join`), in which case
            a nested directory structure will be created.
        :param logger_run_id: Run identifier to use for logger initialization/resumption (applies when
            using wandb, in particular).
        :param raise_error_on_dirname_collision: set to `False` e.g., when continuing a previously executed
            experiment with the same name.
        :param kwargs: for backwards compatibility with old parameter names only
        :return:
        """
        # backward compatibility
        _experiment_name = kwargs.pop("experiment_name", None)
        if _experiment_name is not None:
            run_name = cast(str, _experiment_name)
            deprecation(
                "Parameter run_name should now be used instead of experiment_name. "
                "Support for experiment_name will be removed in the future.",
            )
        assert len(kwargs) == 0, f"Received unexpected arguments: {kwargs}"

        if run_name is None:
            run_name = self.name

        world = self.create_experiment_world(
            override_experiment_name=run_name,
            logger_run_id=logger_run_id,
            raise_error_on_dirname_collision=raise_error_on_dirname_collision,
        )

        persistence_dir = world.persist_directory
        use_persistence = self.config.persistence_enabled

        with logging.FileLoggerContext(
            os.path.join(persistence_dir, self.LOG_FILENAME),
            enabled=use_persistence and self.config.log_file_enabled,
        ):
            trainer_result: InfoStats | None = None
            if self.config.train:
                assert world.trainer is not None
                assert world.train_collector is not None
                assert world.test_collector is not None

                # prefilling buffers with either random or current agent's actions
                if self.sampling_config.start_timesteps > 0:
                    log.info(
                        f"Collecting {self.sampling_config.start_timesteps} initial environment "
                        f"steps before training (random={self.sampling_config.start_timesteps_random})",
                    )
                    world.train_collector.collect(
                        n_step=self.sampling_config.start_timesteps,
                        random=self.sampling_config.start_timesteps_random,
                    )

                log.info("Starting training")
                world.trainer.run()
                if use_persistence:
                    world.logger.finalize()
                log.info(f"Training result:\n{pformat(trainer_result)}")

            # watch agent performance
            if self.config.watch:
                assert world.envs.watch_env is not None
                log.info("Watching agent performance")
                self._watch_agent(
                    self.config.watch_num_episodes,
                    world.policy,
                    world.envs.watch_env,
                    self.config.watch_render,
                )

            return ExperimentResult(world=world, trainer_result=trainer_result)

    @staticmethod
    def _watch_agent(
        num_episodes: int,
        policy: BasePolicy,
        env: BaseVectorEnv,
        render: float,
    ) -> None:
        collector = Collector[CollectStats](policy, env)
        collector.reset()
        result = collector.collect(n_episode=num_episodes, render=render)
        assert result.returns_stat is not None  # for mypy
        assert result.lens_stat is not None  # for mypy
        log.info(
            f"Watched episodes: mean reward={result.returns_stat.mean}, mean episode length={result.lens_stat.mean}",
        )


class ExperimentCollection:
    """Shallow wrapper around a list of experiments providing a simple interface for running them with a launcher."""

    def __init__(self, experiments: list[Experiment]):
        self.experiments = experiments

    def run(
        self,
        launcher: Union["ExpLauncher", "RegisteredExpLauncher"],
    ) -> list[InfoStats | None]:
        from tianshou.evaluation.launcher import RegisteredExpLauncher

        if isinstance(launcher, RegisteredExpLauncher):
            launcher = launcher.create_launcher()
        return launcher.launch(experiments=self.experiments)


class ExperimentBuilder(ABC):
    """A helper class (following the builder pattern) for creating experiments.

    It contains a lot of defaults for the setup which can be adjusted using the
    various `with_` methods. For example, the default optimizer is Adam, but can be
    adjusted using :meth:`with_optim_factory`. Moreover, for simply configuring the default
    optimizer instead of using a different one, one can use :meth:`with_optim_factory_default`.
    """

    def __init__(
        self,
        env_factory: EnvFactory,
        experiment_config: ExperimentConfig | None = None,
        sampling_config: SamplingConfig | None = None,
    ):
        """:param env_factory: controls how environments are to be created.
        :param experiment_config: the configuration for the experiment. If None, will use the default values
            of `ExperimentConfig`.
        :param sampling_config: the sampling configuration to use. If None, will use the default values
            of `SamplingConfig`.
        """
        if experiment_config is None:
            experiment_config = ExperimentConfig()
        if sampling_config is None:
            sampling_config = SamplingConfig()

        self._config = experiment_config
        self._env_factory = env_factory
        self._sampling_config = sampling_config
        self._logger_factory: LoggerFactory | None = None
        self._optim_factory: OptimizerFactory | None = None
        self._policy_wrapper_factory: PolicyWrapperFactory | None = None
        self._trainer_callbacks: TrainerCallbacks = TrainerCallbacks()
        self._name: str = self.__class__.__name__.replace("Builder", "") + "_" + datetime_tag()

    def copy(self) -> Self:
        return deepcopy(self)

    @property
    def experiment_config(self) -> ExperimentConfig:
        return self._config

    @experiment_config.setter
    def experiment_config(self, experiment_config: ExperimentConfig) -> None:
        self._config = experiment_config

    @property
    def sampling_config(self) -> SamplingConfig:
        return self._sampling_config

    @sampling_config.setter
    def sampling_config(self, sampling_config: SamplingConfig) -> None:
        self._sampling_config = sampling_config

    def with_logger_factory(self, logger_factory: LoggerFactory) -> Self:
        """Allows to customize the logger factory to use.

        If this method is not called, the default logger factory :class:`LoggerFactoryDefault` will be used.

        :param logger_factory: the factory to use
        :return: the builder
        """
        self._logger_factory = logger_factory
        return self

    def with_policy_wrapper_factory(self, policy_wrapper_factory: PolicyWrapperFactory) -> Self:
        """Allows to define a wrapper around the policy that is created, extending the original policy.

        :param policy_wrapper_factory: the factory for the wrapper
        :return: the builder
        """
        self._policy_wrapper_factory = policy_wrapper_factory
        return self

    def with_optim_factory(self, optim_factory: OptimizerFactory) -> Self:
        """Allows to customize the gradient-based optimizer to use.

        By default, :class:`OptimizerFactoryAdam` will be used with default parameters.

        :param optim_factory: the optimizer factory
        :return: the builder
        """
        self._optim_factory = optim_factory
        return self

    def with_optim_factory_default(
        self,
        # Keep values in sync with default values in OptimizerFactoryAdam
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

    def with_epoch_train_callback(self, callback: EpochTrainCallback) -> Self:
        """Allows to define a callback function which is called at the beginning of every epoch during training.

        :param callback: the callback
        :return: the builder
        """
        self._trainer_callbacks.epoch_train_callback = callback
        return self

    def with_epoch_test_callback(self, callback: EpochTestCallback) -> Self:
        """Allows to define a callback function which is called at the beginning of testing in each epoch.

        :param callback: the callback
        :return: the builder
        """
        self._trainer_callbacks.epoch_test_callback = callback
        return self

    def with_epoch_stop_callback(self, callback: EpochStopCallback) -> Self:
        """Allows to define a callback that decides whether training shall stop early.

        The callback receives the undiscounted returns of the testing result.

        :param callback: the callback
        :return: the builder
        """
        self._trainer_callbacks.epoch_stop_callback = callback
        return self

    def with_name(
        self,
        name: str,
    ) -> Self:
        """Sets the name of the experiment.

        :param name: the name to use for this experiment, which, when the experiment is run,
            will determine the storage sub-folder by default
        :return: the builder
        """
        self._name = name
        return self

    @abstractmethod
    def _create_agent_factory(self) -> AgentFactory:
        pass

    def _get_optim_factory(self) -> OptimizerFactory:
        if self._optim_factory is None:
            # same mechanism as in `with_optim_factory_default`
            return OptimizerFactoryAdam()
        else:
            return self._optim_factory

    def build(self) -> Experiment:
        """Creates the experiment based on the options specified via this builder.

        :return: the experiment
        """
        agent_factory = self._create_agent_factory()
        agent_factory.set_trainer_callbacks(self._trainer_callbacks)
        if self._policy_wrapper_factory:
            agent_factory.set_policy_wrapper_factory(self._policy_wrapper_factory)
        experiment: Experiment = Experiment(
            config=self._config,
            env_factory=self._env_factory,
            agent_factory=agent_factory,
            sampling_config=self._sampling_config,
            name=self._name,
            logger_factory=self._logger_factory,
        )
        return experiment

    def build_seeded_collection(self, num_experiments: int) -> ExperimentCollection:
        """Creates a collection of experiments with non-overlapping random seeds, starting from the configured seed.

        Useful for performing statistically meaningful evaluations of an algorithm's performance.
        The `rliable` recommendation is to use at least 5 experiments for computing quantities such as the
        interquantile mean and performance profiles. See the usage in example scripts
        like `examples/mujoco/mujoco_ppo_hl_multi.py`.

        Each experiment in the collection will have a unique name created from the original experiment name
        and the seeds used.
        """
        num_train_envs = self.sampling_config.num_train_envs

        seeded_experiments = []
        for i in range(num_experiments):
            builder = self.copy()
            builder.experiment_config.seed += i
            builder.sampling_config.train_seed += i * num_train_envs
            experiment = builder.build()
            experiment.name += f"_{experiment.get_seeding_info_as_str()}"
            seeded_experiments.append(experiment)
        return ExperimentCollection(seeded_experiments)


class RandomActionExperimentBuilder(ExperimentBuilder):
    def _create_agent_factory(self) -> RandomActionAgentFactory:
        return RandomActionAgentFactory(
            sampling_config=self.sampling_config,
            optim_factory=self._get_optim_factory(),
        )


class _BuilderMixinActorFactory(ActorFutureProviderProtocol):
    def __init__(self, continuous_actor_type: ContinuousActorType):
        self._continuous_actor_type = continuous_actor_type
        self._actor_future = ActorFuture()
        self._actor_factory: ActorFactory | None = None

    def with_actor_factory(self, actor_factory: ActorFactory) -> Self:
        """Allows customizing the actor component via the specification of a factory.

        If this function is not called, a default actor factory (with default parameters) will be used.

        :param actor_factory: the factory to use for the creation of the actor network
        :return: the builder
        """
        self._actor_factory = actor_factory
        return self

    def _with_actor_factory_default(
        self,
        hidden_sizes: Sequence[int],
        hidden_activation: ModuleType = torch.nn.ReLU,
        continuous_unbounded: bool = False,
        continuous_conditioned_sigma: bool = False,
    ) -> Self:
        """Adds a default actor factory with the given parameters.

        :param hidden_sizes: the sequence of hidden dimensions to use in the network structure
        :param continuous_unbounded: whether, for continuous action spaces, to apply tanh activation on final logits
        :param continuous_conditioned_sigma: whether, for continuous action spaces, the standard deviation of continuous actions (sigma)
            shall be computed from the input; if False, sigma is an independent parameter.
        :return: the builder
        """
        self._actor_factory = ActorFactoryDefault(
            self._continuous_actor_type,
            hidden_sizes,
            hidden_activation=hidden_activation,
            continuous_unbounded=continuous_unbounded,
            continuous_conditioned_sigma=continuous_conditioned_sigma,
        )
        return self

    def get_actor_future(self) -> ActorFuture:
        """:return: an object, which, in the future, will contain the actor instance that is created for the experiment."""
        return self._actor_future

    def _get_actor_factory(self) -> ActorFactory:
        actor_factory: ActorFactory
        if self._actor_factory is None:
            actor_factory = ActorFactoryDefault(self._continuous_actor_type)
        else:
            actor_factory = self._actor_factory
        return ActorFactoryTransientStorageDecorator(actor_factory, self._actor_future)


class _BuilderMixinActorFactory_ContinuousGaussian(_BuilderMixinActorFactory):
    """Specialization of the actor mixin where, in the continuous case, the actor component outputs Gaussian distribution parameters."""

    def __init__(self) -> None:
        super().__init__(ContinuousActorType.GAUSSIAN)

    def with_actor_factory_default(
        self,
        hidden_sizes: Sequence[int],
        hidden_activation: ModuleType = torch.nn.ReLU,
        continuous_unbounded: bool = False,
        continuous_conditioned_sigma: bool = False,
    ) -> Self:
        """Defines use of the default actor factory, allowing its parameters it to be customized.

        The default actor factory uses an MLP-style architecture.

        :param hidden_sizes: dimensions of hidden layers used by the network
        :param hidden_activation: the activation function to use for hidden layers
        :param continuous_unbounded: whether, for continuous action spaces, to apply tanh activation on final logits
        :param continuous_conditioned_sigma: whether, for continuous action spaces, the standard deviation of continuous actions (sigma)
            shall be computed from the input; if False, sigma is an independent parameter.
        :return: the builder
        """
        return super()._with_actor_factory_default(
            hidden_sizes,
            hidden_activation=hidden_activation,
            continuous_unbounded=continuous_unbounded,
            continuous_conditioned_sigma=continuous_conditioned_sigma,
        )


class _BuilderMixinActorFactory_ContinuousDeterministic(_BuilderMixinActorFactory):
    """Specialization of the actor mixin where, in the continuous case, the actor uses a deterministic policy."""

    def __init__(self) -> None:
        super().__init__(ContinuousActorType.DETERMINISTIC)

    def with_actor_factory_default(
        self,
        hidden_sizes: Sequence[int],
        hidden_activation: ModuleType = torch.nn.ReLU,
    ) -> Self:
        """Defines use of the default actor factory, allowing its parameters it to be customized.

        The default actor factory uses an MLP-style architecture.

        :param hidden_sizes: dimensions of hidden layers used by the network
        :param hidden_activation: the activation function to use for hidden layers
        :return: the builder
        """
        return super()._with_actor_factory_default(hidden_sizes, hidden_activation)


class _BuilderMixinCriticsFactory:
    def __init__(self, num_critics: int, actor_future_provider: ActorFutureProviderProtocol):
        self._actor_future_provider = actor_future_provider
        self._critic_factories: list[CriticFactory | None] = [None] * num_critics

    def _with_critic_factory(self, idx: int, critic_factory: CriticFactory) -> Self:
        self._critic_factories[idx] = critic_factory
        return self

    def _with_critic_factory_default(
        self,
        idx: int,
        hidden_sizes: Sequence[int],
        hidden_activation: ModuleType = torch.nn.ReLU,
    ) -> Self:
        self._critic_factories[idx] = CriticFactoryDefault(
            hidden_sizes,
            hidden_activation=hidden_activation,
        )
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
    def __init__(self, actor_future_provider: ActorFutureProviderProtocol) -> None:
        super().__init__(1, actor_future_provider)

    def with_critic_factory(self, critic_factory: CriticFactory) -> Self:
        """Specifies that the given factory shall be used for the critic.

        :param critic_factory: the critic factory
        :return: the builder
        """
        self._with_critic_factory(0, critic_factory)
        return self

    def with_critic_factory_default(
        self,
        hidden_sizes: Sequence[int] = CriticFactoryDefault.DEFAULT_HIDDEN_SIZES,
        hidden_activation: ModuleType = torch.nn.ReLU,
    ) -> Self:
        """Makes the critic use the default, MLP-style architecture with the given parameters.

        :param hidden_sizes: the sequence of dimensions to use in hidden layers of the network
        :param hidden_activation: the activation function to use for hidden layers
        :return: the builder
        """
        self._with_critic_factory_default(0, hidden_sizes, hidden_activation)
        return self


class _BuilderMixinSingleCriticCanUseActorFactory(_BuilderMixinSingleCriticFactory):
    def __init__(self, actor_future_provider: ActorFutureProviderProtocol) -> None:
        super().__init__(actor_future_provider)

    def with_critic_factory_use_actor(self) -> Self:
        """Makes the first critic reuse the actor's preprocessing network (parameter sharing)."""
        return self._with_critic_factory_use_actor(0)


class _BuilderMixinDualCriticFactory(_BuilderMixinCriticsFactory):
    def __init__(self, actor_future_provider: ActorFutureProviderProtocol) -> None:
        super().__init__(2, actor_future_provider)

    def with_common_critic_factory(self, critic_factory: CriticFactory) -> Self:
        """Specifies that the given factory shall be used for both critics.

        :param critic_factory: the critic factory
        :return: the builder
        """
        for i in range(len(self._critic_factories)):
            self._with_critic_factory(i, critic_factory)
        return self

    def with_common_critic_factory_default(
        self,
        hidden_sizes: Sequence[int] = CriticFactoryDefault.DEFAULT_HIDDEN_SIZES,
        hidden_activation: ModuleType = torch.nn.ReLU,
    ) -> Self:
        """Makes both critics use the default, MLP-style architecture with the given parameters.

        :param hidden_sizes: the sequence of dimensions to use in hidden layers of the network
        :param hidden_activation: the activation function to use for hidden layers
        :return: the builder
        """
        for i in range(len(self._critic_factories)):
            self._with_critic_factory_default(i, hidden_sizes, hidden_activation)
        return self

    def with_common_critic_factory_use_actor(self) -> Self:
        """Makes both critics reuse the actor's preprocessing network (parameter sharing)."""
        for i in range(len(self._critic_factories)):
            self._with_critic_factory_use_actor(i)
        return self

    def with_critic1_factory(self, critic_factory: CriticFactory) -> Self:
        """Specifies that the given factory shall be used for the first critic.

        :param critic_factory: the critic factory
        :return: the builder
        """
        self._with_critic_factory(0, critic_factory)
        return self

    def with_critic1_factory_default(
        self,
        hidden_sizes: Sequence[int] = CriticFactoryDefault.DEFAULT_HIDDEN_SIZES,
        hidden_activation: ModuleType = torch.nn.ReLU,
    ) -> Self:
        """Makes the first critic use the default, MLP-style architecture with the given parameters.

        :param hidden_sizes: the sequence of dimensions to use in hidden layers of the network
        :param hidden_activation: the activation function to use for hidden layers
        :return: the builder
        """
        self._with_critic_factory_default(0, hidden_sizes, hidden_activation)
        return self

    def with_critic1_factory_use_actor(self) -> Self:
        """Makes the first critic reuse the actor's preprocessing network (parameter sharing)."""
        return self._with_critic_factory_use_actor(0)

    def with_critic2_factory(self, critic_factory: CriticFactory) -> Self:
        """Specifies that the given factory shall be used for the second critic.

        :param critic_factory: the critic factory
        :return: the builder
        """
        self._with_critic_factory(1, critic_factory)
        return self

    def with_critic2_factory_default(
        self,
        hidden_sizes: Sequence[int] = CriticFactoryDefault.DEFAULT_HIDDEN_SIZES,
        hidden_activation: ModuleType = torch.nn.ReLU,
    ) -> Self:
        """Makes the second critic use the default, MLP-style architecture with the given parameters.

        :param hidden_sizes: the sequence of dimensions to use in hidden layers of the network
        :param hidden_activation: the activation function to use for hidden layers
        :return: the builder
        """
        self._with_critic_factory_default(1, hidden_sizes, hidden_activation)
        return self

    def with_critic2_factory_use_actor(self) -> Self:
        """Makes the first critic reuse the actor's preprocessing network (parameter sharing)."""
        return self._with_critic_factory_use_actor(1)


class _BuilderMixinCriticEnsembleFactory:
    def __init__(self) -> None:
        self.critic_ensemble_factory: CriticEnsembleFactory | None = None

    def with_critic_ensemble_factory(self, factory: CriticEnsembleFactory) -> Self:
        """Specifies that the given factory shall be used for the critic ensemble.

        If unspecified, the default factory (:class:`CriticEnsembleFactoryDefault`) is used.

        :param factory: the critic ensemble factory
        :return: the builder
        """
        self.critic_ensemble_factory = factory
        return self

    def with_critic_ensemble_factory_default(
        self,
        hidden_sizes: Sequence[int] = CriticFactoryDefault.DEFAULT_HIDDEN_SIZES,
    ) -> Self:
        """Allows to customize the parameters of the default critic ensemble factory.

        :param hidden_sizes: the sequence of sizes of hidden layers in the network architecture
        :return: the builder
        """
        self.critic_ensemble_factory = CriticEnsembleFactoryDefault(hidden_sizes)
        return self

    def _get_critic_ensemble_factory(self) -> CriticEnsembleFactory:
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
):
    def __init__(
        self,
        env_factory: EnvFactory,
        experiment_config: ExperimentConfig | None = None,
        sampling_config: SamplingConfig | None = None,
    ):
        super().__init__(env_factory, experiment_config, sampling_config)
        self._params: DQNParams = DQNParams()
        self._model_factory: IntermediateModuleFactory = IntermediateModuleFactoryFromActorFactory(
            ActorFactoryDefault(ContinuousActorType.UNSUPPORTED, discrete_softmax=False),
        )

    def with_dqn_params(self, params: DQNParams) -> Self:
        self._params = params
        return self

    def with_model_factory(self, module_factory: IntermediateModuleFactory) -> Self:
        """:param module_factory: factory for a module which maps environment observations to a vector of Q-values (one for each action)
        :return: the builder
        """
        self._model_factory = module_factory
        return self

    def with_model_factory_default(
        self,
        hidden_sizes: Sequence[int],
        hidden_activation: ModuleType = torch.nn.ReLU,
    ) -> Self:
        """Allows to configure the default factory for the model of the Q function, which maps environment observations to a vector of
        Q-values (one for each action). The default model is a multi-layer perceptron.

        :param hidden_sizes: the sequence of dimensions used for hidden layers
        :param hidden_activation: the activation function to use for hidden layers (not used for the output layer)
        :return: the builder
        """
        self._model_factory = IntermediateModuleFactoryFromActorFactory(
            ActorFactoryDefault(
                ContinuousActorType.UNSUPPORTED,
                hidden_sizes=hidden_sizes,
                hidden_activation=hidden_activation,
                discrete_softmax=False,
            ),
        )
        return self

    def _create_agent_factory(self) -> AgentFactory:
        return DQNAgentFactory(
            self._params,
            self._sampling_config,
            self._model_factory,
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
        self._preprocess_network_factory: IntermediateModuleFactory = (
            IntermediateModuleFactoryFromActorFactory(
                ActorFactoryDefault(ContinuousActorType.UNSUPPORTED, discrete_softmax=False),
            )
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
