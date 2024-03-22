import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from enum import Enum
from typing import Any, TypeAlias, cast

import gymnasium as gym
import gymnasium.spaces
from gymnasium import Env

from tianshou.env import (
    BaseVectorEnv,
    DummyVectorEnv,
    RayVectorEnv,
    SubprocVectorEnv,
)
from tianshou.highlevel.persistence import Persistence
from tianshou.utils.net.common import TActionShape
from tianshou.utils.string import ToStringMixin

TObservationShape: TypeAlias = int | Sequence[int]

log = logging.getLogger(__name__)


class EnvType(Enum):
    """Enumeration of environment types."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"

    def is_discrete(self) -> bool:
        return self == EnvType.DISCRETE

    def is_continuous(self) -> bool:
        return self == EnvType.CONTINUOUS

    def assert_continuous(self, requiring_entity: Any) -> None:
        if not self.is_continuous():
            raise AssertionError(f"{requiring_entity} requires continuous environments")

    def assert_discrete(self, requiring_entity: Any) -> None:
        if not self.is_discrete():
            raise AssertionError(f"{requiring_entity} requires discrete environments")

    @staticmethod
    def from_env(env: Env) -> "EnvType":
        if isinstance(env.action_space, gymnasium.spaces.Discrete):
            return EnvType.DISCRETE
        elif isinstance(env.action_space, gymnasium.spaces.Box):
            return EnvType.CONTINUOUS
        else:
            raise Exception(f"Unsupported environment type with action space {env.action_space}")


class EnvMode(Enum):
    """Indicates the purpose for which an environment is created."""

    TRAIN = "train"
    TEST = "test"
    WATCH = "watch"


class VectorEnvType(Enum):
    DUMMY = "dummy"
    """Vectorized environment without parallelization; environments are processed sequentially"""
    SUBPROC = "subproc"
    """Parallelization based on `subprocess`"""
    SUBPROC_SHARED_MEM = "shmem"
    """Parallelization based on `subprocess` with shared memory"""
    SUBPROC_SHARED_MEM_FORK_CONTEXT = "shmem_fork"
    """Parallelization based on `subprocess` with shared memory and fork context (relevant for macOS, which uses `spawn`
     by default https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods)"""
    RAY = "ray"
    """Parallelization based on the `ray` library"""

    def create_venv(
        self,
        factories: Sequence[Callable[[], gym.Env]],
    ) -> BaseVectorEnv:
        match self:
            case VectorEnvType.DUMMY:
                return DummyVectorEnv(factories)
            case VectorEnvType.SUBPROC:
                return SubprocVectorEnv(factories)
            case VectorEnvType.SUBPROC_SHARED_MEM:
                return SubprocVectorEnv(factories, share_memory=True)
            case VectorEnvType.SUBPROC_SHARED_MEM_FORK_CONTEXT:
                return SubprocVectorEnv(factories, share_memory=True, context="fork")
            case VectorEnvType.RAY:
                return RayVectorEnv(factories)
            case _:
                raise NotImplementedError(self)


class Environments(ToStringMixin, ABC):
    """Represents (vectorized) environments for a learning process."""

    def __init__(
        self,
        env: gym.Env,
        train_envs: BaseVectorEnv,
        test_envs: BaseVectorEnv,
        watch_env: BaseVectorEnv | None = None,
    ):
        self.env = env
        self.train_envs = train_envs
        self.test_envs = test_envs
        self.watch_env = watch_env
        self.persistence: Sequence[Persistence] = []

    @staticmethod
    def from_factory_and_type(
        factory_fn: Callable[[EnvMode], gym.Env],
        env_type: EnvType,
        venv_type: VectorEnvType,
        num_training_envs: int,
        num_test_envs: int,
        create_watch_env: bool = False,
    ) -> "Environments":
        """Creates a suitable subtype instance from a factory function that creates a single instance and the type of environment (continuous/discrete).

        :param factory_fn: the factory for a single environment instance
        :param env_type: the type of environments created by `factory_fn`
        :param venv_type: the vector environment type to use for parallelization
        :param num_training_envs: the number of training environments to create
        :param num_test_envs: the number of test environments to create
        :param create_watch_env: whether to create an environment for watching the agent
        :return: the instance
        """
        train_envs = venv_type.create_venv(
            [lambda: factory_fn(EnvMode.TRAIN)] * num_training_envs,
        )
        test_envs = venv_type.create_venv(
            [lambda: factory_fn(EnvMode.TEST)] * num_test_envs,
        )
        if create_watch_env:
            watch_env = VectorEnvType.DUMMY.create_venv([lambda: factory_fn(EnvMode.WATCH)])
        else:
            watch_env = None
        env = factory_fn(EnvMode.TRAIN)
        match env_type:
            case EnvType.CONTINUOUS:
                return ContinuousEnvironments(env, train_envs, test_envs, watch_env)
            case EnvType.DISCRETE:
                return DiscreteEnvironments(env, train_envs, test_envs, watch_env)
            case _:
                raise ValueError(f"Environment type {env_type} not handled")

    def _tostring_includes(self) -> list[str]:
        return []

    def _tostring_additional_entries(self) -> dict[str, Any]:
        return self.info()

    def info(self) -> dict[str, Any]:
        return {
            "action_shape": self.get_action_shape(),
            "state_shape": self.get_observation_shape(),
        }

    def set_persistence(self, *p: Persistence) -> None:
        """Associates the given persistence handlers which may persist and restore environment-specific information.

        :param p: persistence handlers
        """
        self.persistence = p

    @abstractmethod
    def get_action_shape(self) -> TActionShape:
        pass

    @abstractmethod
    def get_observation_shape(self) -> TObservationShape:
        pass

    def get_action_space(self) -> gym.Space:
        return self.env.action_space

    def get_observation_space(self) -> gym.Space:
        return self.env.observation_space

    @abstractmethod
    def get_type(self) -> EnvType:
        pass


class ContinuousEnvironments(Environments):
    """Represents (vectorized) continuous environments."""

    def __init__(
        self,
        env: gym.Env,
        train_envs: BaseVectorEnv,
        test_envs: BaseVectorEnv,
        watch_env: BaseVectorEnv | None = None,
    ):
        super().__init__(env, train_envs, test_envs, watch_env)
        self.state_shape, self.action_shape, self.max_action = self._get_continuous_env_info(env)

    @staticmethod
    def from_factory(
        factory_fn: Callable[[EnvMode], gym.Env],
        venv_type: VectorEnvType,
        num_training_envs: int,
        num_test_envs: int,
        create_watch_env: bool = False,
    ) -> "ContinuousEnvironments":
        """Creates an instance from a factory function that creates a single instance.

        :param factory_fn: the factory for a single environment instance
        :param venv_type: the vector environment type to use for parallelization
        :param num_training_envs: the number of training environments to create
        :param num_test_envs: the number of test environments to create
        :param create_watch_env: whether to create an environment for watching the agent
        :return: the instance
        """
        return cast(
            ContinuousEnvironments,
            Environments.from_factory_and_type(
                factory_fn,
                EnvType.CONTINUOUS,
                venv_type,
                num_training_envs,
                num_test_envs,
                create_watch_env,
            ),
        )

    def info(self) -> dict[str, Any]:
        d = super().info()
        d["max_action"] = self.max_action
        return d

    @staticmethod
    def _get_continuous_env_info(
        env: gym.Env,
    ) -> tuple[tuple[int, ...], tuple[int, ...], float]:
        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError(
                "Only environments with continuous action space are supported here. "
                f"But got env with action space: {env.action_space.__class__}.",
            )
        state_shape = env.observation_space.shape or env.observation_space.n  # type: ignore
        if not state_shape:
            raise ValueError("Observation space shape is not defined")
        action_shape = env.action_space.shape
        max_action = env.action_space.high[0]
        return state_shape, action_shape, max_action

    def get_action_shape(self) -> TActionShape:
        return self.action_shape

    def get_observation_shape(self) -> TObservationShape:
        return self.state_shape

    def get_type(self) -> EnvType:
        return EnvType.CONTINUOUS


class DiscreteEnvironments(Environments):
    """Represents (vectorized) discrete environments."""

    def __init__(
        self,
        env: gym.Env,
        train_envs: BaseVectorEnv,
        test_envs: BaseVectorEnv,
        watch_env: BaseVectorEnv | None = None,
    ):
        super().__init__(env, train_envs, test_envs, watch_env)
        self.observation_shape = env.observation_space.shape or env.observation_space.n  # type: ignore
        self.action_shape = env.action_space.shape or env.action_space.n  # type: ignore

    @staticmethod
    def from_factory(
        factory_fn: Callable[[EnvMode], gym.Env],
        venv_type: VectorEnvType,
        num_training_envs: int,
        num_test_envs: int,
        create_watch_env: bool = False,
    ) -> "DiscreteEnvironments":
        """Creates an instance from a factory function that creates a single instance.

        :param factory_fn: the factory for a single environment instance
        :param venv_type: the vector environment type to use for parallelization
        :param num_training_envs: the number of training environments to create
        :param num_test_envs: the number of test environments to create
        :param create_watch_env: whether to create an environment for watching the agent
        :return: the instance
        """
        return cast(
            DiscreteEnvironments,
            Environments.from_factory_and_type(
                factory_fn,
                EnvType.DISCRETE,
                venv_type,
                num_training_envs,
                num_test_envs,
                create_watch_env,
            ),
        )

    def get_action_shape(self) -> TActionShape:
        return self.action_shape

    def get_observation_shape(self) -> TObservationShape:
        return self.observation_shape

    def get_type(self) -> EnvType:
        return EnvType.DISCRETE


class EnvPoolFactory:
    """A factory for the creation of envpool-based vectorized environments, which can be used in conjunction
    with :class:`EnvFactoryRegistered`.
    """

    def _transform_task(self, task: str) -> str:
        return task

    def _transform_kwargs(self, kwargs: dict, mode: EnvMode) -> dict:
        """Transforms gymnasium keyword arguments to be envpool-compatible.

        :param kwargs: keyword arguments that would normally be passed to `gymnasium.make`.
        :param mode: the environment mode
        :return: the transformed keyword arguments
        """
        kwargs = dict(kwargs)
        if "render_mode" in kwargs:
            del kwargs["render_mode"]
        return kwargs

    def create_venv(
        self,
        task: str,
        num_envs: int,
        mode: EnvMode,
        seed: int,
        kwargs: dict,
    ) -> BaseVectorEnv:
        import envpool

        envpool_task = self._transform_task(task)
        envpool_kwargs = self._transform_kwargs(kwargs, mode)
        return envpool.make_gymnasium(
            envpool_task,
            num_envs=num_envs,
            seed=seed,
            **envpool_kwargs,
        )


class EnvFactory(ToStringMixin, ABC):
    """Main interface for the creation of environments (in various forms)."""

    def __init__(self, venv_type: VectorEnvType):
        """:param venv_type: the type of vectorized environment to use for train and test environments.
        watch environments are always created as dummy environments.
        """
        self.venv_type = venv_type

    @abstractmethod
    def create_env(self, mode: EnvMode) -> Env:
        pass

    def create_venv(self, num_envs: int, mode: EnvMode) -> BaseVectorEnv:
        """Create vectorized environments.

        :param num_envs: the number of environments
        :param mode: the mode for which to create. In `WATCH` mode the resulting venv will always be of type `DUMMY` with a single env.

        :return: the vectorized environments
        """
        if mode == EnvMode.WATCH:
            return VectorEnvType.DUMMY.create_venv([lambda: self.create_env(mode)])
        else:
            return self.venv_type.create_venv([lambda: self.create_env(mode)] * num_envs)

    def create_envs(
        self,
        num_training_envs: int,
        num_test_envs: int,
        create_watch_env: bool = False,
    ) -> Environments:
        """Create environments for learning.

        :param num_training_envs: the number of training environments
        :param num_test_envs: the number of test environments
        :param create_watch_env: whether to create an environment for watching the agent
        :return: the environments
        """
        env = self.create_env(EnvMode.TRAIN)
        train_envs = self.create_venv(num_training_envs, EnvMode.TRAIN)
        test_envs = self.create_venv(num_test_envs, EnvMode.TEST)
        watch_env = self.create_venv(1, EnvMode.WATCH) if create_watch_env else None
        match EnvType.from_env(env):
            case EnvType.DISCRETE:
                return DiscreteEnvironments(env, train_envs, test_envs, watch_env)
            case EnvType.CONTINUOUS:
                return ContinuousEnvironments(env, train_envs, test_envs, watch_env)
            case _:
                raise ValueError


class EnvFactoryRegistered(EnvFactory):
    """Factory for environments that are registered with gymnasium and thus can be created via `gymnasium.make`
    (or via `envpool.make_gymnasium`).
    """

    def __init__(
        self,
        *,
        task: str,
        seed: int,
        venv_type: VectorEnvType,
        envpool_factory: EnvPoolFactory | None = None,
        render_mode_train: str | None = None,
        render_mode_test: str | None = None,
        render_mode_watch: str = "human",
        **make_kwargs: Any,
    ):
        """:param task: the gymnasium task/environment identifier
        :param seed: the random seed
        :param venv_type: the type of vectorized environment to use (if `envpool_factory` is not specified)
        :param envpool_factory: the factory to use for vectorized environment creation based on envpool; envpool must be installed.
        :param render_mode_train: the render mode to use for training environments
        :param render_mode_test: the render mode to use for test environments
        :param render_mode_watch: the render mode to use for environments that are used to watch agent performance
        :param make_kwargs: additional keyword arguments to pass on to `gymnasium.make`.
            If envpool is used, the gymnasium parameters will be appropriately translated for use with
            `envpool.make_gymnasium`.
        """
        super().__init__(venv_type)
        self.task = task
        self.envpool_factory = envpool_factory
        self.seed = seed
        self.render_modes = {
            EnvMode.TRAIN: render_mode_train,
            EnvMode.TEST: render_mode_test,
            EnvMode.WATCH: render_mode_watch,
        }
        self.make_kwargs = make_kwargs

    def _create_kwargs(self, mode: EnvMode) -> dict:
        """Adapts the keyword arguments for the given mode.

        :param mode: the mode
        :return: adapted keyword arguments
        """
        kwargs = dict(self.make_kwargs)
        kwargs["render_mode"] = self.render_modes.get(mode)
        return kwargs

    def create_env(self, mode: EnvMode) -> Env:
        """Creates a single environment for the given mode.

        :param mode: the mode
        :return: an environment
        """
        kwargs = self._create_kwargs(mode)
        return gymnasium.make(self.task, **kwargs)

    def create_venv(self, num_envs: int, mode: EnvMode) -> BaseVectorEnv:
        if self.envpool_factory is not None:
            return self.envpool_factory.create_venv(
                self.task,
                num_envs,
                mode,
                self.seed,
                self._create_kwargs(mode),
            )
        else:
            venv = super().create_venv(num_envs, mode)
            venv.seed(self.seed)
            return venv
