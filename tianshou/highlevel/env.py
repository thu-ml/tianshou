from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from enum import Enum
from typing import Any, TypeAlias, cast

import gymnasium as gym

from tianshou.env import (
    BaseVectorEnv,
    DummyVectorEnv,
    RayVectorEnv,
    ShmemVectorEnv,
    SubprocVectorEnv,
)
from tianshou.highlevel.persistence import Persistence
from tianshou.utils.net.common import TActionShape
from tianshou.utils.string import ToStringMixin

TObservationShape: TypeAlias = int | Sequence[int]


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


class VectorEnvType(Enum):
    DUMMY = "dummy"
    """Vectorized environment without parallelization; environments are processed sequentially"""
    SUBPROC = "subproc"
    """Parallelization based on `subprocess`"""
    SUBPROC_SHARED_MEM = "shmem"
    """Parallelization based on `subprocess` with shared memory"""
    RAY = "ray"
    """Parallelization based on the `ray` library"""

    def create_venv(self, factories: list[Callable[[], gym.Env]]) -> BaseVectorEnv:
        match self:
            case VectorEnvType.DUMMY:
                return DummyVectorEnv(factories)
            case VectorEnvType.SUBPROC:
                return SubprocVectorEnv(factories)
            case VectorEnvType.SUBPROC_SHARED_MEM:
                return ShmemVectorEnv(factories)
            case VectorEnvType.RAY:
                return RayVectorEnv(factories)
            case _:
                raise NotImplementedError(self)


class Environments(ToStringMixin, ABC):
    """Represents (vectorized) environments."""

    def __init__(self, env: gym.Env, train_envs: BaseVectorEnv, test_envs: BaseVectorEnv):
        self.env = env
        self.train_envs = train_envs
        self.test_envs = test_envs
        self.persistence: Sequence[Persistence] = []

    @staticmethod
    def from_factory_and_type(
        factory_fn: Callable[[], gym.Env],
        env_type: EnvType,
        venv_type: VectorEnvType,
        num_training_envs: int,
        num_test_envs: int,
        test_factory_fn: Callable[[], gym.Env] | None = None,
    ) -> "Environments":
        """Creates a suitable subtype instance from a factory function that creates a single instance and the type of environment (continuous/discrete).

        :param factory_fn: the factory for a single environment instance
        :param env_type: the type of environments created by `factory_fn`
        :param venv_type: the vector environment type to use for parallelization
        :param num_training_envs: the number of training environments to create
        :param num_test_envs: the number of test environments to create
        :param test_factory_fn: the factory to use for the creation of test environment instances;
            if None, use `factory_fn` for all environments (train and test)
        :return: the instance
        """
        if test_factory_fn is None:
            test_factory_fn = factory_fn
        train_envs = venv_type.create_venv([factory_fn] * num_training_envs)
        test_envs = venv_type.create_venv([test_factory_fn] * num_test_envs)
        env = factory_fn()
        match env_type:
            case EnvType.CONTINUOUS:
                return ContinuousEnvironments(env, train_envs, test_envs)
            case EnvType.DISCRETE:
                return DiscreteEnvironments(env, train_envs, test_envs)
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

    def __init__(self, env: gym.Env, train_envs: BaseVectorEnv, test_envs: BaseVectorEnv):
        super().__init__(env, train_envs, test_envs)
        self.state_shape, self.action_shape, self.max_action = self._get_continuous_env_info(env)

    @staticmethod
    def from_factory(
        factory_fn: Callable[[], gym.Env],
        venv_type: VectorEnvType,
        num_training_envs: int,
        num_test_envs: int,
        test_factory_fn: Callable[[], gym.Env] | None = None,
    ) -> "ContinuousEnvironments":
        """Creates an instance from a factory function that creates a single instance.

        :param factory_fn: the factory for a single environment instance
        :param venv_type: the vector environment type to use for parallelization
        :param num_training_envs: the number of training environments to create
        :param num_test_envs: the number of test environments to create
        :param test_factory_fn: the factory to use for the creation of test environment instances;
            if None, use `factory_fn` for all environments (train and test)
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
                test_factory_fn=test_factory_fn,
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

    def __init__(self, env: gym.Env, train_envs: BaseVectorEnv, test_envs: BaseVectorEnv):
        super().__init__(env, train_envs, test_envs)
        self.observation_shape = env.observation_space.shape or env.observation_space.n  # type: ignore
        self.action_shape = env.action_space.shape or env.action_space.n  # type: ignore

    @staticmethod
    def from_factory(
        factory_fn: Callable[[], gym.Env],
        venv_type: VectorEnvType,
        num_training_envs: int,
        num_test_envs: int,
        test_factory_fn: Callable[[], gym.Env] | None = None,
    ) -> "DiscreteEnvironments":
        """Creates an instance from a factory function that creates a single instance.

        :param factory_fn: the factory for a single environment instance
        :param venv_type: the vector environment type to use for parallelization
        :param num_training_envs: the number of training environments to create
        :param num_test_envs: the number of test environments to create
        :param test_factory_fn: the factory to use for the creation of test environment instances;
            if None, use `factory_fn` for all environments (train and test)
        :return: the instance
        """
        return cast(
            DiscreteEnvironments,
            Environments.from_factory_and_type(
                factory_fn,
                EnvType.CONTINUOUS,
                venv_type,
                num_training_envs,
                num_test_envs,
                test_factory_fn=test_factory_fn,
            ),
        )

    def get_action_shape(self) -> TActionShape:
        return self.action_shape

    def get_observation_shape(self) -> TObservationShape:
        return self.observation_shape

    def get_type(self) -> EnvType:
        return EnvType.DISCRETE


class EnvFactory(ToStringMixin, ABC):
    @abstractmethod
    def create_envs(self, num_training_envs: int, num_test_envs: int) -> Environments:
        pass
