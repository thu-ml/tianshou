"""Factories for the generation of environment-dependent parameters."""
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from tianshou.highlevel.env import ContinuousEnvironments, Environments
from tianshou.utils.string import ToStringMixin

TValue = TypeVar("TValue")
TEnvs = TypeVar("TEnvs", bound=Environments)


class EnvValueFactory(Generic[TValue, TEnvs], ToStringMixin, ABC):
    @abstractmethod
    def create_value(self, envs: TEnvs) -> TValue:
        pass


class FloatEnvValueFactory(EnvValueFactory[float, TEnvs], Generic[TEnvs], ABC):
    """Serves as a type bound for float value factories."""


class FloatEnvValueFactoryMaxActionScaled(FloatEnvValueFactory[ContinuousEnvironments]):
    def __init__(self, value: float):
        """:param value: value with which to scale the max action value"""
        self.value = value

    def create_value(self, envs: ContinuousEnvironments) -> float:
        envs.get_type().assert_continuous(self)
        return envs.max_action * self.value


class MaxActionScaled(FloatEnvValueFactoryMaxActionScaled):
    pass
