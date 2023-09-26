"""Factories for the generation of environment-dependent parameters."""
from abc import ABC, abstractmethod
from typing import TypeVar

from tianshou.highlevel.env import ContinuousEnvironments, Environments

T = TypeVar("T")


class FloatEnvParamFactory(ABC):
    @abstractmethod
    def create_param(self, envs: Environments) -> float:
        pass


class MaxActionScaledFloatEnvParamFactory(FloatEnvParamFactory):
    def __init__(self, value: float):
        """:param value: value with which to scale the max action value"""
        self.value = value

    def create_param(self, envs: Environments) -> float:
        envs.get_type().assert_continuous(self)
        envs: ContinuousEnvironments
        return envs.max_action * self.value
