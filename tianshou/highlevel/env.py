from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from typing import Any

import gymnasium as gym

from tianshou.env import BaseVectorEnv
from tianshou.highlevel.persistence import PersistableConfigProtocol

TShape = int | Sequence[int]


class EnvType(Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"

    def is_discrete(self):
        return self == EnvType.DISCRETE

    def is_continuous(self):
        return self == EnvType.CONTINUOUS

    def assert_continuous(self, requiring_entity: Any):
        if not self.is_continuous():
            raise AssertionError(f"{requiring_entity} requires continuous environments")

    def assert_discrete(self, requiring_entity: Any):
        if not self.is_discrete():
            raise AssertionError(f"{requiring_entity} requires discrete environments")


class Environments(ABC):
    def __init__(self, env: gym.Env | None, train_envs: BaseVectorEnv, test_envs: BaseVectorEnv):
        self.env = env
        self.train_envs = train_envs
        self.test_envs = test_envs

    def info(self) -> dict[str, Any]:
        return {
            "action_shape": self.get_action_shape(),
            "state_shape": self.get_observation_shape(),
        }

    @abstractmethod
    def get_action_shape(self) -> TShape:
        pass

    @abstractmethod
    def get_observation_shape(self) -> TShape:
        pass

    def get_action_space(self) -> gym.Space:
        return self.env.action_space

    def get_observation_space(self) -> gym.Space:
        return self.env.observation_space

    @abstractmethod
    def get_type(self) -> EnvType:
        pass


class ContinuousEnvironments(Environments):
    def __init__(self, env: gym.Env | None, train_envs: BaseVectorEnv, test_envs: BaseVectorEnv):
        super().__init__(env, train_envs, test_envs)
        self.state_shape, self.action_shape, self.max_action = self._get_continuous_env_info(env)

    def info(self):
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
        state_shape = env.observation_space.shape or env.observation_space.n
        if not state_shape:
            raise ValueError("Observation space shape is not defined")
        action_shape = env.action_space.shape
        max_action = env.action_space.high[0]
        return state_shape, action_shape, max_action

    def get_action_shape(self) -> TShape:
        return self.action_shape

    def get_observation_shape(self) -> TShape:
        return self.state_shape

    def get_type(self) -> EnvType:
        return EnvType.CONTINUOUS


class DiscreteEnvironments(Environments):
    def __init__(self, env: gym.Env | None, train_envs: BaseVectorEnv, test_envs: BaseVectorEnv):
        super().__init__(env, train_envs, test_envs)
        self.observation_shape = env.observation_space.shape or env.observation_space.n
        self.action_shape = env.action_space.shape or env.action_space.n

    def get_action_shape(self) -> TShape:
        return self.action_shape

    def get_observation_shape(self) -> TShape:
        return self.observation_shape

    def get_type(self) -> EnvType:
        return EnvType.DISCRETE


class EnvFactory(ABC):
    @abstractmethod
    def create_envs(self, config: PersistableConfigProtocol | None = None) -> Environments:
        pass

    def __call__(self, config: PersistableConfigProtocol | None = None) -> Environments:
        return self.create_envs(config=config)
