import gymnasium as gym

from tianshou.env import DummyVectorEnv
from tianshou.highlevel.env import (
    ContinuousEnvironments,
    DiscreteEnvironments,
    EnvFactory,
    Environments,
)
from tianshou.highlevel.persistence import PersistableConfigProtocol


class DiscreteTestEnvFactory(EnvFactory):
    def __init__(self, test_num=10, train_num=10):
        self.test_num = test_num
        self.train_num = train_num

    def create_envs(self, config: PersistableConfigProtocol | None = None) -> Environments:
        task = "CartPole-v0"
        env = gym.make(task)
        train_envs = DummyVectorEnv([lambda: gym.make(task) for _ in range(self.train_num)])
        test_envs = DummyVectorEnv([lambda: gym.make(task) for _ in range(self.test_num)])
        return DiscreteEnvironments(env, train_envs, test_envs)


class ContinuousTestEnvFactory(EnvFactory):
    def __init__(self, test_num=10, train_num=10):
        self.test_num = test_num
        self.train_num = train_num

    def create_envs(self, config: PersistableConfigProtocol | None = None) -> Environments:
        task = "Pendulum-v1"
        env = gym.make(task)
        train_envs = DummyVectorEnv([lambda: gym.make(task) for _ in range(self.train_num)])
        test_envs = DummyVectorEnv([lambda: gym.make(task) for _ in range(self.test_num)])
        return ContinuousEnvironments(env, train_envs, test_envs)
