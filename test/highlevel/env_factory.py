import gymnasium as gym

from tianshou.env import DummyVectorEnv
from tianshou.highlevel.env import (
    ContinuousEnvironments,
    DiscreteEnvironments,
    EnvFactory,
    Environments,
)


class DiscreteTestEnvFactory(EnvFactory):
    def create_envs(self, num_training_envs: int, num_test_envs: int) -> Environments:
        task = "CartPole-v0"
        env = gym.make(task)
        train_envs = DummyVectorEnv([lambda: gym.make(task) for _ in range(num_training_envs)])
        test_envs = DummyVectorEnv([lambda: gym.make(task) for _ in range(num_test_envs)])
        return DiscreteEnvironments(env, train_envs, test_envs)


class ContinuousTestEnvFactory(EnvFactory):
    def create_envs(self, num_training_envs: int, num_test_envs: int) -> Environments:
        task = "Pendulum-v1"
        env = gym.make(task)
        train_envs = DummyVectorEnv([lambda: gym.make(task) for _ in range(num_training_envs)])
        test_envs = DummyVectorEnv([lambda: gym.make(task) for _ in range(num_test_envs)])
        return ContinuousEnvironments(env, train_envs, test_envs)
