from abc import ABC, abstractmethod

from tianshou.algorithm import Algorithm
from tianshou.data import BaseCollector, Collector, ReplayBuffer
from tianshou.env import BaseVectorEnv


class CollectorFactory(ABC):
    @abstractmethod
    def create_collector(
        self,
        algorithm: Algorithm,
        vector_env: BaseVectorEnv,
        buffer: ReplayBuffer | None = None,
        exploration_noise: bool = False,
    ) -> BaseCollector:
        """
        Creates a collector for the given algorithm and vectorized environment.

        :param algorithm: the algorithm
        :param vector_env: the vectorized environment
        :param buffer: the replay buffer to be used by the collector;
            if None, a new buffer will be created with default parameters
        :param exploration_noise: whether action shall be modified using the policy's exploration noise
        :return: the collector
        """


class CollectorFactoryDefault(CollectorFactory):
    def create_collector(
        self,
        algorithm: Algorithm,
        vector_env: BaseVectorEnv,
        buffer: ReplayBuffer | None = None,
        exploration_noise: bool = False,
    ) -> BaseCollector:
        return Collector(
            algorithm.policy, vector_env, buffer=buffer, exploration_noise=exploration_noise
        )
