from dataclasses import dataclass
from pprint import pprint
from typing import Generic, TypeVar

import numpy as np
import torch

from tianshou.data import Collector
from tianshou.highlevel.agent import AgentFactory
from tianshou.highlevel.env import EnvFactory
from tianshou.highlevel.logger import LoggerFactory
from tianshou.policy import BasePolicy
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


@dataclass
class RLSamplingConfig:
    """Sampling, epochs, parallelization, buffers, collectors, and batching."""

    num_epochs: int = 100
    step_per_epoch: int = 30000
    batch_size: int = 64
    num_train_envs: int = 64
    num_test_envs: int = 10
    buffer_size: int = 4096
    step_per_collect: int = 2048
    repeat_per_collect: int = 10
    update_per_step: int = 1


class RLExperiment(Generic[TPolicy, TTrainer]):
    def __init__(
        self,
        config: RLExperimentConfig,
        env_factory: EnvFactory,
        logger_factory: LoggerFactory,
        agent_factory: AgentFactory,
    ):
        self.config = config
        self.env_factory = env_factory
        self.logger_factory = logger_factory
        self.agent_factory = agent_factory

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
