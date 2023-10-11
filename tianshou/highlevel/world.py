import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from tianshou.data import Collector
    from tianshou.highlevel.env import Environments
    from tianshou.highlevel.logger import Logger
    from tianshou.policy import BasePolicy
    from tianshou.trainer import BaseTrainer


@dataclass
class World:
    envs: "Environments"
    policy: "BasePolicy"
    train_collector: "Collector"
    test_collector: "Collector"
    logger: "Logger"
    trainer: Optional["BaseTrainer"] = None

    @property
    def directory(self):
        return self.logger.log_path

    def path(self, filename: str) -> str:
        return os.path.join(self.directory, filename)
