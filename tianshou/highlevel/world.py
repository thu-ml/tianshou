import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from tianshou.data import Collector
    from tianshou.highlevel.env import Environments
    from tianshou.highlevel.logger import TLogger
    from tianshou.policy import BasePolicy
    from tianshou.trainer import BaseTrainer


@dataclass
class World:
    envs: "Environments"
    policy: "BasePolicy"
    train_collector: "Collector"
    test_collector: "Collector"
    logger: "TLogger"
    persist_directory: str
    restore_directory: str
    trainer: Optional["BaseTrainer"] = None

    def persist_path(self, filename: str) -> str:
        return os.path.join(self.persist_directory, filename)

    def restore_path(self, filename: str) -> str:
        return os.path.join(self.restore_directory, filename)
