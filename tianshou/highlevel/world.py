import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from tianshou.data import BaseCollector
    from tianshou.highlevel.env import Environments
    from tianshou.highlevel.logger import TLogger
    from tianshou.policy import BasePolicy
    from tianshou.trainer import BaseTrainer


@dataclass(kw_only=True)
class World:
    """Container for instances and configuration items that are relevant to an experiment."""

    envs: "Environments"
    policy: "BasePolicy"
    train_collector: Optional["BaseCollector"] = None
    test_collector: Optional["BaseCollector"] = None
    logger: "TLogger"
    persist_directory: str
    restore_directory: str | None
    trainer: Optional["BaseTrainer"] = None

    def persist_path(self, filename: str) -> str:
        return os.path.abspath(os.path.join(self.persist_directory, filename))

    def restore_path(self, filename: str) -> str:
        if self.restore_directory is None:
            raise ValueError(
                "Path cannot be formed because no directory for restoration was provided",
            )
        return os.path.join(self.restore_directory, filename)
