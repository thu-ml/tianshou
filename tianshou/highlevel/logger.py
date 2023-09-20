import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from torch.utils.tensorboard import SummaryWriter

from tianshou.utils import TensorboardLogger, WandbLogger

TLogger = TensorboardLogger | WandbLogger


@dataclass
class Logger:
    logger: TLogger
    log_path: str


class LoggerFactory(ABC):
    @abstractmethod
    def create_logger(self, log_name: str, run_id: int | None, config_dict: dict) -> Logger:
        pass


@dataclass
class LoggerConfig:
    """Logging config."""

    logdir: str = "log"
    logger: Literal["tensorboard", "wandb"] = "tensorboard"
    wandb_project: str = "mujoco.benchmark"
    """Only used if logger is wandb."""


class DefaultLoggerFactory(LoggerFactory):
    def __init__(self, config: LoggerConfig):
        self.config = config

    def create_logger(self, log_name: str, run_id: str | None, config_dict: dict) -> Logger:
        writer = SummaryWriter(self.config.logdir)
        writer.add_text("args", str(self.config))
        if self.config.logger == "wandb":
            logger = WandbLogger(
                save_interval=1,
                name=log_name.replace(os.path.sep, "__"),
                run_id=run_id,
                config=config_dict,
                project=self.config.wandb_project,
            )
            logger.load(writer)
        elif self.config.logger == "tensorboard":
            logger = TensorboardLogger(writer)
        else:
            raise ValueError(f"Unknown logger: {self.config.logger}")
        log_path = os.path.join(self.config.logdir, log_name)
        os.makedirs(log_path, exist_ok=True)
        return Logger(logger=logger, log_path=log_path)
