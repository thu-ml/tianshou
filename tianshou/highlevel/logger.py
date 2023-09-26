import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, TypeAlias

from torch.utils.tensorboard import SummaryWriter

from tianshou.utils import TensorboardLogger, WandbLogger

TLogger: TypeAlias = TensorboardLogger | WandbLogger


@dataclass
class Logger:
    logger: TLogger
    log_path: str


class LoggerFactory(ABC):
    @abstractmethod
    def create_logger(self, log_name: str, run_id: int | None, config_dict: dict) -> Logger:
        pass


class DefaultLoggerFactory(LoggerFactory):
    def __init__(
        self,
        log_dir: str = "log",
        logger_type: Literal["tensorboard", "wandb"] = "tensorboard",
        wandb_project: str | None = None,
    ):
        if logger_type == "wandb" and wandb_project is None:
            raise ValueError("Must provide 'wandb_project'")
        self.log_dir = log_dir
        self.logger_type = logger_type
        self.wandb_project = wandb_project

    def create_logger(self, log_name: str, run_id: str | None, config_dict: dict) -> Logger:
        writer = SummaryWriter(self.log_dir)
        writer.add_text(
            "args",
            str(
                dict(
                    log_dir=self.log_dir,
                    logger_type=self.logger_type,
                    wandb_project=self.wandb_project,
                ),
            ),
        )
        if self.logger_type == "wandb":
            logger = WandbLogger(
                save_interval=1,
                name=log_name.replace(os.path.sep, "__"),
                run_id=run_id,
                config=config_dict,
                project=self.wandb_project,
            )
            logger.load(writer)
        elif self.logger_type == "tensorboard":
            logger = TensorboardLogger(writer)
        else:
            raise ValueError(f"Unknown logger type '{self.logger_type}'")
        log_path = os.path.join(self.log_dir, log_name)
        os.makedirs(log_path, exist_ok=True)
        return Logger(logger=logger, log_path=log_path)
