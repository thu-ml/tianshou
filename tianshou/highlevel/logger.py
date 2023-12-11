import os
from abc import ABC, abstractmethod
from typing import Literal, TypeAlias

from torch.utils.tensorboard import SummaryWriter

from tianshou.utils import BaseLogger, TensorboardLogger, WandbLogger
from tianshou.utils.string import ToStringMixin

TLogger: TypeAlias = BaseLogger


class LoggerFactory(ToStringMixin, ABC):
    @abstractmethod
    def create_logger(
        self,
        log_dir: str,
        experiment_name: str,
        run_id: str | None,
        config_dict: dict,
    ) -> TLogger:
        """Creates the logger.

        :param log_dir: path to the directory in which log data is to be stored
        :param experiment_name: the name of the job, which may contain `os.path.sep`
        :param run_id: a unique name, which, depending on the logging framework, may be used to identify the logger
        :param config_dict: a dictionary with data that is to be logged
        :return: the logger
        """


class LoggerFactoryDefault(LoggerFactory):
    def __init__(
        self,
        logger_type: Literal["tensorboard", "wandb"] = "tensorboard",
        wandb_project: str | None = None,
    ):
        if logger_type == "wandb" and wandb_project is None:
            raise ValueError("Must provide 'wandb_project'")
        self.logger_type = logger_type
        self.wandb_project = wandb_project

    def create_logger(
        self,
        log_dir: str,
        experiment_name: str,
        run_id: str | None,
        config_dict: dict,
    ) -> TLogger:
        writer = SummaryWriter(log_dir)
        writer.add_text(
            "args",
            str(
                dict(
                    log_dir=log_dir,
                    logger_type=self.logger_type,
                    wandb_project=self.wandb_project,
                ),
            ),
        )
        match self.logger_type:
            case "wandb":
                wandb_logger = WandbLogger(
                    save_interval=1,
                    name=experiment_name.replace(os.path.sep, "__"),
                    run_id=run_id,
                    config=config_dict,
                    project=self.wandb_project,
                )
                wandb_logger.load(writer)
                return wandb_logger
            case "tensorboard":
                return TensorboardLogger(writer)
            case _:
                raise ValueError(f"Unknown logger type '{self.logger_type}'")
