import os
from abc import ABC, abstractmethod
from typing import Literal, TypeAlias

from torch.utils.tensorboard import SummaryWriter

from tianshou.utils import BaseLogger, TensorboardLogger, WandbLogger
from tianshou.utils.logger.base import LoggerManager
from tianshou.utils.logger.pandas_logger import PandasLogger
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
        logger_type: Literal["tensorboard", "wandb", "pandas"] = "tensorboard",
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
        if self.logger_type in ["wandb", "tensorboard"]:
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
            case "pandas":
                return PandasLogger(log_dir, exclude_arrays=False)
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


class LoggerManagerFactory(LoggerFactory):
    def __init__(
            self,
            logger_types: list[Literal["tensorboard", "wandb", "pandas"]] = ["tensorboard", "pandas"],
            wandb_project: str | None = None,
    ):
        self.logger_types = logger_types
        self.wandb_project = wandb_project

        self.factories = {
            "pandas": LoggerFactoryDefault(logger_type="pandas"),
            "wandb": LoggerFactoryDefault(logger_type="wandb", wandb_project=wandb_project),
            "tensorboard": LoggerFactoryDefault(logger_type="tensorboard"),
        }

    def create_logger(
        self,
        log_dir: str,
        experiment_name: str,
        run_id: str | None,
        config_dict: dict,
    ) -> TLogger:
        logger_manager = LoggerManager()
        for logger_type in self.logger_types:
            logger_manager.loggers.append(
                self.factories[logger_type].create_logger(
                    log_dir,
                    experiment_name,
                    run_id,
                    config_dict,
                )
            )
        return logger_manager
