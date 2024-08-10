import os
from abc import ABC, abstractmethod
from typing import Literal, TypeAlias

from sensai.util.string import ToStringMixin
from torch.utils.tensorboard import SummaryWriter

from tianshou.utils import BaseLogger, TensorboardLogger, WandbLogger

TLogger: TypeAlias = BaseLogger


class LoggerFactory(ToStringMixin, ABC):
    @abstractmethod
    def create_logger(
        self,
        log_dir: str,
        experiment_name: str,
        run_id: str | None,
        config_dict: dict | None = None,
    ) -> TLogger:
        """Creates the logger.

        :param log_dir: path to the directory in which log data is to be stored
        :param experiment_name: the name of the job, which may contain `os.path.delimiter`
        :param run_id: a unique name, which, depending on the logging framework, may be used to identify the logger
        :param config_dict: a dictionary with data that is to be logged
        :return: the logger
        """

    @abstractmethod
    def get_logger_class(self) -> type[TLogger]:
        """Returns the class of the logger that is to be created."""


class LoggerFactoryDefault(LoggerFactory):
    def __init__(
        self,
        logger_type: Literal["tensorboard", "wandb", "pandas"] = "tensorboard",
        wand_entity: str | None = None,
        wandb_project: str | None = None,
        group: str | None = None,
        job_type: str | None = None,
        save_interval: int = 1,
    ):
        if logger_type == "wandb" and wandb_project is None:
            raise ValueError("Must provide 'wandb_project'")
        self.logger_type = logger_type
        self.wandb_entity = wand_entity
        self.wandb_project = wandb_project
        self.group = group
        self.job_type = job_type
        self.save_interval = save_interval

    def create_logger(
        self,
        log_dir: str,
        experiment_name: str,
        run_id: str | None,
        config_dict: dict | None = None,
    ) -> TLogger:
        match self.logger_type:
            case "wandb":
                logger = WandbLogger(
                    save_interval=self.save_interval,
                    name=experiment_name.replace(os.path.sep, "__"),
                    run_id=run_id,
                    config=config_dict,
                    entity=self.wandb_entity,
                    project=self.wandb_project,
                    group=self.group,
                    job_type=self.job_type,
                    log_dir=log_dir,
                )
                writer = self._create_writer(log_dir)  # writer has to be created after wandb.init!
                logger.load(writer)
                return logger
            case "tensorboard":
                writer = self._create_writer(log_dir)
                return TensorboardLogger(writer)
            case _:
                raise ValueError(f"Unknown logger type '{self.logger_type}'")

    def _create_writer(self, log_dir: str) -> SummaryWriter:
        """Creates a tensorboard writer and adds a text artifact."""
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
        return writer

    def get_logger_class(self) -> type[TLogger]:
        match self.logger_type:
            case "wandb":
                return WandbLogger
            case "tensorboard":
                return TensorboardLogger
            case _:
                raise ValueError(f"Unknown logger type '{self.logger_type}'")
