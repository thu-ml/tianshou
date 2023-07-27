import argparse
import datetime
import os
from typing import Literal, Optional, Tuple, Union

from torch.utils.tensorboard import SummaryWriter

from tianshou.config import LoggerConfig
from .tensorboard import TensorboardLogger
from .wandb import WandbLogger


def get_logger_for_run(
    algo_name: str,
    task: str,
    logger_config: LoggerConfig,
    config: dict,
    seed: int,
    resume_id: Optional[Union[str, int]],
) -> Tuple[str, Union[WandbLogger, TensorboardLogger]]:
    """

    :param algo_name:
    :param task:
    :param logger_config:
    :param config: the experiment config
    :param seed:
    :param resume_id: used as run_id by wandb, unused for tensorboard
    :return:
    """
    """Returns the log_path and logger for an experiment."""
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    log_name = os.path.join(task, algo_name, str(seed), now)
    log_path = os.path.join(logger_config.logdir, log_name)

    logger = get_logger(
        logger_config.logger,
        log_path,
        log_name=log_name,
        run_id=resume_id,
        config=config,
        wandb_project=logger_config.wandb_project,
    )
    return log_path, logger


def get_logger(
    kind: Literal["wandb", "tensorboard"],
    log_path: str,
    log_name="",
    run_id: Optional[Union[str, int]] = None,
    config: Optional[Union[dict, argparse.Namespace]] = None,
    wandb_project: Optional[str] = None,
):
    """

    :param kind:
    :param log_path:
    :param log_name: currently only used for wandb
    :param run_id: typically equals the restore_id in an experiment, only used
        for wandb
    :param config: config of the experiment, added as text to the log
    :param wandb_project:
    :return:
    """
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(config))
    if kind == "wandb":
        assert wandb_project is not None
        assert log_name
        logger = WandbLogger(
            save_interval=1,
            name=log_name.replace(os.path.sep, "__"),
            run_id=run_id,
            config=config,
            project=wandb_project,
        )
        logger.load(writer)
    elif kind == "tensorboard":
        logger = TensorboardLogger(writer)
    else:
        raise ValueError(f"Unknown logger: {kind}")
    return logger
