import argparse
import contextlib
import os
from collections.abc import Callable

from torch.utils.tensorboard import SummaryWriter

from tianshou.utils import BaseLogger, TensorboardLogger
from tianshou.utils.logger.base import VALID_LOG_VALS_TYPE

with contextlib.suppress(ImportError):
    import wandb


class WandbLogger(BaseLogger):
    """Weights and Biases logger that sends data to https://wandb.ai/.

    This logger creates three panels with plots: train, test, and update.
    Make sure to select the correct access for each panel in weights and biases:

    Example of usage:
    ::

        logger = WandbLogger()
        logger.load(SummaryWriter(log_path))
        result = OnpolicyTrainer(policy, train_collector, test_collector,
                                  logger=logger).run()

    :param train_interval: the log interval in log_train_data(). Default to 1000.
    :param test_interval: the log interval in log_test_data(). Default to 1.
    :param update_interval: the log interval in log_update_data().
        Default to 1000.
    :param info_interval: the log interval in log_info_data(). Default to 1.
    :param save_interval: the save interval in save_data(). Default to 1 (save at
        the end of each epoch).
    :param write_flush: whether to flush tensorboard result after each
        add_scalar operation. Default to True.
    :param str project: W&B project name. Default to "tianshou".
    :param str name: W&B run name. Default to None. If None, random name is assigned.
    :param str entity: W&B team/organization name. Default to None.
    :param str run_id: run id of W&B run to be resumed. Default to None.
    :param argparse.Namespace config: experiment configurations. Default to None.
    """

    def __init__(
        self,
        train_interval: int = 1000,
        test_interval: int = 1,
        update_interval: int = 1000,
        info_interval: int = 1,
        save_interval: int = 1000,
        write_flush: bool = True,
        project: str | None = None,
        name: str | None = None,
        entity: str | None = None,
        run_id: str | None = None,
        config: argparse.Namespace | dict | None = None,
        monitor_gym: bool = True,
    ) -> None:
        super().__init__(train_interval, test_interval, update_interval, info_interval)
        self.last_save_step = -1
        self.save_interval = save_interval
        self.write_flush = write_flush
        self.restored = False
        if project is None:
            project = os.getenv("WANDB_PROJECT", "tianshou")

        self.wandb_run = (
            wandb.init(
                project=project,
                name=name,
                id=run_id,
                resume="allow",
                entity=entity,
                sync_tensorboard=True,
                monitor_gym=monitor_gym,
                config=config,  # type: ignore
            )
            if not wandb.run
            else wandb.run
        )
        self.wandb_run._label(repo="tianshou")  # type: ignore
        self.tensorboard_logger: TensorboardLogger | None = None

    def load(self, writer: SummaryWriter) -> None:
        self.writer = writer
        self.tensorboard_logger = TensorboardLogger(
            writer,
            self.train_interval,
            self.test_interval,
            self.update_interval,
            self.save_interval,
            self.write_flush,
        )

    def write(self, step_type: str, step: int, data: dict[str, VALID_LOG_VALS_TYPE]) -> None:
        if self.tensorboard_logger is None:
            raise Exception(
                "`logger` needs to load the Tensorboard Writer before "
                "writing data. Try `logger.load(SummaryWriter(log_path))`",
            )
        self.tensorboard_logger.write(step_type, step, data)

    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn: Callable[[int, int, int], str] | None = None,
    ) -> None:
        """Use writer to log metadata when calling ``save_checkpoint_fn`` in trainer.

        :param epoch: the epoch in trainer.
        :param env_step: the env_step in trainer.
        :param gradient_step: the gradient_step in trainer.
        :param function save_checkpoint_fn: a hook defined by user, see trainer
            documentation for detail.
        """
        if save_checkpoint_fn and epoch - self.last_save_step >= self.save_interval:
            self.last_save_step = epoch
            checkpoint_path = save_checkpoint_fn(epoch, env_step, gradient_step)

            checkpoint_artifact = wandb.Artifact(
                "run_" + self.wandb_run.id + "_checkpoint",  # type: ignore
                type="model",
                metadata={
                    "save/epoch": epoch,
                    "save/env_step": env_step,
                    "save/gradient_step": gradient_step,
                    "checkpoint_path": str(checkpoint_path),
                },
            )
            checkpoint_artifact.add_file(str(checkpoint_path))
            self.wandb_run.log_artifact(checkpoint_artifact)  # type: ignore

    def restore_data(self) -> tuple[int, int, int]:
        checkpoint_artifact = self.wandb_run.use_artifact(  # type: ignore
            f"run_{self.wandb_run.id}_checkpoint:latest",  # type: ignore
        )
        assert checkpoint_artifact is not None, "W&B dataset artifact doesn't exist"

        checkpoint_artifact.download(
            os.path.dirname(checkpoint_artifact.metadata["checkpoint_path"]),
        )

        try:  # epoch / gradient_step
            epoch = checkpoint_artifact.metadata["save/epoch"]
            self.last_save_step = self.last_log_test_step = epoch
            gradient_step = checkpoint_artifact.metadata["save/gradient_step"]
            self.last_log_update_step = gradient_step
        except KeyError:
            epoch, gradient_step = 0, 0
        try:  # offline trainer doesn't have env_step
            env_step = checkpoint_artifact.metadata["save/env_step"]
            self.last_log_train_step = env_step
        except KeyError:
            env_step = 0
        return epoch, env_step, gradient_step
