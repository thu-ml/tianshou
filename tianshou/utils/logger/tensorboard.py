import os
import warnings
from typing import Any, Callable, Optional, Tuple

import wandb
from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter
from wandb.sdk.wandb_run import Run

from tianshou.utils.logger.base import LOG_DATA_TYPE, BaseLogger


class TensorboardLogger(BaseLogger):
    """A logger that relies on tensorboard SummaryWriter by default to visualize \
    and log statistics.

    :param SummaryWriter writer: the writer to log data.
    :param int train_interval: the log interval in log_train_data(). Default to 1000.
    :param int test_interval: the log interval in log_test_data(). Default to 1.
    :param int update_interval: the log interval in log_update_data(). Default to 1000.
    :param int save_interval: the save interval in save_data(). Default to 1 (save at
        the end of each epoch).
    :param Run wandb_run: the Weights & Biases run to help save and load models. Default to `None`.
    """

    def __init__(
        self,
        writer: SummaryWriter,
        train_interval: int = 1000,
        test_interval: int = 1,
        update_interval: int = 1000,
        save_interval: int = 1,
        wandb_run: Optional[Run] = None,
    ) -> None:
        super().__init__(train_interval, test_interval, update_interval)
        self.save_interval = save_interval
        self.last_save_step = -1
        self.writer = writer
        self.wandb_run = wandb_run

    def write(self, step_type: str, step: int, data: LOG_DATA_TYPE) -> None:
        for k, v in data.items():
            self.writer.add_scalar(k, v, global_step=step)
        self.writer.flush()  # issue #482

    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn: Optional[Callable[[int, int, int], None]] = None,
    ) -> None:
        if save_checkpoint_fn and epoch - self.last_save_step >= self.save_interval:
            self.last_save_step = epoch
            checkpoint_path = save_checkpoint_fn(epoch, env_step, gradient_step)
            self.write("save/epoch", epoch, {"save/epoch": epoch})
            self.write("save/env_step", env_step, {"save/env_step": env_step})
            self.write(
                "save/gradient_step", gradient_step,
                {"save/gradient_step": gradient_step}
            )

            if self.wandb_run:
                checkpoint_artifact = wandb.Artifact(
                    'run_' + self.wandb_run.id + '_checkpoint',  # type: ignore
                    type='model',
                    metadata={
                        "save/epoch": epoch,
                        "save/env_step": env_step,
                        "save/gradient_step": gradient_step,
                        "checkpoint_path": str(checkpoint_path)
                    }
                )
                checkpoint_artifact.add_file(str(checkpoint_path))
                self.wandb_run.log_artifact(checkpoint_artifact)  # type: ignore

    def restore_data(self) -> Tuple[int, int, int]:
        if self.wandb_run:
            checkpoint_artifact = self.wandb_run.use_artifact(    # type: ignore
                'run_' + self.wandb_run.id + '_checkpoint:latest'  # type: ignore
            )
            assert checkpoint_artifact is not None, "W&B dataset artifact doesn't exist"

            checkpoint_artifact.download(
                os.path.dirname(checkpoint_artifact.metadata['checkpoint_path'])
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

        else:
            ea = event_accumulator.EventAccumulator(self.writer.log_dir)
            ea.Reload()

            try:  # epoch / gradient_step
                epoch = ea.scalars.Items("save/epoch")[-1].step
                self.last_save_step = self.last_log_test_step = epoch
                gradient_step = ea.scalars.Items("save/gradient_step")[-1].step
                self.last_log_update_step = gradient_step
            except KeyError:
                epoch, gradient_step = 0, 0
            try:  # offline trainer doesn't have env_step
                env_step = ea.scalars.Items("save/env_step")[-1].step
                self.last_log_train_step = env_step
            except KeyError:
                env_step = 0

        return epoch, env_step, gradient_step


class BasicLogger(TensorboardLogger):
    """BasicLogger has changed its name to TensorboardLogger in #427.

    This class is for compatibility.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "Deprecated soon: BasicLogger has renamed to TensorboardLogger in #427."
        )
        super().__init__(*args, **kwargs)
