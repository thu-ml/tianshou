import argparse
import os
from typing import Optional

from tianshou.utils import BaseLogger
from tianshou.utils.logger.base import LOG_DATA_TYPE

try:
    import wandb
except ImportError:
    pass


def wandb_init(
    args: argparse.Namespace, run_name: str, resume_id: int
) -> wandb.sdk.wandb_run.Run:
    wandb_run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "tianshou"),
        name=run_name,
        resume_id=resume_id,
        sync_tensorboard=True,
        monitor_gym=True,
        config=args,  # type: ignore
    ) if not wandb.run else wandb.run
    wandb_run._label(repo="tianshou")  # type: ignore

    return wandb_run


class WandbLogger(BaseLogger):
    """Weights and Biases logger that sends data to https://wandb.ai/.

    This logger creates three panels with plots: train, test, and update.
    Make sure to select the correct access for each panel in weights and biases:

    - ``train/env_step`` for train plots
    - ``test/env_step`` for test plots
    - ``update/gradient_step`` for update plots

    Example of usage:
    ::

        with wandb.init(project="My Project"):
            logger = WandBLogger()
            result = onpolicy_trainer(policy, train_collector, test_collector,
                    logger=logger)

    :param int train_interval: the log interval in log_train_data(). Default to 1000.
    :param int test_interval: the log interval in log_test_data(). Default to 1.
    :param int update_interval: the log interval in log_update_data().
        Default to 1000.
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
        save_interval: int = 1000,
        project: str = 'tianshou',
        name: Optional[str] = None,
        entity: Optional[str] = None,
        run_id: Optional[str] = None,
        config: Optional[argparse.Namespace] = None,
    ) -> None:
        raise Exception(
            """`WandbLogger` is deprecated, please use the following code instead:

from tianshou.utils.logger.wandb import wandb_init
run_name = None
wandb_run = wandb_init(args, run_name, args.resume_id)
writer = SummaryWriter(log_path)
writer.add_text("args", str(args))
logger = TensorboardLogger(writer, wandb_run)
        """
        )

    def write(self, step_type: str, step: int, data: LOG_DATA_TYPE) -> None:
        pass
