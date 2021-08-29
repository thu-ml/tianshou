from tianshou.utils import BaseLogger
from tianshou.utils.logger.base import LOG_DATA_TYPE

try:
    import wandb
except ImportError:
    pass


class WandBLogger(BaseLogger):
    """Weights and Biases logger that sends data to Weights and Biases.

    Creates three panels with plots: train, test, and update.
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
    """

    def __init__(
        self,
        train_interval: int = 1000,
        test_interval: int = 1,
        update_interval: int = 1000,
    ) -> None:
        super().__init__(train_interval, test_interval, update_interval)

    def write(self, step_type: str, step: int, data: LOG_DATA_TYPE) -> None:
        data[step_type] = step
        wandb.log(data)
