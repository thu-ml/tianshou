from tianshou.utils import BaseLogger
import wandb


class WandBLogger(BaseLogger):
    """Weights and Biases logger that sends data to https://www.wandb.com/
    Creates three panels with plots: train, test and update.
    Make sure to select the correct access for each panel in weights and biases:

    - `train/env_step` for train plots
    - `test/env_ste`   for test plots
    - `update/gradient_step` for update plots

    Example of usage:

        with wandb.init(project="My Project"):
            logger = WandBLogger()

            result = onpolicy_trainer(policy, train_collector, test_collector,
                    logger=logger)

    :param int train_interval: the log interval in log_train_data(). Default to 1000.
    :param int test_interval: the log interval in log_test_data(). Default to 1.
    :param int update_interval: the log interval in log_update_data().
        Default to 1000."""
    def __init__(
        self,
        train_interval: int = 1000,
        test_interval: int = 1,
        update_interval: int = 1000
    ) -> None:
        super().__init__(writer=None)

        self.train_interval = train_interval
        self.test_interval = test_interval
        self.update_interval = update_interval
        self.last_log_train_step = -1
        self.last_log_test_step = -1
        self.last_log_update_step = -1

    def write(self, key, x, y, **kwargs):
        pass

    def log_train_data(self, collect_result: dict, step: int) -> None:
        if collect_result["n/ep"] > 0:
            collect_result["rew"] = collect_result["rews"].mean()
            collect_result["len"] = collect_result["lens"].mean()
            if step - self.last_log_train_step >= self.train_interval:

                log_data = {
                    "train/env_step": step,
                    "train/episode": collect_result["n/ep"],
                    "train/reward": collect_result["rew"],
                    "train/length": collect_result["len"]}
                wandb.log(log_data)

                self.last_log_train_step = step

    def log_test_data(self, collect_result: dict, step: int) -> None:
        assert collect_result["n/ep"] > 0
        rews, lens = collect_result["rews"], collect_result["lens"]
        rew, rew_std, len_, len_std = rews.mean(), rews.std(), lens.mean(), lens.std()
        collect_result.update(rew=rew, rew_std=rew_std, len=len_, len_std=len_std)
        if step - self.last_log_test_step >= self.test_interval:

            log_data = {
                "test/env_step": step,
                "test/reward": rew,
                "test/length": len_,
                "test/reward_std": rew_std,
                "test/length_std": len_std}

            wandb.log(log_data)
            self.last_log_test_step = step

    def log_update_data(self, update_result: dict, step: int) -> None:
        if step - self.last_log_update_step >= self.update_interval:
            log_data = {}

            for k, v in update_result.items():
                log_data[f'update/{k}'] = v

            log_data['update/gradient_step'] = step
            wandb.log(log_data)

            self.last_log_update_step = step
