import warnings
import numpy as np
from numbers import Number
from abc import ABC, abstractmethod
from typing import Any, Tuple, Union
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator


class BaseLogger(ABC):
    """The base class for any logger which is compatible with trainer."""

    def __init__(self, writer: Any) -> None:
        super().__init__()
        self.writer = writer

    @abstractmethod
    def write(
        self, key: str, x: int, y: Union[Number, np.number, np.ndarray], **kwargs: Any
    ) -> None:
        """Specify how the writer is used to log data.

        :param str key: namespace which the input data tuple belongs to.
        :param int x: stands for the ordinate of the input data tuple.
        :param y: stands for the abscissa of the input data tuple.
        """
        pass

    def log_train_data(self, collect_result: dict, step: int) -> None:
        """Use writer to log statistics generated during training.

        :param collect_result: a dict containing information of data collected in
            training stage, i.e., returns of collector.collect().
        :param int step: stands for the timestep the collect_result being logged.
        """
        pass

    def log_update_data(self, update_result: dict, step: int) -> None:
        """Use writer to log statistics generated during updating.

        :param update_result: a dict containing information of data collected in
            updating stage, i.e., returns of policy.update().
        :param int step: stands for the timestep the collect_result being logged.
        """
        pass

    def log_test_data(self, collect_result: dict, step: int, epoch: int) -> None:
        """Use writer to log statistics generated during evaluating.

        :param collect_result: a dict containing information of data collected in
            evaluating stage, i.e., returns of collector.collect().
        :param int step: stands for the timestep the collect_result being logged.
        :param int epoch: stands for the epoch the collect_result being logged.
        """
        pass

    def restore_data(self) -> Tuple[int, float, float, int, int, int, float, int]:
        """Return the metadata from existing log.

        If it finds nothing or an error occurs during the recover process, it will
        return the default parameters.

        :return: best_epoch, best_reward, best_reward_std, epoch, env_step,
            gradient_step, last_rew, last_len
        """
        warnings.warn("Please specify an existing tensorboard logdir to resume.")
        # epoch == -1 is invalid, so that it should be forcely updated by trainer
        return -1, 0.0, 0.0, 0, 0, 0, 0.0, 0


class BasicLogger(BaseLogger):
    """A loggger that relies on tensorboard SummaryWriter by default to visualize \
    and log statistics.

    You can also rewrite write() func to use your own writer.

    :param SummaryWriter writer: the writer to log data.
    :param int train_interval: the log interval in log_train_data(). Default to 1.
    :param int test_interval: the log interval in log_test_data(). Default to 1.
    :param int update_interval: the log interval in log_update_data(). Default to 1000.
    """

    def __init__(
        self,
        writer: SummaryWriter,
        train_interval: int = 1,
        test_interval: int = 1,
        update_interval: int = 1000,
        resume: bool = True,
    ) -> None:
        super().__init__(writer)
        self.train_interval = train_interval
        self.test_interval = test_interval
        self.update_interval = update_interval
        self.last_log_train_step = -1
        self.last_log_test_step = -1
        self.last_log_update_step = -1

    def write(
        self, key: str, x: int, y: Union[Number, np.number, np.ndarray], **kwargs: Any
    ) -> None:
        self.writer.add_scalar(key, y, global_step=x)

    def log_train_data(self, collect_result: dict, step: int) -> None:
        """Use writer to log statistics generated during training.

        :param collect_result: a dict containing information of data collected in
            training stage, i.e., returns of collector.collect().
        :param int step: stands for the timestep the collect_result being logged.

        .. note::

            ``collect_result`` will be modified in-place with "rew" and "len" keys.
        """
        if collect_result["n/ep"] > 0:
            collect_result["rew"] = collect_result["rews"].mean()
            collect_result["len"] = collect_result["lens"].mean()
            if step - self.last_log_train_step >= self.train_interval:
                self.write("train/n/ep", step, collect_result["n/ep"])
                self.write("train/rew", step, collect_result["rew"])
                self.write("train/len", step, collect_result["len"])
                self.last_log_train_step = step

    def log_test_data(self, collect_result: dict, step: int, epoch: int) -> None:
        """Use writer to log statistics generated during evaluating.

        :param collect_result: a dict containing information of data collected in
            evaluating stage, i.e., returns of collector.collect().
        :param int step: stands for the timestep the collect_result being logged.
        :param int epoch: stands for the epoch the collect_result being logged.

        .. note::

            ``collect_result`` will be modified in-place with "rew", "rew_std", "len",
            and "len_std" keys.
        """
        assert collect_result["n/ep"] > 0
        rews, lens = collect_result["rews"], collect_result["lens"]
        rew, rew_std, len_, len_std = rews.mean(), rews.std(), lens.mean(), lens.std()
        collect_result.update(rew=rew, rew_std=rew_std, len=len_, len_std=len_std)
        if step - self.last_log_test_step >= self.test_interval:
            self.write("test/epoch", step, epoch)  # type: ignore
            self.write("test/rew", step, rew)
            self.write("test/len", step, len_)
            self.write("test/rew_std", step, rew_std)
            self.write("test/len_std", step, len_std)
            self.last_log_test_step = step

    def log_update_data(self, update_result: dict, step: int) -> None:
        if step - self.last_log_update_step >= self.update_interval:
            self.write("train/gradient_step", step, step)  # type: ignore
            for k, v in update_result.items():
                self.write(k, step, v)
            self.last_log_update_step = step

    def restore_data(self) -> Tuple[int, float, float, int, int, int, float, int]:
        ea = event_accumulator.EventAccumulator(self.writer.log_dir)
        ea.Reload()
        epoch, best_epoch, best_reward, best_reward_std = 0, -1, 0.0, 0.0
        try:  # best_*
            for test_rew, test_rew_std in zip(
                ea.scalars.Items("test/rew"), ea.scalars.Items("test/rew_std")
            ):
                rew, rew_std = test_rew.value, test_rew_std.value
                if best_epoch == -1 or best_reward < rew:
                    best_epoch, best_reward, best_reward_std = 0, rew, rew_std
                self.last_log_test_step = test_rew.step
            epoch = int(ea.scalars.Items("test/epoch")[-1].value)
        except KeyError:
            pass
        try:  # env_step / last_*
            item = ea.scalars.Items("train/rew")[-1]
            self.last_log_train_step = env_step = item.step
            last_rew = item.value
            last_len = ea.scalars.Items("train/len")[-1].value
        except KeyError:
            last_rew, last_len, env_step = 0.0, 0, 0
        try:
            self.last_log_update_step = gradient_step = int(ea.scalars.Items(
                "train/gradient_step")[-1].value)
        except KeyError:
            gradient_step = 0
        return best_epoch, best_reward, best_reward_std, \
            epoch, env_step, gradient_step, last_rew, last_len


class LazyLogger(BasicLogger):
    """A loggger that does nothing. Used as the placeholder in trainer."""

    def __init__(self) -> None:
        super().__init__(None)  # type: ignore

    def write(
        self, key: str, x: int, y: Union[Number, np.number, np.ndarray], **kwargs: Any
    ) -> None:
        """The LazyLogger writes nothing."""
        pass
