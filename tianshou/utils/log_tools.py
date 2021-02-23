import numpy as np
from numbers import Number
from typing import Any, Union
from abc import ABC, abstractmethod
from torch.utils.tensorboard import SummaryWriter


class BaseLogger(ABC):
    """The base class for any logger which is compatible with trainer."""

    def __init__(self, writer: Any) -> None:
        super().__init__()
        self.writer = writer

    @abstractmethod
    def write(
        self,
        key: str,
        x: Union[Number, np.number, np.ndarray],
        y: Union[Number, np.number, np.ndarray],
        **kwargs: Any,
    ) -> None:
        """Specifies how writer is used to log data.

        :param key: namespace which the input data tuple belongs to.
        :param x: stands for the ordinate of the input data tuple.
        :param y: stands for the abscissa of the input data tuple.
        """
        pass

    def log_train_data(self, collect_result: dict, step: int) -> None:
        """Use writer to log statistics generated during training.

        :param collect_result: a dict containing information of data collected in
            training stage, i.e., returns of collect() method in collector.
        :param int step: TODO
        """
        pass

    def log_update_data(self, update_result: dict, step: int) -> None:
        """Use writer to log statistics generated during updating.

        :param update_result: a dict containing information of data collected in
            updating stage, i.e., returns of update() method in policy.
        :param int step: TODO
        """
        pass

    def log_test_data(self, collect_result: dict, step: int) -> None:
        """Use writer to log statistics generated during evaluating.

        :param collect_result: a dict containing information of data collected in
            evaluating stage, e.g. returns of collect() method in collector.
        :param int step: TODO
        """
        pass


class BasicLogger(BaseLogger):
    """A loggger that relies on tensorboard SummaryWriter by default to visualize \
    and log statistics.

    You can also rewrite write() func to use your own writer.

    :param SummaryWriter writer: TODO
    :param int train_interval: TODO
    :param int test_interval: TODO
    :param int update_interval: TODO
    """

    def __init__(
        self,
        writer: SummaryWriter,
        train_interval: int = 1,
        test_interval: int = 1,
        update_interval: int = 1000,
    ) -> None:
        super().__init__(writer)
        self.train_interval = train_interval
        self.test_interval = test_interval
        self.update_interval = update_interval
        self.last_log_train_step = -1
        self.last_log_test_step = -1
        self.last_log_update_step = -1

    def write(
        self,
        key: str,
        x: Union[Number, np.number, np.ndarray],
        y: Union[Number, np.number, np.ndarray],
        **kwargs: Any,
    ) -> None:
        self.writer.add_scalar(key, y, global_step=x)

    def log_train_data(self, collect_result: dict, step: int) -> None:
        """TODO: mention it may inplace modify collect_result."""
        if collect_result["n/ep"] > 0:
            collect_result["rew"] = collect_result["rews"].mean()
            collect_result["len"] = collect_result["lens"].mean()
            if step - self.last_log_train_step >= self.train_interval:
                self.write("train/n/ep", step, collect_result["n/ep"])
                self.write("train/rew", step, collect_result["rew"])
                self.write("train/len", step, collect_result["len"])
                self.last_log_train_step = step

    def log_test_data(self, collect_result: dict, step: int) -> None:
        """TODO: mention it may inplace modify collect_result."""
        assert collect_result["n/ep"] > 0
        rews, lens = collect_result["rews"], collect_result["lens"]
        rew, rew_std, len_, len_std = rews.mean(), rews.std(), lens.mean(), lens.std()
        collect_result.update(rew=rew, rew_std=rew_std, len=len_, len_std=len_std)
        if step - self.last_log_test_step >= self.test_interval:
            self.write("test/rew", step, rew)
            self.write("test/len", step, len_)
            self.write("test/rew_std", step, rew_std)
            self.write("test/len_std", step, len_std)
            self.last_log_test_step = step

    def log_update_data(self, update_result: dict, step: int) -> None:
        if step - self.last_log_update_step >= self.update_interval:
            for k, v in update_result.items():
                self.write("train/" + k, step, v)  # save in train/
            self.last_log_update_step = step


class LazyLogger(BasicLogger):
    """A loggger that does nothing. Used as the placeholder in trainer."""

    def __init__(self) -> None:
        super().__init__(None)  # type: ignore

    def write(
        self,
        key: str,
        x: Union[Number, np.number, np.ndarray],
        y: Union[Number, np.number, np.ndarray],
        **kwargs: Any,
    ) -> None:
        """The LazyLogger writes nothing."""
        pass
