import numpy as np
from numbers import Number
from abc import ABC, abstractmethod
from typing import Any, Tuple, Union, Callable, Optional


WRITE_TYPE = Union[int, Number, np.number, np.ndarray]


class BaseLogger(ABC):
    """The base class for any logger which is compatible with trainer."""

    def __init__(self, writer: Any) -> None:
        super().__init__()
        self.writer = writer

    @abstractmethod
    def write(self, key: str, x: int, y: WRITE_TYPE, **kwargs: Any) -> None:
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

    def log_test_data(self, collect_result: dict, step: int) -> None:
        """Use writer to log statistics generated during evaluating.

        :param collect_result: a dict containing information of data collected in
            evaluating stage, i.e., returns of collector.collect().
        :param int step: stands for the timestep the collect_result being logged.
        """
        pass

    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn: Optional[Callable[[int, int, int], None]] = None,
    ) -> None:
        """Use writer to log metadata when calling ``save_checkpoint_fn`` in trainer.

        :param int epoch: the epoch in trainer.
        :param int env_step: the env_step in trainer.
        :param int gradient_step: the gradient_step in trainer.
        :param function save_checkpoint_fn: a hook defined by user, see trainer
            documentation for detail.
        """
        pass

    def restore_data(self) -> Tuple[int, int, int]:
        """Return the metadata from existing log.

        If it finds nothing or an error occurs during the recover process, it will
        return the default parameters.

        :return: epoch, env_step, gradient_step.
        """
        pass


class LazyLogger(BaseLogger):
    """A logger that does nothing. Used as the placeholder in trainer."""

    def __init__(self) -> None:
        super().__init__(None)  # type: ignore

    def write(self, key: str, x: int, y: WRITE_TYPE, **kwargs: Any) -> None:
        """The LazyLogger writes nothing."""
        pass
