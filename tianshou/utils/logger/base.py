import typing
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict
from numbers import Number

import numpy as np

from tianshou.data import BaseStats, CollectStats, InfoStats, UpdateStats

LOG_DATA_TYPE = dict[str, int | Number | np.number | np.ndarray]


class BaseLogger(ABC):
    """The base class for any logger which is compatible with trainer.

    Try to overwrite write() method to use your own writer.

    :param train_interval: the log interval in log_train_data(). Default to 1000.
    :param test_interval: the log interval in log_test_data(). Default to 1.
    :param update_interval: the log interval in log_update_data(). Default to 1000.
    :param info_interval: the log interval in log_info_data(). Default to 1.
    """

    def __init__(
        self,
        train_interval: int = 1000,
        test_interval: int = 1,
        update_interval: int = 1000,
        info_interval: int = 1,
    ) -> None:
        super().__init__()
        self.train_interval = train_interval
        self.test_interval = test_interval
        self.update_interval = update_interval
        self.info_interval = info_interval
        self.last_log_train_step = -1
        self.last_log_test_step = -1
        self.last_log_update_step = -1
        self.last_log_info_step = -1

    @abstractmethod
    def write(self, step_type: str, step: int, data: LOG_DATA_TYPE) -> None:
        """Specify how the writer is used to log data.

        :param str step_type: namespace which the data dict belongs to.
        :param step: stands for the ordinate of the data dict.
        :param data: the data to write with format ``{key: value}``.
        """

    @staticmethod
    # adapted from https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
    def _flatten_dict(
            dictionary: dict,
            parent_key="",
            delimiter="/",
            exclude_arrays: bool = True,
    ) -> LOG_DATA_TYPE:
        """
        Flattens a nested dictionary by recursively traversing all levels and compressing the keys.

        :param dictionary: The nested dictionary to be flattened.
        :param parent_key: The parent key of the current level of the dictionary (default is an empty string).
        :param delimiter: The delimiter used to join the keys of the flattened dictionary (default is "/").
        :param exclude_arrays: Specifies whether to exclude arrays when flattening (default is True).
        :return: A flattened dictionary where the keys are compressed.
        """
        items = []
        for key, value in dictionary.items():
            new_key = parent_key + delimiter + key if parent_key else key
            if isinstance(value, dict):
                items.extend(
                    BaseLogger._flatten_dict(
                        value,
                        new_key,
                        delimiter=delimiter,
                        exclude_arrays=exclude_arrays,
                    ).items(),
                )
            else:
                if isinstance(value, typing.get_args(typing.get_args(LOG_DATA_TYPE)[1])):
                    items.append((new_key, value))
        return dict(items)

    def prepare_dataclass_for_logging(self, stats: BaseStats, data_scope: str = "", exclude: set[str] | None = None
                                      ) -> LOG_DATA_TYPE:
        """Convert the dataclass to a dictionary with flat hierarchy.

        :param stats: the dataclass instance to be converted.
        :param data_scope: the scope of the data, e.g., train, test, update, info.
        :param exclude: set of field names to exclude from the dictionary.
        :return: a dictionary of the dataclass object with flat hierarchy
        """
        stat_dict = asdict(stats)
        flattened_stat_dict = self._flatten_dict(
            stat_dict,
            data_scope,
            exclude_arrays=True,  # TODO: make this configurable if we want to use histograms or log images
        )
        if exclude is not None:
            flattened_stat_dict = {k: v for k, v in flattened_stat_dict.items() if k not in exclude}

        return flattened_stat_dict

    def log_train_data(self, collect_result: CollectStats, step: int) -> None:
        """Use writer to log statistics generated during training.

        :param collect_result: a dataclass object containing information of data collected in
            training stage, i.e., returns of collector.collect().
        :param step: stands for the timestep the collect_result being logged.
        """
        if (
            collect_result.n_collected_episodes > 0
            and step - self.last_log_train_step >= self.train_interval
        ):
            log_data = self.prepare_dataclass_for_logging(collect_result, data_scope="train")
            self.write("train/env_step", step, log_data)
            self.last_log_train_step = step

    def log_test_data(self, collect_result: CollectStats, step: int) -> None:
        """Use writer to log statistics generated during evaluating.

        :param collect_result: a dataclass object containing information of data collected in
            evaluating stage, i.e., returns of collector.collect().
        :param step: stands for the timestep the collect_result being logged.
        """
        assert collect_result.n_collected_episodes > 0
        if step - self.last_log_test_step >= self.test_interval:
            log_data = self.prepare_dataclass_for_logging(collect_result, data_scope="test")
            self.write("test/env_step", step, log_data)
            self.last_log_test_step = step

    def log_update_data(self, update_result: UpdateStats, step: int) -> None:
        """Use writer to log statistics generated during updating.

        :param update_result: a dataclass object containing information of data collected in
            updating stage, i.e., returns of policy.update().
        :param step: stands for the timestep the collect_result being logged.
        """
        if step - self.last_log_update_step >= self.update_interval:
            log_data = self.prepare_dataclass_for_logging(update_result, data_scope="update")
            self.write("update/gradient_step", step, log_data)
            self.last_log_update_step = step

    def log_info_data(self, info: InfoStats, step: int) -> None:
        """Use writer to log global statistics.

        :param info: a dataclass object containing information of data collected at the end of an epoch.
        :param step: stands for the timestep the collect_result being logged.
        """
        if step - self.last_log_info_step >= self.info_interval:
            log_data = self.prepare_dataclass_for_logging(info, data_scope="info")
            self.write("info/epoch", step, log_data)
            self.last_log_info_step = step

    @abstractmethod
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

    @abstractmethod
    def restore_data(self) -> tuple[int, int, int]:
        """Return the metadata from existing log.

        If it finds nothing or an error occurs during the recover process, it will
        return the default parameters.

        :return: epoch, env_step, gradient_step.
        """


class LazyLogger(BaseLogger):
    """A logger that does nothing. Used as the placeholder in trainer."""

    def __init__(self) -> None:
        super().__init__()

    def write(self, step_type: str, step: int, data: LOG_DATA_TYPE) -> None:
        """The LazyLogger writes nothing."""

    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn: Callable[[int, int, int], str] | None = None,
    ) -> None:
        pass

    def restore_data(self) -> tuple[int, int, int]:
        return 0, 0, 0
