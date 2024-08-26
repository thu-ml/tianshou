from collections.abc import Callable
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter

from tianshou.utils.logger.base import (
    VALID_LOG_VALS,
    VALID_LOG_VALS_TYPE,
    BaseLogger,
    TRestoredData,
)


class TensorboardLogger(BaseLogger):
    """A logger that relies on tensorboard SummaryWriter by default to visualize and log statistics.

    :param SummaryWriter writer: the writer to log data.
    :param train_interval: the log interval in log_train_data(). Default to 1000.
    :param test_interval: the log interval in log_test_data(). Default to 1.
    :param update_interval: the log interval in log_update_data(). Default to 1000.
    :param info_interval: the log interval in log_info_data(). Default to 1.
    :param save_interval: the save interval in save_data(). Default to 1 (save at
        the end of each epoch).
    :param write_flush: whether to flush tensorboard result after each
        add_scalar operation. Default to True.
    """

    def __init__(
        self,
        writer: SummaryWriter,
        train_interval: int = 1000,
        test_interval: int = 1,
        update_interval: int = 1000,
        info_interval: int = 1,
        save_interval: int = 1,
        write_flush: bool = True,
    ) -> None:
        super().__init__(train_interval, test_interval, update_interval, info_interval)
        self.save_interval = save_interval
        self.write_flush = write_flush
        self.last_save_step = -1
        self.writer = writer

    def prepare_dict_for_logging(
        self,
        input_dict: dict[str, Any],
        parent_key: str = "",
        delimiter: str = "/",
        exclude_arrays: bool = True,
    ) -> dict[str, VALID_LOG_VALS_TYPE]:
        """Flattens and filters a nested dictionary by recursively traversing all levels and compressing the keys.

        Filtering is performed with respect to valid logging data types.

        :param input_dict: The nested dictionary to be flattened and filtered.
        :param parent_key: The parent key used as a prefix before the input_dict keys.
        :param delimiter: The delimiter used to separate the keys.
        :param exclude_arrays: Whether to exclude numpy arrays from the output.
        :return: A flattened dictionary where the keys are compressed and values are filtered.
        """
        result = {}

        def add_to_result(
            cur_dict: dict,
            prefix: str = "",
        ) -> None:
            for key, value in cur_dict.items():
                if exclude_arrays and isinstance(value, np.ndarray):
                    continue

                new_key = prefix + delimiter + str(key)
                new_key = new_key.lstrip(delimiter)

                if isinstance(value, dict):
                    add_to_result(
                        value,
                        new_key,
                    )
                elif isinstance(value, VALID_LOG_VALS):
                    result[new_key] = value

        add_to_result(input_dict, prefix=parent_key)
        return result

    def write(self, step_type: str, step: int, data: dict[str, Any]) -> None:
        scope, step_name = step_type.split("/")
        self.writer.add_scalar(step_type, step, global_step=step)
        for k, v in data.items():
            scope_key = f"{scope}/{k}"
            if isinstance(v, np.ndarray):
                self.writer.add_histogram(scope_key, v, global_step=step, bins="auto")
            elif isinstance(v, Figure):
                self.writer.add_figure(scope_key, v, global_step=step)
            else:
                self.writer.add_scalar(scope_key, v, global_step=step)
        if self.write_flush:  # issue 580
            self.writer.flush()  # issue #482

    def finalize(self) -> None:
        self.writer.close()

    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn: Callable[[int, int, int], str] | None = None,
    ) -> None:
        if save_checkpoint_fn and epoch - self.last_save_step >= self.save_interval:
            self.last_save_step = epoch
            save_checkpoint_fn(epoch, env_step, gradient_step)
            self.write("save/epoch", epoch, {"save/epoch": epoch})
            self.write("save/env_step", env_step, {"save/env_step": env_step})
            self.write(
                "save/gradient_step",
                gradient_step,
                {"save/gradient_step": gradient_step},
            )

    def restore_data(self) -> tuple[int, int, int]:
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

    @staticmethod
    def restore_logged_data(
        log_path: str,
    ) -> TRestoredData:
        """Restores the logged data from the tensorboard log directory.

        The result is a nested dictionary where the keys are the tensorboard keys
        and the values are the corresponding numpy arrays. The keys in each level
        form a nested structure, where the hierarchy is represented by the slashes
        in the tensorboard key-strings.
        """
        ea = event_accumulator.EventAccumulator(log_path)
        ea.Reload()

        def add_value_to_innermost_nested_dict(
            data_dict: dict[str, Any],
            key_string: str,
            value: Any,
        ) -> None:
            """A particular logic, walking through the keys in the
            `key_string` and adding the value to the `data_dict` in a nested manner,
            creating nested dictionaries on the fly if necessary, or updating existing ones.
            The value is added only to the innermost-nested dictionary.


            Example:
            -------
            >>> data_dict = {}
            >>> add_value_to_innermost_nested_dict(data_dict, "a/b/c", 1)
            >>> data_dict
            {"a": {"b": {"c": 1}}}
            """
            keys = key_string.split("/")

            cur_nested_dict = data_dict
            # walk through the intermediate keys to reach the innermost-nested dict,
            # creating nested dictionaries on the fly if necessary
            for k in keys[:-1]:
                cur_nested_dict = cur_nested_dict.setdefault(k, {})
            # After the loop above,
            # this is the innermost-nested dict, where the value is finally set
            # for the last key in the key_string
            cur_nested_dict[keys[-1]] = value

        restored_data: dict[str, np.ndarray | dict] = {}
        for key_string in ea.scalars.Keys():
            add_value_to_innermost_nested_dict(
                restored_data,
                key_string,
                np.array([s.value for s in ea.scalars.Items(key_string)]),
            )

        return restored_data
