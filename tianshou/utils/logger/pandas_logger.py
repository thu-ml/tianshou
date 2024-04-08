import os
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import pandas as pd

from tianshou.utils import BaseLogger, logging
from tianshou.utils.logger.base import VALID_LOG_VALS, VALID_LOG_VALS_TYPE, DataScope


class PandasLogger(BaseLogger):
    def __init__(
        self,
        log_dir: str,
        train_interval: int = 1000,
        test_interval: int = 1,
        update_interval: int = 1000,
        info_interval: int = 1,
        exclude_arrays: bool = True,
    ) -> None:
        super().__init__(
            train_interval,
            test_interval,
            update_interval,
            info_interval,
            exclude_arrays,
        )
        self.log_path = log_dir
        self.data: dict[str, list] = defaultdict(list)
        self.last_save_step = -1

    def prepare_dict_for_logging(self, data: dict[str, Any]) -> dict[str, VALID_LOG_VALS_TYPE]:
        """Removes invalid data types from the log data."""
        filtered_dict = data.copy()

        def filter_dict(data_dict: dict[str, Any]) -> None:
            """Filter in place."""
            for key, value in data_dict.items():
                if isinstance(value, dict):
                    filter_dict(value)
                else:
                    if not isinstance(value, VALID_LOG_VALS):
                        filtered_dict.pop(key)

        filter_dict(data)
        return filtered_dict

    def write(self, step_type: str, step: int, data: dict[str, Any]) -> None:
        scope, step_name = step_type.split("/")
        data[step_name] = step
        self.data[scope].append(data)

    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn: Callable[[int, int, int], str] | None = None,
    ) -> None:
        self.last_save_step = epoch
        # create and dump a dataframe
        for k, v in self.data.items():
            df = pd.DataFrame(v)
            df.to_csv(os.path.join(self.log_path, k + "_log.csv"), index_label="index")
            df.to_pickle(os.path.join(self.log_path, k + "_log.pkl"))

    def restore_data(self) -> tuple[int, int, int]:
        scopes = [ds.value for ds in DataScope]
        for scope in scopes:
            try:
                self.data[scope].extend(
                    list(
                        pd.read_pickle(os.path.join(self.log_path, scope + "_log.pkl"))
                        .T.to_dict()
                        .values(),
                    ),
                )
            except FileNotFoundError:
                logging.warning(f"Failed to restore {scope} data")

        try:  # epoch / gradient_step
            epoch = self.data["info"][-1]["epoch"]
            self.last_save_step = self.last_log_test_step = epoch
        except (KeyError, IndexError):
            epoch = 0
        try:
            gradient_step = self.data["update"][-1]["gradient_step"]
            self.last_log_update_step = gradient_step
        except (KeyError, IndexError):
            gradient_step = 0
        try:  # offline trainer doesn't have env_step
            env_step = self.data["train"][-1]["env_step"]
            self.last_log_train_step = env_step
        except (KeyError, IndexError):
            env_step = 0

        return epoch, env_step, gradient_step

    def restore_logged_data(
        self,
        log_path: str,
    ) -> dict[str, dict[str, VALID_LOG_VALS_TYPE | dict[str, VALID_LOG_VALS_TYPE]]]:
        data = {}

        def merge_dicts(list_of_dicts: list[dict]) -> dict[str, Any]:
            """Merge a list of dictionaries into a single dictionary.

            On the top level, each list element is a dict corresponding to a logging time point. Each element may
            consist of scalar values, dictionaries, or arrays. The function merges the dictionaries and concatenates the
            lists such that the output dict has the logged keys and values are the corresponding logged values in list.
            """
            result: dict[str, Any] = defaultdict(list)
            for d in list_of_dicts:
                for key, value in d.items():
                    if isinstance(value, dict):
                        result[key] = merge_dicts([result.get(key, {}), value])
                    elif isinstance(value, list):
                        result[key] = result.get(key, []) + value
                    else:
                        result[key].append(value)
            return result

        scopes = [ds.value for ds in DataScope]
        for scope in scopes:
            try:
                dict_list = list(
                    pd.read_pickle(os.path.join(log_path, scope + "_log.pkl")).T.to_dict().values(),
                )
                data[scope] = dict(merge_dicts(dict_list))
            except FileNotFoundError:
                logging.warning(f"Failed to restore {scope} data")
                data[scope] = {}
        return data
