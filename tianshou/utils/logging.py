"""
Partial copy of sensai.util.logging
"""
# ruff: noqa
import atexit
import logging as lg
import sys
from collections.abc import Callable
from datetime import datetime
from io import StringIO
from logging import *
from typing import Any, TypeVar, cast

log = getLogger(__name__)  # type: ignore

LOG_DEFAULT_FORMAT = "%(levelname)-5s %(asctime)-15s %(name)s:%(funcName)s - %(message)s"

# Holds the log format that is configured by the user (using function `configure`), such
# that it can be reused in other places
_logFormat = LOG_DEFAULT_FORMAT


def set_numerical_fields_to_precision(data: dict[str, Any], precision: int = 3) -> dict[str, Any]:
    """Returns a copy of the given dictionary with all numerical values rounded to the given precision.

    Note: does not recurse into nested dictionaries.

    :param data: a dictionary
    :param precision: the precision to be used
    """
    result = {}
    for k, v in data.items():
        if isinstance(v, float):
            v = round(v, precision)
        result[k] = v
    return result


def remove_log_handlers() -> None:
    """Removes all current log handlers."""
    logger = getLogger()
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])


def remove_log_handler(handler: Handler) -> None:
    getLogger().removeHandler(handler)


def is_log_handler_active(handler: Handler) -> bool:
    """Checks whether the given handler is active.

    :param handler: a log handler
    :return: True if the handler is active, False otherwise
    """
    return handler in getLogger().handlers


# noinspection PyShadowingBuiltins
def configure(format: str = LOG_DEFAULT_FORMAT, level: int = lg.DEBUG) -> None:
    """Configures logging to stdout with the given format and log level,
    also configuring the default log levels of some overly verbose libraries as well as some pandas output options.

    :param format: the log format
    :param level: the minimum log level
    """
    global _logFormat
    _logFormat = format
    remove_log_handlers()
    basicConfig(level=level, format=format, stream=sys.stdout)
    # set log levels of third-party libraries
    getLogger("numba").setLevel(INFO)


T = TypeVar("T")


# noinspection PyShadowingBuiltins
def run_main(
    main_fn: Callable[[], T], format: str = LOG_DEFAULT_FORMAT, level: int = lg.DEBUG
) -> T | None:
    """Configures logging with the given parameters, ensuring that any exceptions that occur during
    the execution of the given function are logged.
    Logs two additional messages, one before the execution of the function, and one upon its completion.

    :param main_fn: the function to be executed
    :param format: the log message format
    :param level: the minimum log level
    :return: the result of `main_fn`
    """
    configure(format=format, level=level)
    log.info("Starting")  # type: ignore
    try:
        result = main_fn()
        log.info("Done")  # type: ignore
        return result
    except Exception as e:
        log.error("Exception during script execution", exc_info=e)  # type: ignore
        return None


def run_cli(
    main_fn: Callable[..., T], format: str = LOG_DEFAULT_FORMAT, level: int = lg.DEBUG
) -> T | None:
    """
    Configures logging with the given parameters and runs the given main function as a
    CLI using `jsonargparse` (which is configured to also parse attribute docstrings, such
    that dataclasses can be used as function arguments).
    Using this function requires that `jsonargparse` and `docstring_parser` be available.
    Like `run_main`, two additional log messages will be logged (at the beginning and end
    of the execution), and it is ensured that all exceptions will be logged.

    :param main_fn: the function to be executed
    :param format: the log message format
    :param level: the minimum log level
    :return: the result of `main_fn`
    """
    from jsonargparse import set_docstring_parse_options, CLI

    set_docstring_parse_options(attribute_docstrings=True)
    return run_main(lambda: CLI(main_fn), format=format, level=level)


def datetime_tag() -> str:
    """:return: a string tag for use in log file names which contains the current date and time (compact but readable)"""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


_fileLoggerPaths: list[str] = []
_isAtExitReportFileLoggerRegistered = False
_memoryLogStream: StringIO | None = None


def _at_exit_report_file_logger() -> None:
    for path in _fileLoggerPaths:
        print(f"A log file was saved to {path}")


def add_file_logger(path: str, register_atexit: bool = True) -> FileHandler:
    global _isAtExitReportFileLoggerRegistered
    log.info(f"Logging to {path} ...")  # type: ignore
    handler = FileHandler(path)
    handler.setFormatter(Formatter(_logFormat))
    Logger.root.addHandler(handler)
    _fileLoggerPaths.append(path)
    if not _isAtExitReportFileLoggerRegistered and register_atexit:
        atexit.register(_at_exit_report_file_logger)
        _isAtExitReportFileLoggerRegistered = True
    return handler


def add_memory_logger() -> None:
    """Enables in-memory logging (if it is not already enabled), i.e. all log statements are written to a memory buffer and can later be
    read via function `get_memory_log()`.
    """
    global _memoryLogStream
    if _memoryLogStream is not None:
        return
    _memoryLogStream = StringIO()
    handler = StreamHandler(_memoryLogStream)
    handler.setFormatter(Formatter(_logFormat))
    Logger.root.addHandler(handler)


def get_memory_log() -> Any:
    """:return: the in-memory log (provided that `add_memory_logger` was called beforehand)"""
    assert _memoryLogStream is not None, "This should not have happened and might be a bug."
    return _memoryLogStream.getvalue()


class FileLoggerContext:
    def __init__(self, path: str, enabled: bool = True):
        self.enabled = enabled
        self.path = path
        self._log_handler: Handler | None = None

    def __enter__(self) -> None:
        if self.enabled:
            self._log_handler = add_file_logger(self.path, register_atexit=False)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._log_handler is not None:
            remove_log_handler(self._log_handler)
