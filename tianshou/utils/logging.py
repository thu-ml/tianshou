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
from typing import Any

log = getLogger(__name__)

LOG_DEFAULT_FORMAT = "%(levelname)-5s %(asctime)-15s %(name)s:%(funcName)s - %(message)s"
_logFormat = LOG_DEFAULT_FORMAT


def remove_log_handlers():
    """Removes all current log handlers."""
    logger = getLogger()
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])


def is_log_handler_active(handler):
    """Checks whether the given handler is active.

    :param handler: a log handler
    :return: True if the handler is active, False otherwise
    """
    return handler in getLogger().handlers


# noinspection PyShadowingBuiltins
def configure(format=LOG_DEFAULT_FORMAT, level=lg.DEBUG):
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


# noinspection PyShadowingBuiltins
def run_main(main_fn: Callable[[], Any], format=LOG_DEFAULT_FORMAT, level=lg.DEBUG):
    """Configures logging with the given parameters, ensuring that any exceptions that occur during
    the execution of the given function are logged.
    Logs two additional messages, one before the execution of the function, and one upon its completion.

    :param main_fn: the function to be executed
    :param format: the log message format
    :param level: the minimum log level
    :return: the result of `main_fn`
    """
    configure(format=format, level=level)
    log.info("Starting")
    try:
        result = main_fn()
        log.info("Done")
        return result
    except Exception as e:
        log.error("Exception during script execution", exc_info=e)


def datetime_tag() -> str:
    """:return: a string tag for use in log file names which contains the current date and time (compact but readable)"""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


_fileLoggerPaths: list[str] = []
_isAtExitReportFileLoggerRegistered = False
_memoryLogStream: StringIO | None = None


def _at_exit_report_file_logger():
    for path in _fileLoggerPaths:
        print(f"A log file was saved to {path}")


def add_file_logger(path):
    global _isAtExitReportFileLoggerRegistered
    log.info(f"Logging to {path} ...")
    handler = FileHandler(path)
    handler.setFormatter(Formatter(_logFormat))
    Logger.root.addHandler(handler)
    _fileLoggerPaths.append(path)
    if not _isAtExitReportFileLoggerRegistered:
        atexit.register(_at_exit_report_file_logger)
        _isAtExitReportFileLoggerRegistered = True


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


def get_memory_log():
    """:return: the in-memory log (provided that `add_memory_logger` was called beforehand)"""
    return _memoryLogStream.getvalue()
