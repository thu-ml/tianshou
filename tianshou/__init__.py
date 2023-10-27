from tianshou import data, env, exploration, policy, trainer, utils

__version__ = "1.0.0"

__all__ = [
    "env",
    "data",
    "utils",
    "policy",
    "trainer",
    "exploration",
]


def _configure_logging() -> None:
    from sensai.util import logging

    def logging_configure_callback() -> None:
        logging.getLogger("numba").setLevel(logging.INFO)

    logging.set_configure_callback(logging_configure_callback)


_configure_logging()
