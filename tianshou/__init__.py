# isort: skip_file
# NOTE: Import order is important to avoid circular import errors!
from tianshou import data, env, exploration, algorithm, trainer, utils

__version__ = "1.2.0-dev"


def _register_log_config_callback() -> None:
    from sensai.util import logging

    def configure() -> None:
        logging.getLogger("numba").setLevel(logging.INFO)

    logging.set_configure_callback(configure)


_register_log_config_callback()


__all__ = [
    "env",
    "data",
    "utils",
    "algorithm",
    "trainer",
    "exploration",
]
