from tianshou import data, env, exploration, policy, trainer, utils

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
    "policy",
    "trainer",
    "exploration",
]
