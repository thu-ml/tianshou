from typing import TYPE_CHECKING, Any, Tuple, Union

import cloudpickle
import gymnasium
import numpy as np

try:
    from tianshou.env.pettingzoo_env import PettingZooEnv
except ImportError:
    PettingZooEnv = None  # type: ignore

if TYPE_CHECKING:
    import gym

ENV_TYPE = Union[gymnasium.Env, "gym.Env", PettingZooEnv]

gym_new_venv_step_type = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                               np.ndarray]


class CloudpickleWrapper(object):
    """A cloudpickle wrapper used in SubprocVectorEnv."""

    def __init__(self, data: Any) -> None:
        self.data = data

    def __getstate__(self) -> str:
        return cloudpickle.dumps(self.data)

    def __setstate__(self, data: str) -> None:
        self.data = cloudpickle.loads(data)
