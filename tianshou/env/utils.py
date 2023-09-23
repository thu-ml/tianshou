from typing import Any

import cloudpickle
import gymnasium
import numpy as np

from tianshou.env.pettingzoo_env import PettingZooEnv

ENV_TYPE = gymnasium.Env | PettingZooEnv

gym_new_venv_step_type = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class CloudpickleWrapper:
    """A cloudpickle wrapper used in SubprocVectorEnv."""

    def __init__(self, data: Any) -> None:
        self.data = data

    def __getstate__(self) -> str:
        return cloudpickle.dumps(self.data)

    def __setstate__(self, data: str) -> None:
        self.data = cloudpickle.loads(data)
