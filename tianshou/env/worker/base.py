import gym
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Callable, Any


class EnvWorker(ABC, gym.Env):
    """An abstract worker for an environment."""

    def __init__(self, env_fn: Callable[[], gym.Env]) -> None:
        self._env_fn = env_fn

    def __getattribute__(self, key: str):
        if key not in ('observation_space', 'action_space'):
            return super().__getattribute__(key)
        else:
            return self.__getattr__(key)

    @abstractmethod
    def __getattr__(self, key: str) -> Any:
        pass

    @abstractmethod
    def reset(self) -> Any:
        pass

    @abstractmethod
    def send_action(self, action: np.ndarray) -> None:
        pass

    def get_result(self
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.result

    def step(self, action: np.ndarray
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``send_action`` and ``get_result`` are coupled in sync simulation,
        so typically users only call ``step`` function. But they can be called
        separately in async simulation, i.e. someone calls ``send_action``
        first, and calls ``get_result`` later.
        """
        self.send_action(action)
        return self.get_result()

    @staticmethod
    def wait(workers: List['EnvWorker']) -> List['EnvWorker']:
        """Given a list of workers, return those ready ones."""
        raise NotImplementedError

    @abstractmethod
    def seed(self, seed: Optional[int] = None) -> int:
        pass

    @abstractmethod
    def render(self, **kwargs) -> Any:
        """Renders the environment."""
        pass

    @abstractmethod
    def close(self) -> Any:
        pass
