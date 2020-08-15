import gym
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Callable, Any


class EnvWorker(ABC, gym.Env):
    """An abstract worker for an environment."""

    def __init__(self, env_fn: Callable[[], gym.Env]) -> None:
        self._env_fn = env_fn
        self.is_closed = False

    def __getattribute__(self, key: str) -> Any:
        """Any class who inherits ``gym.Env`` will inherit some attributes,
        like ``action_space``. However, we would like the attribute lookup to
        go straight into the worker (in fact, this vector env's action_space
        is always None).
        """
        if key in ['metadata', 'reward_range', 'spec', 'action_space',
                   'observation_space']:  # reserved keys in gym.Env
            return self.__getattr__(key)
        else:
            return super().__getattribute__(key)

    @abstractmethod
    def __getattr__(self, key: str) -> Any:
        pass

    @abstractmethod
    def reset(self) -> Any:
        pass

    @abstractmethod
    def send_action(self, action: np.ndarray) -> None:
        pass

    def get_result(self) -> Tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    def wait(workers: List['EnvWorker'],
             wait_num: int,
             timeout: Optional[float] = None) -> List['EnvWorker']:
        """Given a list of workers, return those ready ones."""
        raise NotImplementedError

    @abstractmethod
    def seed(self, seed: Optional[int] = None) -> List[int]:
        pass

    @abstractmethod
    def render(self, **kwargs) -> Any:
        """Renders the environment."""
        pass

    @abstractmethod
    def close_env(self) -> None:
        pass

    def close(self) -> None:
        if self.is_closed:
            return None
        self.is_closed = True
        self.close_env()
