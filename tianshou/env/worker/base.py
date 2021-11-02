from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Tuple

import gym
import numpy as np


class EnvWorker(ABC):
    """An abstract worker for an environment."""

    def __init__(self, env_fn: Callable[[], gym.Env]) -> None:
        self._env_fn = env_fn
        self.is_closed = False
        self.result: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        self.action_space = self.get_env_attr("action_space")  # noqa: B009

    @abstractmethod
    def get_env_attr(self, key: str) -> Any:
        pass

    @abstractmethod
    def set_env_attr(self, key: str, value: Any) -> None:
        pass

    @abstractmethod
    def reset(self) -> Any:
        pass

    @abstractmethod
    def send_action(self, action: np.ndarray) -> None:
        pass

    def get_result(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.result

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Perform one timestep of the environment's dynamic.

        "send_action" and "get_result" are coupled in sync simulation, so
        typically users only call "step" function. But they can be called
        separately in async simulation, i.e. someone calls "send_action" first,
        and calls "get_result" later.
        """
        self.send_action(action)
        return self.get_result()

    @staticmethod
    def wait(
        workers: List["EnvWorker"],
        wait_num: int,
        timeout: Optional[float] = None
    ) -> List["EnvWorker"]:
        """Given a list of workers, return those ready ones."""
        raise NotImplementedError

    def seed(self, seed: Optional[int] = None) -> Optional[List[int]]:
        return self.action_space.seed(seed)  # issue 299

    @abstractmethod
    def render(self, **kwargs: Any) -> Any:
        """Render the environment."""
        pass

    @abstractmethod
    def close_env(self) -> None:
        pass

    def close(self) -> None:
        if self.is_closed:
            return None
        self.is_closed = True
        self.close_env()
