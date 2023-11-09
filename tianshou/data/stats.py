import numpy as np
import numpy.typing as npt
from abc import ABC
from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Union


@dataclass
class CollectorStats:
    """A data structure for storing the statistics of the collector."""

    n_collected_episodes: int = 0
    n_collected_steps: int = 0
    rews: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    lens: npt.NDArray[np.int64] = field(default_factory=lambda: np.array([]))
    idxs: npt.NDArray[np.int64] = field(default_factory=lambda: np.array([]))
    rew_mean: float = 0.  # should names stay the same?
    rew_std: float = 0.
    len_mean: float = 0.  # should names stay the same?
    len_std: float = 0.

    def get_test_data_dict(self):
        log_data = {
            "test/reward": self.rew_mean,
            "test/length": self.len_mean,
            "test/reward_std": self.rew_std,
            "test/length_std": self.len_std,
        }

        return log_data

    def get_train_data_dict(self):
        log_data = {
            "train/episode": self.n_collected_episodes,
            "train/reward": self.rew_mean,
            "train/length": self.len_mean,
        }

        return log_data


@dataclass
class Stats(ABC):
    def update(self, stats: Union[dict, dataclass]):
        if is_dataclass(stats):
            stats = asdict(stats)
        for k, v in stats.items():
            setattr(self, k, v)


@dataclass
class EpochStats(Stats):
    """A data structure for storing episode statistics."""

    env_step: int
    gradient_step: int
    n_collected_episodes: int
    n_collected_steps: int
    rew: float
    len: int


@dataclass
class TestStats(Stats):
    """A data structure for storing test statistics."""

    test_reward: float
    test_reward_std: float
    best_reward: float
    best_reward_std: float
    best_epoch: int
