from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(kw_only=True)
class BaseStats:
    """This class serves as a base class for all statistics data structures."""
    def update(self, stats: dict):
        for k, v in stats.items():
            assert hasattr(self, k), f"Unknown key {k} in stats dict {stats}."
            setattr(self, k, v)


@dataclass(kw_only=True)
class SequenceSummaryStats(BaseStats):
    """A data structure for storing the statistics of a sequence."""

    mean: float
    std: float
    max: float
    min: float

    @classmethod
    def from_sequence(cls, sequence: Sequence[float]):
        return cls(
            mean=np.mean(sequence),
            std=np.std(sequence),
            max=np.max(sequence),
            min=np.min(sequence),
        )


@dataclass(kw_only=True)
class UpdateStats(BaseStats):
    """A data structure for storing statistics of the policy update step."""

    train_time: float = 0.0
    """The time for learning models."""
    loss: BaseStats
    """The loss statistics of the policy learn step."""
    smoothed_loss: dict = field(default_factory=dict)
    """The smoothed loss statistics of the policy learn step."""


@dataclass(kw_only=True)
class CollectStats(BaseStats):
    """A data structure for storing the statistics of the collector."""

    n_collected_episodes: int = 0
    """The number of collected episodes."""
    n_collected_steps: int = 0
    """The number of collected steps."""
    collect_time: float = 0.0
    """The time for collecting transitions."""
    collect_speed: float = 0.0
    """The speed of collecting (env_step per second)."""
    rews: np.ndarray = None
    """The collected episode returns."""
    rews_stat: SequenceSummaryStats = None
    """Stats of the collected returns."""
    lens: np.ndarray = None
    """The collected episode lengths."""
    lens_stat: SequenceSummaryStats = None
    """Stats of the collected episode lengths."""


@dataclass(kw_only=True)
class InfoStats(BaseStats):
    """A data structure for storing information about the learning process."""

    gradient_step: int
    """The total gradient step."""
    best_reward: float
    """The best reward over the test results."""
    best_reward_std: float
    """Standard deviation of the best reward over the test results."""
    train_step: int
    """The total collected step of training collector."""
    train_episode: int
    """The total collected episode of training collector."""
    test_step: int
    """The total collected step of test collector."""
    test_episode: int
    """The total collected episode of test collector."""

    timing: BaseStats
    """The timing statistics."""


@dataclass(kw_only=True)
class EpochStats(BaseStats):
    """A data structure for storing episode statistics."""

    epoch: int
    """The current epoch."""

    train_stat: CollectStats = None
    """The statistics of the last call to the training collector."""
    test_stat: CollectStats = None
    """The statistics of the last call to the test collector."""
    update_stat: UpdateStats = None
    """The statistics of the last model update step."""
    info_stat: InfoStats = None
    """The information of the collector."""
