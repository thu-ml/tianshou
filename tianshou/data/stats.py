from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from tianshou.utils.print import DataclassPPrintMixin

if TYPE_CHECKING:
    from tianshou.data import CollectStats, CollectStatsBase
    from tianshou.policy.base import TrainingStats


@dataclass(kw_only=True)
class SequenceSummaryStats(DataclassPPrintMixin):
    """A data structure for storing the statistics of a sequence."""

    mean: float
    std: float
    max: float
    min: float

    @classmethod
    def from_sequence(cls, sequence: Sequence[float | int] | np.ndarray) -> "SequenceSummaryStats":
        return cls(
            mean=float(np.mean(sequence)),
            std=float(np.std(sequence)),
            max=float(np.max(sequence)),
            min=float(np.min(sequence)),
        )


@dataclass(kw_only=True)
class TimingStats(DataclassPPrintMixin):
    """A data structure for storing timing statistics."""

    total_time: float = 0.0
    """The total time elapsed."""
    train_time: float = 0.0
    """The total time elapsed for training (collecting samples plus model update)."""
    train_time_collect: float = 0.0
    """The total time elapsed for collecting training transitions."""
    train_time_update: float = 0.0
    """The total time elapsed for updating models."""
    test_time: float = 0.0
    """The total time elapsed for testing models."""
    update_speed: float = 0.0
    """The speed of updating (env_step per second)."""


@dataclass(kw_only=True)
class InfoStats(DataclassPPrintMixin):
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

    timing: TimingStats
    """The timing statistics."""


@dataclass(kw_only=True)
class EpochStats(DataclassPPrintMixin):
    """A data structure for storing epoch statistics."""

    epoch: int
    """The current epoch."""

    train_collect_stat: "CollectStatsBase"
    """The statistics of the last call to the training collector."""
    test_collect_stat: Optional["CollectStats"]
    """The statistics of the last call to the test collector."""
    training_stat: "TrainingStats"
    """The statistics of the last model update step."""
    info_stat: InfoStats
    """The information of the collector."""
