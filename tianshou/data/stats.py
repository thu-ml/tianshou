import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from tianshou.utils.print import DataclassPPrintMixin

if TYPE_CHECKING:
    from tianshou.data import CollectStats, CollectStatsBase
    from tianshou.policy.base import TrainingStats

log = logging.getLogger(__name__)


@dataclass(kw_only=True)
class SequenceSummaryStats(DataclassPPrintMixin):
    """A data structure for storing the statistics of a sequence."""

    mean: float
    std: float
    max: float
    min: float

    @classmethod
    def from_sequence(cls, sequence: Sequence[float | int] | np.ndarray) -> "SequenceSummaryStats":
        if len(sequence) == 0:
            return cls(mean=0.0, std=0.0, max=0.0, min=0.0)

        if hasattr(sequence, "shape") and len(sequence.shape) > 1:
            log.warning(
                f"Sequence has shape {sequence.shape}, but only 1D sequences are supported. "
                "Stats will be computed from the flattened sequence. For computing stats "
                "for each dimension consider using the function `compute_dim_to_summary_stats`.",
            )

        return cls(
            mean=float(np.mean(sequence)),
            std=float(np.std(sequence)),
            max=float(np.max(sequence)),
            min=float(np.min(sequence)),
        )


def compute_dim_to_summary_stats(
    arr: Sequence[Sequence[float]] | np.ndarray,
) -> dict[int, SequenceSummaryStats]:
    """Compute summary statistics for each dimension of a sequence.

    :param arr: a 2-dim arr (or sequence of sequences) from which to compute the statistics.
    :return: A dictionary of summary statistics for each dimension.
    """
    stats = {}
    for dim, seq in enumerate(arr):
        stats[dim] = SequenceSummaryStats.from_sequence(seq)
    return stats


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
    best_score: float
    """The best score over the test results. The one with the highest score will be considered the best model."""
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
    training_stat: Optional["TrainingStats"]
    """The statistics of the last model update step.
    Can be None if no model update is performed, typically in the last training iteration."""
    info_stat: InfoStats
    """The information of the collector."""
