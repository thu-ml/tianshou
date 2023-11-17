from typing import Literal
from pydantic import RootModel
from pydantic.dataclasses import dataclass


@dataclass(kw_only=True, config=dict(arbitrary_types_allowed=True, extra='allow'))
class Stats:
    def update(self, stats: dict):
        for k, v in stats.items():
            setattr(self, k, v)

    def to_dict(self,
                mode: Literal["python", "json"] = "python",
                exclude: set[str] = None):
        return RootModel(self).model_dump(mode=mode,
                                          exclude=exclude)


@dataclass
class CollectorStats(Stats):
    """A data structure for storing the statistics of the collector."""

    n_collected_episodes: int = 0
    """The total collected episode of the collector."""
    n_collected_steps: int = 0
    """The total collected step of the collector."""
    rew_mean: float = 0.
    """The mean of the collected rewards."""
    rew_std: float = 0.
    """The standard deviation of the collected rewards."""
    len_mean: float = 0.
    """The mean of the collected episode lengths."""
    len_std: float = 0.
    """The standard deviation of the collected episode lengths."""


@dataclass
class InfoStats(Stats):
    """A data structure for storing information from collectors."""

    train_step: int = None
    """The total collected step of training collector."""
    train_episode: int = None
    """The total collected episode of training collector."""
    train_time_collector: float = None
    """The time for collecting transitions in the training collector."""
    train_time_model: float = None
    """The time for training models."""
    train_speed: float = None
    """The speed of training (env_step per second)."""
    test_step: int = None
    """The total collected step of test collector."""
    test_episode: int = None
    """The total collected episode of test collector."""
    test_time: float = None
    """The time for testing."""
    test_speed: float = None
    """The speed of testing (env_step per second)."""
    best_reward: float = None
    """The best reward over the test results."""
    best_reward_std: float = None
    """Standard deviation of the best reward over the test results."""
    duration: float = None
    """The total elapsed time."""


@dataclass
class EpochStats(Stats):
    """A data structure for storing episode statistics."""

    epoch: int
    """The current epoch."""
    env_step: int
    """The total collected step of training collector."""
    gradient_step: int
    """The total gradient step."""

    train_stat: CollectorStats = None
    """The statistics of the training collector."""
    test_stat: CollectorStats = None
    """The statistics of the test collector."""
    losses: dict[str, float] = None
    """The losses of the model."""
    info: InfoStats = None
    """The information of the collector."""
