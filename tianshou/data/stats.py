from collections.abc import MutableMapping
from dataclasses import InitVar, asdict, dataclass, field

import numpy as np


# adapted from https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
def flatten_dict(
    dictionary: dict,
    parent_key="",
    separator="/",
    exclude_arrays: bool = True,
    exclude_none: bool = True,
):
    """Flatten a nested dictionary."""
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(
                flatten_dict(
                    value,
                    new_key,
                    separator=separator,
                    exclude_arrays=exclude_arrays,
                ).items(),
            )
        else:
            if exclude_arrays and isinstance(value, np.ndarray | list):
                continue
            if exclude_none and value is None:
                continue
            items.append((new_key, value))
    return dict(items)


@dataclass(kw_only=True)
class BaseStats:
    """Abstract base class for statistics."""

    def update(self, stats: dict):
        for k, v in stats.items():
            if np.isscalar(v):
                setattr(self, k, v)
            elif isinstance(v, np.ndarray | list):
                setattr(self, k, ArrayStats(_array=v))
            elif isinstance(v, BaseStats):
                getattr(self, k).update(v.to_dict())

    def to_dict(self, exclude: set[str] | None = None, exclude_arrays=True, exclude_none=True):
        """Convert the dataclass to a dictionary.

        :param exclude: set of field names to exclude from the dictionary.
        :param exclude_arrays: if True, exclude arrays from the dictionary.
        :param exclude_none: if True, exclude None from the dictionary.
        :return: a dictionary of the dataclass instance
        """
        stat_dict = flatten_dict(
            asdict(self),
            exclude_arrays=exclude_arrays,
            exclude_none=exclude_none,
        )
        if exclude is not None:
            stat_dict = {k: v for k, v in stat_dict.items() if k not in exclude}
        return stat_dict


@dataclass(kw_only=True)
class ArrayStats(BaseStats):
    """A data structure for storing the statistics of an array."""

    _array: np.ndarray | list[float] = field(repr=False)
    """The array to be analyzed."""
    mean: float = field(init=False)
    """The mean of the array."""
    std: float = field(init=False)
    """The standard deviation of the array."""
    max: float = field(init=False)
    """The maximum of the array."""
    min: float = field(init=False)
    """The minimum of the array."""

    def __post_init__(self):
        self.mean: float = np.mean(self._array)
        """The mean of the array."""
        self.std: float = np.std(self._array)
        """The standard deviation of the array."""
        self.max: float = np.max(self._array)
        """The maximum of the array."""
        self.min: float = np.min(self._array)
        """The minimum of the array."""


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
    array_rews: InitVar[np.ndarray]
    """The collected episodes' returns."""
    array_lens: InitVar[np.ndarray]
    """The collected episodes' lengths."""
    rews: ArrayStats = field(init=False)
    """Stats of the collected returns."""
    lens: ArrayStats = field(init=False)
    """Stats of the collected lengths."""

    def __post_init__(self, array_rews, array_lens):
        if array_rews is not None:
            if len(array_rews) > 0:
                self.rews = ArrayStats(_array=array_rews)
            else:
                self.rews = array_rews
        else:
            self.rews = None
        if array_lens is not None:
            if len(array_lens) > 0:
                self.lens = ArrayStats(_array=array_lens)
            else:
                self.lens = array_lens
        else:
            self.lens = None


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
