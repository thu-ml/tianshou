import logging
import time
import warnings
from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass, field
from typing import Any, Generic, Optional, Protocol, Self, TypedDict, TypeVar, cast

import gymnasium as gym
import numpy as np
import torch
from overrides import override
from torch.distributions import Categorical, Distribution

from tianshou.data import (
    Batch,
    CachedReplayBuffer,
    ReplayBuffer,
    ReplayBufferManager,
    SequenceSummaryStats,
    VectorReplayBuffer,
    to_numpy,
)
from tianshou.data.buffer.base import MalformedBufferError
from tianshou.data.stats import compute_dim_to_summary_stats
from tianshou.data.types import (
    ActBatchProtocol,
    DistBatchProtocol,
    ObsBatchProtocol,
    RolloutBatchProtocol,
)
from tianshou.env import BaseVectorEnv, DummyVectorEnv
from tianshou.policy import BasePolicy
from tianshou.policy.base import episode_mc_return_to_go
from tianshou.utils.print import DataclassPPrintMixin
from tianshou.utils.torch_utils import torch_train_mode

log = logging.getLogger(__name__)

DEFAULT_BUFFER_MAXSIZE = int(1e4)

_TArrLike = TypeVar("_TArrLike", bound="np.ndarray | torch.Tensor | Batch | None")


class CollectActionBatchProtocol(Protocol):
    """A protocol for results of computing actions from a batch of observations within a single collect step.

    All fields all have length R (the dist is a Distribution of batch size R),
    where R is the number of ready envs.
    """

    act: np.ndarray | torch.Tensor
    act_normalized: np.ndarray | torch.Tensor
    policy_entry: Batch
    dist: Distribution | None
    hidden_state: np.ndarray | torch.Tensor | Batch | None


class CollectStepBatchProtocol(RolloutBatchProtocol):
    """A batch of steps collected from a single collect step from multiple envs in parallel.

    All fields have length R (the dist is a Distribution of batch size R), where R is the number of ready envs.
    This is essentially the response of the vectorized environment to making a step
    with :class:`CollectActionBatchProtocol`.
    """

    dist: Distribution | None


class EpisodeBatchProtocol(RolloutBatchProtocol):
    """Marker interface for a batch containing a single episode.

    Instances are created by retrieving an episode from the buffer when the :class:`Collector` encounters
    `done=True`.
    """


def get_stddev_from_dist(dist: Distribution) -> torch.Tensor:
    """Return the standard deviation of the given distribution.

    Same as `dist.stddev` for all distributions except `Categorical`, where it is computed
    by assuming that the output values 0, ..., K have the corresponding numerical meaning.
    See `here <https://discuss.pytorch.org/t/pytorch-distribution-mean-returns-nan/61978/9>`_
    for a discussion on `stddev` and `mean` of `Categorical`.
    """
    if isinstance(dist, Categorical):
        # torch doesn't implement stddev for Categorical, so we compute it ourselves
        probs = torch.atleast_2d(dist.probs)
        n_actions = probs.shape[-1]
        possible_actions = torch.arange(n_actions, device=dist.probs.device).float()

        mean = torch.sum(probs * possible_actions, dim=1)
        var = torch.sum(probs * (possible_actions - mean.unsqueeze(1)) ** 2, dim=1)
        stddev = torch.sqrt(var)
        if len(dist.batch_shape) == 0:
            return stddev
        return torch.atleast_2d(stddev).T

    return dist.stddev if dist is not None else torch.tensor([])


@dataclass(kw_only=True)
class CollectStatsBase(DataclassPPrintMixin):
    """The most basic stats, often used for offline learning."""

    n_collected_episodes: int = 0
    """The number of collected episodes."""
    n_collected_steps: int = 0
    """The number of collected steps."""


@dataclass(kw_only=True)
class CollectStats(CollectStatsBase):
    """A data structure for storing the statistics of rollouts.

    Custom stats collection logic can be implemented by subclassing this class and
    overriding the `update_` methods.

    Ideally, it is instantiated once with correct values and then never modified.
    However, during the collection process instances of modified
    using the `update_` methods. Then the arrays and their corresponding  `_stats` fields
    may become out of sync (we don't update the stats after each update for performance reasons,
    only at the end of the collection). The same for the `collect_time` and `collect_speed`.
    In the `Collector`, :meth:`refresh_sequence_stats` and :meth:`set_collect_time` are
    is called at the end of the collection to update the stats. But for other use cases,
    the users should keep in mind to call this method manually if using the `update_`
    methods.
    """

    collect_time: float = 0.0
    """The time for collecting transitions."""
    collect_speed: float = 0.0
    """The speed of collecting (env_step per second)."""
    returns: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    """The collected episode returns."""
    returns_stat: SequenceSummaryStats | None = None
    """Stats of the collected returns."""
    lens: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    """The collected episode lengths."""
    lens_stat: SequenceSummaryStats | None = None
    """Stats of the collected episode lengths."""
    pred_dist_std_array: np.ndarray | None = None
    """The standard deviations of the predicted distributions."""
    pred_dist_std_array_stat: dict[int, SequenceSummaryStats] | None = None
    """Stats of the standard deviations of the predicted distributions (maps action dim to stats)"""

    @classmethod
    def with_autogenerated_stats(
        cls,
        returns: np.ndarray,
        lens: np.ndarray,
        n_collected_episodes: int = 0,
        n_collected_steps: int = 0,
        collect_time: float = 0.0,
        collect_speed: float = 0.0,
    ) -> Self:
        """Return a new instance with the stats autogenerated from the given lists."""
        returns_stat = SequenceSummaryStats.from_sequence(returns) if returns.size > 0 else None
        lens_stat = SequenceSummaryStats.from_sequence(lens) if lens.size > 0 else None
        return cls(
            n_collected_episodes=n_collected_episodes,
            n_collected_steps=n_collected_steps,
            collect_time=collect_time,
            collect_speed=collect_speed,
            returns=returns,
            returns_stat=returns_stat,
            lens=np.array(lens, int),
            lens_stat=lens_stat,
        )

    def update_at_step_batch(
        self,
        step_batch: CollectStepBatchProtocol,
        refresh_sequence_stats: bool = False,
    ) -> None:
        self.n_collected_steps += len(step_batch)
        dist = step_batch.dist
        action_std: torch.Tensor | None = None

        if dist is not None:
            action_std = np.atleast_2d(to_numpy(get_stddev_from_dist(dist)))

            if self.pred_dist_std_array is None:
                self.pred_dist_std_array = np.atleast_2d(to_numpy(action_std))
            else:
                self.pred_dist_std_array = np.concatenate(
                    (self.pred_dist_std_array, np.atleast_2d(to_numpy(action_std))),
                )
        if refresh_sequence_stats:
            self.refresh_std_array_stats()

    def update_at_episode_done(
        self,
        episode_batch: EpisodeBatchProtocol,
        # NOTE: in the MARL setting this is not actually a float but rather an array or list, see todo below
        episode_return: float,
        refresh_sequence_stats: bool = False,
    ) -> None:
        self.lens = np.concatenate((self.lens, [len(episode_batch)]), dtype=int)  # type: ignore
        self.n_collected_episodes += 1
        if self.returns.size == 0:
            # TODO: needed for non-1dim arrays returns that happen in the MARL setting
            #   There are multiple places that assume the returns to be 1dim, so this is a hack
            #   Since MARL support is currently not a priority, we should either raise an error or
            #   implement proper support for it. At the moment tests like `test_collector_with_multi_agent` fail
            #   when assuming 1d returns
            self.returns = np.array([episode_return], dtype=float)
        else:
            self.returns = np.concatenate((self.returns, [episode_return]), dtype=float)  # type: ignore
        if refresh_sequence_stats:
            self.refresh_return_stats()
            self.refresh_len_stats()

    def set_collect_time(self, collect_time: float, update_collect_speed: bool = True) -> None:
        if collect_time < 0:
            raise ValueError(f"Collect time should be non-negative, but got {collect_time=}.")

        self.collect_time = collect_time
        if update_collect_speed:
            if collect_time == 0:
                log.error(
                    "Collect time is 0, setting collect speed to 0. Did you make a rounding error?",
                )
                self.collect_speed = 0.0
            else:
                self.collect_speed = self.n_collected_steps / collect_time

    def refresh_return_stats(self) -> None:
        if self.returns.size > 0:
            self.returns_stat = SequenceSummaryStats.from_sequence(self.returns)
        else:
            self.returns_stat = None

    def refresh_len_stats(self) -> None:
        if self.lens.size > 0:
            self.lens_stat = SequenceSummaryStats.from_sequence(self.lens)
        else:
            self.lens_stat = None

    def refresh_std_array_stats(self) -> None:
        if self.pred_dist_std_array is not None and self.pred_dist_std_array.size > 0:
            # need to use .T because action dim supposed to be the first axis in compute_dim_to_summary_stats
            self.pred_dist_std_array_stat = compute_dim_to_summary_stats(self.pred_dist_std_array.T)
        else:
            self.pred_dist_std_array_stat = None

    def refresh_all_sequence_stats(self) -> None:
        self.refresh_return_stats()
        self.refresh_len_stats()
        self.refresh_std_array_stats()


TCollectStats = TypeVar("TCollectStats", bound=CollectStats)


def _nullable_slice(obj: _TArrLike, indices: np.ndarray) -> _TArrLike:
    """Return None, or the values at the given indices if the object is not None."""
    if obj is not None:
        return obj[indices]  # type: ignore[index, return-value]
    return None  # type: ignore[unreachable]


def _dict_of_arr_to_arr_of_dicts(dict_of_arr: dict[str, np.ndarray | dict]) -> np.ndarray:
    return np.array(Batch(dict_of_arr).to_list_of_dicts())


def _HACKY_create_info_batch(info_array: np.ndarray) -> Batch:
    """TODO: this exists because of multiple bugs in Batch and to restore backwards compatibility.
    Batch should be fixed and this function should be removed asap!.
    """
    if info_array.dtype != np.dtype("O"):
        raise ValueError(
            f"Expected info_array to have dtype=object, but got {info_array.dtype}.",
        )

    truthy_info_indices = info_array.nonzero()[0]
    falsy_info_indices = set(range(len(info_array))) - set(truthy_info_indices)
    falsy_info_indices = np.array(list(falsy_info_indices), dtype=int)

    if len(falsy_info_indices) == len(info_array):
        return Batch()

    some_nonempty_info = None
    for info in info_array:
        if info:
            some_nonempty_info = info
            break

    info_array = copy(info_array)
    info_array[falsy_info_indices] = some_nonempty_info
    result_batch_parent = Batch(info=info_array)
    result_batch_parent.info[falsy_info_indices] = {}
    return result_batch_parent.info


class BaseCollector(Generic[TCollectStats], ABC):
    """Used to collect data from a vector environment into a buffer using a given policy.

    .. note::

        Please make sure the given environment has a time limitation if using `n_episode`
        collect option.

    .. note::

        In past versions of Tianshou, the replay buffer passed to `__init__`
        was automatically reset. This is not done in the current implementation.
    """

    def __init__(
        self,
        policy: BasePolicy,
        env: BaseVectorEnv | gym.Env,
        buffer: ReplayBuffer | None = None,
        exploration_noise: bool = False,
        # The typing is correct, there's a bug in mypy, see https://github.com/python/mypy/issues/3737
        collect_stats_class: type[TCollectStats] = CollectStats,  # type: ignore[assignment]
        raise_on_nan_in_buffer: bool = True,
    ) -> None:
        if isinstance(env, gym.Env) and not hasattr(env, "__len__"):
            warnings.warn("Single environment detected, wrap to DummyVectorEnv.")
            # Unfortunately, mypy seems to ignore the isinstance in lambda, maybe a bug in mypy
            env = DummyVectorEnv([lambda: env])  # type: ignore

        if buffer is None:
            buffer = VectorReplayBuffer(DEFAULT_BUFFER_MAXSIZE * len(env), len(env))

        self.buffer: ReplayBuffer | ReplayBufferManager = buffer
        self.raise_on_nan_in_buffer = raise_on_nan_in_buffer
        self.policy = policy
        self.env = cast(BaseVectorEnv, env)
        self.exploration_noise = exploration_noise
        self.collect_step, self.collect_episode, self.collect_time = 0, 0, 0.0

        self._action_space = self.env.action_space
        self._is_closed = False

        self._validate_buffer()
        self.collect_stats_class = collect_stats_class

    def _validate_buffer(self) -> None:
        buf = self.buffer
        # TODO: a bit weird but true - all VectorReplayBuffers inherit from ReplayBufferManager.
        #  We should probably rename the manager
        if isinstance(buf, ReplayBufferManager) and buf.buffer_num < self.env_num:
            raise ValueError(
                f"Buffer has only {buf.buffer_num} buffers, but at least {self.env_num=} are needed.",
            )
        if isinstance(buf, CachedReplayBuffer) and buf.cached_buffer_num < self.env_num:
            raise ValueError(
                f"Buffer has only {buf.cached_buffer_num} cached buffers, but at least {self.env_num=} are needed.",
            )
        # Non-VectorReplayBuffer. TODO: probably shouldn't rely on isinstance
        if not isinstance(buf, ReplayBufferManager):
            if buf.maxsize == 0:
                raise ValueError("Buffer maxsize should be greater than 0.")
            if self.env_num > 1:
                raise ValueError(
                    f"Cannot use {type(buf).__name__} to collect from multiple envs ({self.env_num=}). "
                    f"Please use the corresponding VectorReplayBuffer instead.",
                )

    @property
    def env_num(self) -> int:
        return len(self.env)

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    def close(self) -> None:
        """Close the collector and the environment."""
        self.env.close()
        self._is_closed = True

    def reset(
        self,
        reset_buffer: bool = True,
        reset_stats: bool = True,
        gym_reset_kwargs: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reset the environment, statistics, and data needed to start the collection.

        :param reset_buffer: if true, reset the replay buffer attached
            to the collector.
        :param reset_stats: if true, reset the statistics attached to the collector.
        :param gym_reset_kwargs: extra keyword arguments to pass into the environment's
            reset function. Defaults to None (extra keyword arguments)
        :return: The initial observation and info from the environment.
        """
        obs_NO, info_N = self.reset_env(gym_reset_kwargs=gym_reset_kwargs)
        if reset_buffer:
            self.reset_buffer()
        if reset_stats:
            self.reset_stat()
        self._is_closed = False
        return obs_NO, info_N

    def reset_stat(self) -> None:
        """Reset the statistic variables."""
        self.collect_step, self.collect_episode, self.collect_time = 0, 0, 0.0

    def reset_buffer(self, keep_statistics: bool = False) -> None:
        """Reset the data buffer."""
        self.buffer.reset(keep_statistics=keep_statistics)

    def reset_env(
        self,
        gym_reset_kwargs: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reset the environments and the initial obs, info, and hidden state of the collector."""
        gym_reset_kwargs = gym_reset_kwargs or {}
        obs_NO, info_N = self.env.reset(**gym_reset_kwargs)
        # TODO: hack, wrap envpool envs such that they don't return a dict
        if isinstance(info_N, dict):  # type: ignore[unreachable]
            # this can happen if the env is an envpool env. Then the thing returned by reset is a dict
            # with array entries instead of an array of dicts
            # We use Batch to turn it into an array of dicts
            info_N = _dict_of_arr_to_arr_of_dicts(info_N)  # type: ignore[unreachable]
        return obs_NO, info_N

    @abstractmethod
    def _collect(
        self,
        n_step: int | None = None,
        n_episode: int | None = None,
        random: bool = False,
        render: float | None = None,
        gym_reset_kwargs: dict[str, Any] | None = None,
    ) -> TCollectStats:
        pass

    def collect(
        self,
        n_step: int | None = None,
        n_episode: int | None = None,
        random: bool = False,
        render: float | None = None,
        reset_before_collect: bool = False,
        gym_reset_kwargs: dict[str, Any] | None = None,
    ) -> TCollectStats:
        """Collect the specified number of steps or episodes to the buffer.

        .. note::

            One and only one collection specification is permitted, either
            ``n_step`` or ``n_episode``.

        To ensure an unbiased sampling result with the `n_episode` option, this function will
        first collect ``n_episode - env_num`` episodes, then for the last ``env_num``
        episodes, they will be collected evenly from each env.

        :param n_step: how many steps to collect.
        :param n_episode: how many episodes to collect.
        :param random: whether to sample randomly from the action space instead of using the policy for collecting data.
        :param render: the sleep time between rendering consecutive frames.
        :param reset_before_collect: whether to reset the environment before collecting data.
            (The collector needs the initial `obs` and `info` to function properly.)
        :param gym_reset_kwargs: extra keyword arguments to pass into the environment's
            reset function. Only used if reset_before_collect is True.


        :return: The collected stats
        """
        # check that exactly one of n_step or n_episode is set and that the other is larger than 0
        self._validate_n_step_n_episode(n_episode, n_step)

        if reset_before_collect:
            self.reset(reset_buffer=False, gym_reset_kwargs=gym_reset_kwargs)

        pre_collect_time = time.time()
        with torch_train_mode(self.policy, enabled=False):
            collect_stats = self._collect(
                n_step=n_step,
                n_episode=n_episode,
                random=random,
                render=render,
                gym_reset_kwargs=gym_reset_kwargs,
            )
        collect_time = time.time() - pre_collect_time
        collect_stats.set_collect_time(collect_time, update_collect_speed=True)
        collect_stats.refresh_all_sequence_stats()

        if self.raise_on_nan_in_buffer and self.buffer.hasnull():
            nan_batch = self.buffer.isnull().apply_values_transform(np.sum)

            raise MalformedBufferError(
                "NaN detected in the buffer. You can drop them with `buffer.dropnull()`. "
                f"This error is most often caused by an incorrect use of {EpisodeRolloutHook.__name__}"
                "together with the `n_steps` (instead of `n_episodes`) option, or by "
                f"an incorrect implementation of {StepHook.__name__}."
                "Here an overview of the number of NaNs per field: \n"
                f"{nan_batch}",
            )

        return collect_stats

    def _validate_n_step_n_episode(self, n_episode: int | None, n_step: int | None) -> None:
        if not n_step and not n_episode:
            raise ValueError(
                f"Only one of n_step and n_episode should be set to a value larger than zero "
                f"but got {n_step=}, {n_episode=}.",
            )
        if n_step is None and n_episode is None:
            raise ValueError(
                "Exactly one of n_step and n_episode should be set but got None for both.",
            )
        if n_step and n_step % self.env_num != 0:
            warnings.warn(
                f"{n_step=} is not a multiple of ({self.env_num=}), "
                "which may cause extra transitions being collected into the buffer.",
            )
        if n_episode and self.env_num > n_episode:
            warnings.warn(
                f"{n_episode=} should be larger than {self.env_num=} to "
                f"collect at least one trajectory in each environment.",
            )


class Collector(BaseCollector[TCollectStats], Generic[TCollectStats]):
    """Collects transitions from a vectorized env by computing and applying actions batch-wise."""

    # NAMING CONVENTION (mostly suffixes):
    # episode - An episode means a rollout until done (terminated or truncated). After an episode is completed,
    #     the corresponding env is either reset or removed from the ready envs.
    # N - number of envs, always fixed and >= R.
    # R - number ready env ids. Note that this might change when envs get idle.
    #     This can only happen in n_episode case, see explanation in the corresponding block.
    #     For n_step, we always use all envs to collect the data, while for n_episode,
    #     R will be at most n_episode at the beginning, but can decrease during the collection.
    # O - dimension(s) of observations
    # A - dimension(s) of actions
    # H - dimension(s) of hidden state
    # D - number of envs that reached done in the current collect iteration. Only relevant in n_episode case.
    # S - number of surplus envs, i.e., envs that are ready but won't be used in the next iteration.
    #     Only used in n_episode case. Then, R becomes R-S.
    # local_index - selecting from the locally available environments. In more details:
    #     Each env is associated to a number in [0,..., N-1]. At any moment there are R ready envs,
    #     but they are not necessarily equal to [0, ..., R-1]. Let the R corresponding indices be
    #     [r_0, ..., r_(R-1)] (each r_i is in [0, ... N-1]). If the local index is
    #     [0, 1, 2], it means that we want to select envs [r_0, r_1, r_2].
    #     We will usually select from the ready envs by slicing like `ready_env_idx_R[local_index]`
    # global_index - the index in [0, ..., N-1]. Slicing a `_R` index by a local_index produces the
    #     corresponding global index. In the example above:
    #     1. _R index is [r_0, ..., r_(R-1)]
    #     2. local_index is [0, 1, 2]
    #     3. global_index is [r_0, r_1, r_2] and can be used to select from an array of length N
    #
    def __init__(
        self,
        policy: BasePolicy,
        env: gym.Env | BaseVectorEnv,
        buffer: ReplayBuffer | None = None,
        exploration_noise: bool = False,
        on_episode_done_hook: Optional["EpisodeRolloutHookProtocol"] = None,
        on_step_hook: Optional["StepHookProtocol"] = None,
        raise_on_nan_in_buffer: bool = True,
        collect_stats_class: type[TCollectStats] = CollectStats,  # type: ignore[assignment]
    ) -> None:
        """
        :param policy: a tianshou policy, each :class:`BasePolicy` is capable of computing a batch
            of actions from a batch of observations.
        :param env: a ``gymnasium.Env`` environment or a vectorized instance of the
            :class:`~tianshou.env.BaseVectorEnv` class. The latter is strongly recommended, as with
            a gymnasium env the collection will not happen in parallel (a `DummyVectorEnv`
            will be constructed internally from the passed env)
        :param buffer: an instance of the :class:`~tianshou.data.ReplayBuffer` class.
            If set to None, will instantiate a :class:`~tianshou.data.VectorReplayBuffer`
            of size :data:`DEFAULT_BUFFER_MAXSIZE` * (number of envs)
            as the default buffer.
        :param exploration_noise: determine whether the action needs to be modified
            with the corresponding policy's exploration noise. If so, "policy.
            exploration_noise(act, batch)" will be called automatically to add the
            exploration noise into action..
        :param on_episode_done_hook: if passed will be executed when an episode is done.
            The input to the hook will be a `RolloutBatch` that contains the entire episode (and nothing else).
            If a dict is returned by the hook it will be used to add new entries to the buffer
            for the episode that just ended. The values of the dict should be arrays with floats
            of the same length as the input rollout batch.
            Note that multiple hooks can be combined using :class:`EpisodeRolloutHookMerged`.
            A typical example of a hook is :class:`EpisodeRolloutHookMCReturn` which adds the Monte Carlo return
            as a field to the buffer.

            Care must be taken when using such hook, as for unfinished episodes one can easily end
            up with NaNs in the buffer. It is recommended to use the hooks only with the `n_episode` option
            in `collect`, or to strip the buffer of NaNs after the collection.
        :param on_step_hook: if passed will be executed after each step of the collection but before the
            resulting rollout batch is added to the buffer. The inputs to the hook will be
            the action distributions computed from the previous observations (following the
            :class:`CollectActionBatchProtocol`) using the policy, and the resulting
            rollout batch (following the :class:`RolloutBatchProtocol`). **Note** that modifying
            the rollout batch with this hook also modifies the data that is collected to the buffer!
        :param raise_on_nan_in_buffer: whether to raise a `RuntimeError` if NaNs are found in the buffer after
            a collection step. Especially useful when episode-level hooks are passed for making
            sure that nothing is broken during the collection. Consider setting to False if
            the NaN-check becomes a bottleneck.
        :param collect_stats_class: the class to use for collecting statistics. Allows customizing
            the stats collection logic by passing a subclass of :class:`CollectStats`. Changing
            this is rarely necessary and is mainly done by "power users".
        """
        super().__init__(
            policy,
            env,
            buffer,
            exploration_noise=exploration_noise,
            collect_stats_class=collect_stats_class,
            raise_on_nan_in_buffer=raise_on_nan_in_buffer,
        )

        self._pre_collect_obs_RO: np.ndarray | None = None
        self._pre_collect_info_R: np.ndarray | None = None
        self._pre_collect_hidden_state_RH: np.ndarray | torch.Tensor | Batch | None = None

        self._is_closed = False
        self._on_episode_done_hook = on_episode_done_hook
        self._on_step_hook = on_step_hook
        self.collect_step, self.collect_episode, self.collect_time = 0, 0, 0.0

    def set_on_episode_done_hook(self, hook: Optional["EpisodeRolloutHookProtocol"]) -> None:
        self._on_episode_done_hook = hook

    def set_on_step_hook(self, hook: Optional["StepHookProtocol"]) -> None:
        self._on_step_hook = hook

    def get_on_episode_done_hook(self) -> Optional["EpisodeRolloutHookProtocol"]:
        return self._on_episode_done_hook

    def get_on_step_hook(self) -> Optional["StepHookProtocol"]:
        return self._on_step_hook

    def close(self) -> None:
        super().close()
        self._pre_collect_obs_RO = None
        self._pre_collect_info_R = None

    def run_on_episode_done(
        self,
        episode_batch: EpisodeBatchProtocol,
    ) -> dict[str, np.ndarray] | None:
        """Executes the `on_episode_done_hook` that was passed on init.

        One of the main uses of this public method is to allow users to override it in custom
        subclasses of :class:`Collector`. This way, they can override the init to no longer accept
        the `on_episode_done` provider.
        """
        if self._on_episode_done_hook is not None:
            return self._on_episode_done_hook(episode_batch)
        return None

    def run_on_step_hook(
        self,
        action_batch: CollectActionBatchProtocol,
        rollout_batch: RolloutBatchProtocol,
    ) -> None:
        """Executes the instance's `on_step_hook`.

        One of the main uses of this public method is to allow users to override it in custom
        subclasses of the `Collector`. This way, they can override the init to no longer accept
        the `on_step_hook` provider.
        """
        if self._on_step_hook is not None:
            self._on_step_hook(action_batch, rollout_batch)

    def reset_env(
        self,
        gym_reset_kwargs: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reset the environments and the initial obs, info, and hidden state of the collector."""
        obs_NO, info_N = super().reset_env(gym_reset_kwargs=gym_reset_kwargs)
        # We assume that R = N when reset is called.
        # TODO: there is currently no mechanism that ensures this and it's a public method!
        self._pre_collect_obs_RO = obs_NO
        self._pre_collect_info_R = info_N
        self._pre_collect_hidden_state_RH = None
        return obs_NO, info_N

    def _compute_action_policy_hidden(
        self,
        random: bool,
        ready_env_ids_R: np.ndarray,
        last_obs_RO: np.ndarray,
        last_info_R: np.ndarray,
        last_hidden_state_RH: np.ndarray | torch.Tensor | Batch | None = None,
    ) -> CollectActionBatchProtocol:
        """Returns the action, the normalized action, a "policy" entry, and the hidden state."""
        if random:
            try:
                act_normalized_RA = np.array(
                    [self._action_space[i].sample() for i in ready_env_ids_R],
                )
            # TODO: test whether envpool env explicitly
            except TypeError:  # envpool's action space is not for per-env
                act_normalized_RA = np.array([self._action_space.sample() for _ in ready_env_ids_R])
            act_RA = self.policy.map_action_inverse(np.array(act_normalized_RA))
            policy_R = Batch()
            hidden_state_RH = None
            # TODO: instead use a (uniform) Distribution instance that corresponds to sampling from action_space
            action_dist_R = None

        else:
            info_batch = _HACKY_create_info_batch(last_info_R)
            obs_batch_R = cast(ObsBatchProtocol, Batch(obs=last_obs_RO, info=info_batch))

            act_batch_RA: ActBatchProtocol | DistBatchProtocol = self.policy(
                obs_batch_R,
                last_hidden_state_RH,
            )

            act_RA = to_numpy(act_batch_RA.act)
            if self.exploration_noise:
                act_RA = self.policy.exploration_noise(act_RA, obs_batch_R)
            act_normalized_RA = self.policy.map_action(act_RA)

            # TODO: cleanup the whole policy in batch thing
            # todo policy_R can also be none, check
            policy_R = act_batch_RA.get("policy", Batch())
            if not isinstance(policy_R, Batch):
                raise RuntimeError(
                    f"The policy result should be a {Batch}, but got {type(policy_R)}",
                )

            hidden_state_RH = act_batch_RA.get("state", None)
            # TODO: do we need the conditional? Would be better to just add hidden_state which could be None
            if hidden_state_RH is not None:
                policy_R.hidden_state = (
                    hidden_state_RH  # save state into buffer through policy attr
                )
            # can't use act_batch_RA.dist directly as act_batch_RA might not have that attribute
            action_dist_R = act_batch_RA.get("dist")

        return cast(
            CollectActionBatchProtocol,
            Batch(
                act=act_RA,
                act_normalized=act_normalized_RA,
                policy_entry=policy_R,
                dist=action_dist_R,
                hidden_state=hidden_state_RH,
            ),
        )

    # TODO: reduce complexity, remove the noqa
    def _collect(  # noqa: C901
        self,
        n_step: int | None = None,
        n_episode: int | None = None,
        random: bool = False,
        render: float | None = None,
        gym_reset_kwargs: dict[str, Any] | None = None,
    ) -> TCollectStats:
        """This method is currently very complex, but it's difficult to break it down into smaller chunks.

        Please read the block-comment of the class to understand the notation
        in the implementation.

        It does the collection by executing the following logic:

        0. Keep track of n_step and n_episode for being able to stop the collection.
        1.  Create a CollectStats instance to store the statistics of the collection.
        2.  Compute actions (with policy or sampling from action space) for the R currently active envs.
        3.  Perform a step in these R envs.
        4.  Perform on-step hook on the result
        5.  Update the CollectStats (using `update_at_step_batch`) and the internal counters after the step
        6.  Add the resulting R transitions to the buffer
        7.  Find the D envs that reached done in the current iteration
        8.  Reset the envs that reached done
        9.  Extract episodes for the envs that reached done from the buffer
        10. Perform on-episode-done hook. If it has a return, modify the transitions belonging to the episodes inside the buffer inplace
        11. Update the CollectStats instance with the episodes from 9. by using `update_on_episode_done`
        12. Prepare next step in while loop by saving the last observations and infos
        13. Remove S surplus envs from collection mechanism, thereby reducing R to R-S, to increase performance
        14. Update instance-level collection counters (contrary to counters with a lifetime of the collect execution)
        15. Prepare for the next call of collect (save last observations and info to collector state)

        You can search for Step <n> to find where it happens
        """
        # TODO: can't do it init since AsyncCollector is currently a subclass of Collector
        if self.env.is_async:
            raise ValueError(
                f"Please use {AsyncCollector.__name__} for asynchronous environments. "
                f"Env class: {self.env.__class__.__name__}.",
            )

        if n_step is not None:
            ready_env_ids_R = np.arange(self.env_num)
        elif n_episode is not None:
            if self.env_num > n_episode:
                log.warning(
                    f"Number of episodes ({n_episode}) is smaller than the number of environments "
                    f"({self.env_num}). This means that {self.env_num - n_episode} "
                    f"environments (or, equivalently, parallel workers) will not be used!",
                )
            ready_env_ids_R = np.arange(min(self.env_num, n_episode))
        else:
            raise RuntimeError("Input validation failed, this is a bug and shouldn't have happened")

        if self._pre_collect_obs_RO is None or self._pre_collect_info_R is None:
            raise ValueError(
                "Initial obs and info should not be None. "
                "Either reset the collector (using reset or reset_env) or pass reset_before_collect=True to collect.",
            )

        # Step 0
        # get the first obs to be the current obs in the n_step case as
        # episodes as a new call to collect does not restart trajectories
        # (which we also really don't want)
        step_count = 0
        num_collected_episodes = 0
        episode_returns: list[float] = []
        episode_lens: list[int] = []
        episode_start_indices: list[int] = []

        # Step 1
        collect_stats = self.collect_stats_class()

        # in case we select fewer episodes than envs, we run only some of them
        last_obs_RO = _nullable_slice(self._pre_collect_obs_RO, ready_env_ids_R)
        last_info_R = _nullable_slice(self._pre_collect_info_R, ready_env_ids_R)
        last_hidden_state_RH = _nullable_slice(
            self._pre_collect_hidden_state_RH,
            ready_env_ids_R,
        )

        while True:
            # todo check if we need this when using cur_rollout_batch
            # if len(cur_rollout_batch) != len(ready_env_ids):
            #     raise RuntimeError(
            #         f"The length of the collected_rollout_batch {len(cur_rollout_batch)}) is not equal to the length of ready_env_ids"
            #         f"{len(ready_env_ids)}. This should not happen and could be a bug!",
            #     )
            # restore the state: if the last state is None, it won't store

            # Step 2
            # get the next action and related stats from the previous observation
            collect_action_computation_batch_R = self._compute_action_policy_hidden(
                random=random,
                ready_env_ids_R=ready_env_ids_R,
                last_obs_RO=last_obs_RO,
                last_info_R=last_info_R,
                last_hidden_state_RH=last_hidden_state_RH,
            )

            # Step 3
            obs_next_RO, rew_R, terminated_R, truncated_R, info_R = self.env.step(
                collect_action_computation_batch_R.act_normalized,
                ready_env_ids_R,
            )
            if isinstance(info_R, dict):  # type: ignore[unreachable]
                # This can happen if the env is an envpool env. Then the info returned by step is a dict
                info_R = _dict_of_arr_to_arr_of_dicts(info_R)  # type: ignore[unreachable]
            done_R = np.logical_or(terminated_R, truncated_R)

            current_step_batch_R = cast(
                CollectStepBatchProtocol,
                Batch(
                    obs=last_obs_RO,
                    dist=collect_action_computation_batch_R.dist,
                    act=collect_action_computation_batch_R.act,
                    policy=collect_action_computation_batch_R.policy_entry,
                    obs_next=obs_next_RO,
                    rew=rew_R,
                    terminated=terminated_R,
                    truncated=truncated_R,
                    done=done_R,
                    info=info_R,
                ),
            )

            # TODO: only makes sense if render_mode is human.
            #  Also, doubtful whether it makes sense at all for true vectorized envs
            if render:
                self.env.render()
                if not np.isclose(render, 0):
                    time.sleep(render)

            # Step 4
            self.run_on_step_hook(
                collect_action_computation_batch_R,
                current_step_batch_R,
            )

            # Step 5, collect statistics
            collect_stats.update_at_step_batch(current_step_batch_R)
            num_episodes_done_this_iter = np.sum(done_R)
            num_collected_episodes += num_episodes_done_this_iter
            step_count += len(ready_env_ids_R)

            # Step 6
            # add data into the buffer. Since the buffer is essentially an array, we don't want
            # to add the dist. One should not have arrays of dists but rather a single, batch-wise dist.
            # Tianshou already implements slicing of dists, but we don't yet implement merging multiple
            # dists into one, which would be necessary to make a buffer with dists work properly
            batch_to_add_R = copy(current_step_batch_R)
            batch_to_add_R.pop("dist")
            batch_to_add_R = cast(RolloutBatchProtocol, batch_to_add_R)
            insertion_idx_R, ep_return_R, ep_len_R, ep_start_idx_R = self.buffer.add(
                batch_to_add_R,
                buffer_ids=ready_env_ids_R,
            )

            # preparing for the next iteration
            # obs_next, info and hidden_state will be modified inplace in the code below,
            # so we copy to not affect the data in the buffer
            last_obs_RO = copy(obs_next_RO)
            last_info_R = copy(info_R)
            last_hidden_state_RH = copy(collect_action_computation_batch_R.hidden_state)

            # Preparing last_obs_RO, last_info_R, last_hidden_state_RH for the next while-loop iteration
            # Resetting envs that reached done, or removing some of them from the collection if needed (see below)
            if num_episodes_done_this_iter > 0:
                # TODO: adjust the whole index story, don't use np.where, just slice with boolean arrays
                # D - number of envs that reached done in the rollout above
                # local_idx - see block comment on class level
                # Step 7
                env_done_local_idx_D = np.where(done_R)[0]
                episode_lens_D = ep_len_R[env_done_local_idx_D]
                episode_returns_D = ep_return_R[env_done_local_idx_D]
                episode_start_indices_D = ep_start_idx_R[env_done_local_idx_D]

                episode_lens.extend(episode_lens_D)
                episode_returns.extend(episode_returns_D)
                episode_start_indices.extend(episode_start_indices_D)

                # Step 8
                # now we copy obs_next to obs, but since there might be
                # finished episodes, we have to reset finished envs first.
                gym_reset_kwargs = gym_reset_kwargs or {}
                # The index env_done_idx_D was based on 0, ..., R
                # However, each env has an index in the context of the vectorized env and buffer. So the env 0 being done means
                # that some env of the corresponding "global" index was done. The mapping between "local" index in
                # 0,...,R and this global index is maintained by the ready_env_ids_R array.
                # See the class block comment for more details
                env_done_global_idx_D = ready_env_ids_R[env_done_local_idx_D]
                obs_reset_DO, info_reset_D = self.env.reset(
                    env_id=env_done_global_idx_D,
                    **gym_reset_kwargs,
                )

                # Set the hidden state to zero or None for the envs that reached done
                # TODO: does it have to be so complicated? We should have a single clear type for hidden_state instead of
                #  this complex logic
                self._reset_hidden_state_based_on_type(env_done_local_idx_D, last_hidden_state_RH)

                # Step 9
                # execute episode hooks for those envs which emitted 'done'
                for local_done_idx, cur_ep_return in zip(
                    env_done_local_idx_D,
                    episode_returns_D,
                    strict=True,
                ):
                    # retrieve the episode batch from the buffer using the episode start and stop indices
                    ep_start_idx, ep_stop_idx = (
                        int(ep_start_idx_R[local_done_idx]),
                        int(insertion_idx_R[local_done_idx] + 1),
                    )

                    ep_index_array = self.buffer.get_buffer_indices(ep_start_idx, ep_stop_idx)
                    ep_batch = cast(EpisodeBatchProtocol, self.buffer[ep_index_array])

                    # Step 10
                    episode_hook_additions = self.run_on_episode_done(ep_batch)
                    if episode_hook_additions is not None:
                        if n_episode is None:
                            raise ValueError(
                                "An on_episode_done_hook with non-empty returns is not supported for n_step collection."
                                "Such hooks should only be used when collecting full episodes. Got a on_episode_done_hook "
                                f"that would add the following fields to the buffer: {list(episode_hook_additions)}.",
                            )

                        for key, episode_addition in episode_hook_additions.items():
                            self.buffer.set_array_at_key(
                                episode_addition,
                                key,
                                index=ep_index_array,
                            )
                            # executing the same logic in the episode-batch since stats computation
                            # may depend on the presence of additional fields
                            ep_batch.set_array_at_key(
                                episode_addition,
                                key,
                            )
                    # Step 11
                    # Finally, update the stats
                    collect_stats.update_at_episode_done(
                        episode_batch=ep_batch,
                        episode_return=cur_ep_return,
                    )

                # Step 12
                # preparing for the next iteration
                last_obs_RO[env_done_local_idx_D] = obs_reset_DO
                last_info_R[env_done_local_idx_D] = info_reset_D

                # Step 13
                # Handling the case when we have more ready envs than desired and are not done yet
                #
                # This can only happen if we are collecting a fixed number of episodes
                # If we have more ready envs than there are remaining episodes to collect,
                # we will remove some of them for the next rollout
                # One effect of this is the following: only envs that have completed an episode
                # in the last step can ever be removed from the ready envs.
                # Thus, this guarantees that each env will contribute at least one episode to the
                # collected data (the buffer). This effect was previous called "avoiding bias in selecting environments"
                # However, it is not at all clear whether this is actually useful or necessary.
                # Additional naming convention:
                # S - number of surplus envs
                # TODO: can the whole block be removed? If we have too many episodes, we could just strip the last ones.
                #   Changing R to R-S highly increases the complexity of the code.
                if n_episode:
                    remaining_episodes_to_collect = n_episode - num_collected_episodes
                    surplus_env_num = len(ready_env_ids_R) - remaining_episodes_to_collect
                    if surplus_env_num > 0:
                        # R becomes R-S here, preparing for the next iteration in while loop
                        # Everything that was of length R needs to be filtered and become of length R-S.
                        # Note that this won't be the last iteration, as one iteration equals one
                        # step and we still need to collect the remaining episodes to reach the breaking condition.

                        # creating the mask
                        env_to_be_ignored_ind_local_S = env_done_local_idx_D[:surplus_env_num]
                        env_should_remain_R = np.ones_like(ready_env_ids_R, dtype=bool)
                        env_should_remain_R[env_to_be_ignored_ind_local_S] = False
                        # stripping the "idle" indices, shortening the relevant quantities from R to R-S
                        ready_env_ids_R = ready_env_ids_R[env_should_remain_R]
                        last_obs_RO = last_obs_RO[env_should_remain_R]
                        last_info_R = last_info_R[env_should_remain_R]
                        if collect_action_computation_batch_R.hidden_state is not None:
                            last_hidden_state_RH = last_hidden_state_RH[env_should_remain_R]  # type: ignore[index]

            if (n_step and step_count >= n_step) or (
                n_episode and num_collected_episodes >= n_episode
            ):
                break

        # Check if we screwed up somewhere
        if self.buffer.hasnull():
            nan_batch = self.buffer.isnull().apply_values_transform(np.sum)

            raise MalformedBufferError(
                "NaN detected in the buffer. You can drop them with `buffer.dropnull()`. "
                "This error is most often caused by an incorrect use of `EpisodeRolloutHooks`"
                "together with the `n_steps` (instead of `n_episodes`) option, or by "
                "an incorrect implementation of `StepHook`."
                "Here an overview of the number of NaNs per field: \n"
                f"{nan_batch}",
            )

        # Step 14
        # update instance-lifetime counters, different from collect_stats
        self.collect_step += step_count
        self.collect_episode += num_collected_episodes

        # Step 15
        if n_step:
            # persist for future collect iterations
            self._pre_collect_obs_RO = last_obs_RO
            self._pre_collect_info_R = last_info_R
            self._pre_collect_hidden_state_RH = last_hidden_state_RH
        elif n_episode:
            # reset envs and the _pre_collect fields
            self.reset_env(gym_reset_kwargs)  # todo still necessary?
        return collect_stats

    @staticmethod
    def _reset_hidden_state_based_on_type(
        env_ind_local_D: np.ndarray,
        last_hidden_state_RH: np.ndarray | torch.Tensor | Batch | None,
    ) -> None:
        if isinstance(last_hidden_state_RH, torch.Tensor):
            last_hidden_state_RH[env_ind_local_D].zero_()  # type: ignore[index]
        elif isinstance(last_hidden_state_RH, np.ndarray):
            last_hidden_state_RH[env_ind_local_D] = (
                None if last_hidden_state_RH.dtype == object else 0
            )
        elif isinstance(last_hidden_state_RH, Batch):
            last_hidden_state_RH.empty_(env_ind_local_D)
        # todo is this inplace magic and just working?


class AsyncCollector(Collector[CollectStats]):
    """Async Collector handles async vector environment.

    Please refer to :class:`~tianshou.data.Collector` for a more detailed explanation.
    """

    def __init__(
        self,
        policy: BasePolicy,
        env: BaseVectorEnv,
        buffer: ReplayBuffer | None = None,
        exploration_noise: bool = False,
        raise_on_nan_in_buffer: bool = True,
    ) -> None:
        if not env.is_async:
            # TODO: raise an exception?
            log.error(
                f"Please use {Collector.__name__} if not using async venv. "
                f"Env class: {env.__class__.__name__}",
            )
        # assert env.is_async
        warnings.warn("Using async setting may collect extra transitions into buffer.")
        super().__init__(
            policy,
            env,
            buffer,
            exploration_noise,
            collect_stats_class=CollectStats,
            raise_on_nan_in_buffer=raise_on_nan_in_buffer,
        )
        # E denotes the number of parallel environments: self.env_num
        # At init, E=R but during collection R <= E
        # Keep in sync with reset!
        self._ready_env_ids_R: np.ndarray = np.arange(self.env_num)
        self._current_obs_in_all_envs_EO: np.ndarray | None = copy(self._pre_collect_obs_RO)
        self._current_info_in_all_envs_E: np.ndarray | None = copy(self._pre_collect_info_R)
        self._current_hidden_state_in_all_envs_EH: np.ndarray | torch.Tensor | Batch | None = copy(
            self._pre_collect_hidden_state_RH,
        )
        self._current_action_in_all_envs_EA: np.ndarray = np.empty(self.env_num)
        self._current_policy_in_all_envs_E: Batch | None = None

    def reset(
        self,
        reset_buffer: bool = True,
        reset_stats: bool = True,
        gym_reset_kwargs: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reset the environment, statistics, and data needed to start the collection.

        :param reset_buffer: if true, reset the replay buffer attached
            to the collector.
        :param reset_stats: if true, reset the statistics attached to the collector.
        :param gym_reset_kwargs: extra keyword arguments to pass into the environment's
            reset function. Defaults to None (extra keyword arguments)
        :return: The initial observation and info from the environment.
        """
        # This sets the _pre_collect attrs
        result = super().reset(
            reset_buffer=reset_buffer,
            reset_stats=reset_stats,
            gym_reset_kwargs=gym_reset_kwargs,
        )
        # Keep in sync with init!
        self._ready_env_ids_R = np.arange(self.env_num)
        # E denotes the number of parallel environments self.env_num
        self._current_obs_in_all_envs_EO = copy(self._pre_collect_obs_RO)
        self._current_info_in_all_envs_E = copy(self._pre_collect_info_R)
        self._current_hidden_state_in_all_envs_EH = copy(self._pre_collect_hidden_state_RH)
        self._current_action_in_all_envs_EA = np.empty(self.env_num)
        self._current_policy_in_all_envs_E = None
        return result

    @override
    def reset_env(
        self,
        gym_reset_kwargs: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # we need to step through the envs and wait until they are ready to be able to interact with them
        if self.env.waiting_id:
            self.env.step(None, id=self.env.waiting_id)
        return super().reset_env(gym_reset_kwargs=gym_reset_kwargs)

    @override
    def _collect(
        self,
        n_step: int | None = None,
        n_episode: int | None = None,
        random: bool = False,
        render: float | None = None,
        gym_reset_kwargs: dict[str, Any] | None = None,
    ) -> CollectStats:
        start_time = time.time()

        step_count = 0
        num_collected_episodes = 0
        episode_returns: list[float] = []
        episode_lens: list[int] = []
        episode_start_indices: list[int] = []

        ready_env_ids_R = self._ready_env_ids_R
        # last_obs_RO= self._current_obs_in_all_envs_EO[ready_env_ids_R] # type: ignore[index]
        # last_info_R = self._current_info_in_all_envs_E[ready_env_ids_R] # type: ignore[index]
        # last_hidden_state_RH = self._current_hidden_state_in_all_envs_EH[ready_env_ids_R] # type: ignore[index]
        # last_obs_RO = self._pre_collect_obs_RO
        # last_info_R = self._pre_collect_info_R
        # last_hidden_state_RH = self._pre_collect_hidden_state_RH
        if self._current_obs_in_all_envs_EO is None or self._current_info_in_all_envs_E is None:
            raise RuntimeError(
                "Current obs or info array is None, did you call reset or pass reset_at_collect=True?",
            )

        last_obs_RO = self._current_obs_in_all_envs_EO[ready_env_ids_R]
        last_info_R = self._current_info_in_all_envs_E[ready_env_ids_R]
        last_hidden_state_RH = _nullable_slice(
            self._current_hidden_state_in_all_envs_EH,
            ready_env_ids_R,
        )
        # Each iteration of the AsyncCollector is only stepping a subset of the
        # envs. The last observation/ hidden state of the ones not included in
        # the current iteration has to be retained. This is done by copying the
        while True:
            # todo do we need this?
            # todo extend to all current attributes but some could be None at init
            if self._current_obs_in_all_envs_EO is None:
                raise RuntimeError(
                    "Current obs is None, did you call reset or pass reset_at_collect=True?",
                )
            if (
                not len(self._current_obs_in_all_envs_EO)
                == len(self._current_action_in_all_envs_EA)
                == self.env_num
            ):  # major difference
                raise RuntimeError(
                    f"{len(self._current_obs_in_all_envs_EO)=} and"
                    f"{len(self._current_action_in_all_envs_EA)=} have to equal"
                    f" {self.env_num=} as it tracks the current transition"
                    f"in all envs",
                )

            # get the next action
            collect_batch_R = self._compute_action_policy_hidden(
                random=random,
                ready_env_ids_R=ready_env_ids_R,
                last_obs_RO=last_obs_RO,
                last_info_R=last_info_R,
                last_hidden_state_RH=last_hidden_state_RH,
            )

            # save act_RA/policy_R/ hidden_state_RH before env.step
            self._current_action_in_all_envs_EA[ready_env_ids_R] = collect_batch_R.act
            if self._current_policy_in_all_envs_E:
                self._current_policy_in_all_envs_E[ready_env_ids_R] = collect_batch_R.policy_entry
            else:
                self._current_policy_in_all_envs_E = collect_batch_R.policy_entry  # first iteration
            if collect_batch_R.hidden_state is not None:
                if self._current_hidden_state_in_all_envs_EH is not None:
                    # Need to cast since if it's a Tensor, the assignment might in fact fail if hidden_state_RH is not
                    # a tensor as well. This is hard to express with proper typing, even using @overload, so we cheat
                    # and hope that if one of the two is a tensor, the other one is as well.
                    self._current_hidden_state_in_all_envs_EH = cast(
                        np.ndarray | Batch,
                        self._current_hidden_state_in_all_envs_EH,
                    )
                    self._current_hidden_state_in_all_envs_EH[
                        ready_env_ids_R
                    ] = collect_batch_R.hidden_state
                else:
                    self._current_hidden_state_in_all_envs_EH = collect_batch_R.hidden_state

            # step in env
            obs_next_RO, rew_R, terminated_R, truncated_R, info_R = self.env.step(
                collect_batch_R.act_normalized,
                ready_env_ids_R,
            )
            done_R = np.logical_or(terminated_R, truncated_R)
            # Not all environments of the AsyncCollector might have performed a step in this iteration.
            # Change batch_of_envs_with_step_in_this_iteration here to reflect that ready_env_ids_R has changed.
            # This means especially that R is potentially changing every iteration
            try:
                ready_env_ids_R = cast(np.ndarray, info_R["env_id"])
            # TODO: don't use bare Exception!
            except Exception:
                ready_env_ids_R = np.array([i["env_id"] for i in info_R])

            current_iteration_batch = cast(
                RolloutBatchProtocol,
                Batch(
                    obs=self._current_obs_in_all_envs_EO[ready_env_ids_R],
                    act=self._current_action_in_all_envs_EA[ready_env_ids_R],
                    policy=self._current_policy_in_all_envs_E[ready_env_ids_R],
                    obs_next=obs_next_RO,
                    rew=rew_R,
                    terminated=terminated_R,
                    truncated=truncated_R,
                    done=done_R,
                    info=info_R,
                ),
            )

            if render:
                self.env.render()
                if render > 0 and not np.isclose(render, 0):
                    time.sleep(render)

            # add data into the buffer
            ptr_R, ep_rew_R, ep_len_R, ep_idx_R = self.buffer.add(
                current_iteration_batch,
                buffer_ids=ready_env_ids_R,
            )

            # collect statistics
            num_episodes_done_this_iter = np.sum(done_R)
            step_count += len(ready_env_ids_R)
            num_collected_episodes += num_episodes_done_this_iter

            # preparing for the next iteration
            # todo seem we can get rid of this last_sth stuff altogether
            last_obs_RO = copy(obs_next_RO)
            last_info_R = copy(info_R)
            last_hidden_state_RH = copy(
                self._current_hidden_state_in_all_envs_EH[ready_env_ids_R],  # type: ignore[index]
            )
            if num_episodes_done_this_iter:
                env_ind_local_D = np.where(done_R)[0]
                env_ind_global_D = ready_env_ids_R[env_ind_local_D]
                episode_lens.extend(ep_len_R[env_ind_local_D])
                episode_returns.extend(ep_rew_R[env_ind_local_D])
                episode_start_indices.extend(ep_idx_R[env_ind_local_D])

                # now we copy obs_next_RO to obs, but since there might be
                # finished episodes, we have to reset finished envs first.
                gym_reset_kwargs = gym_reset_kwargs or {}
                obs_reset_DO, info_reset_D = self.env.reset(
                    env_id=env_ind_global_D,
                    **gym_reset_kwargs,
                )
                last_obs_RO[env_ind_local_D] = obs_reset_DO
                last_info_R[env_ind_local_D] = info_reset_D

                self._reset_hidden_state_based_on_type(env_ind_local_D, last_hidden_state_RH)

            # update based on the current transition in all envs
            self._current_obs_in_all_envs_EO[ready_env_ids_R] = last_obs_RO
            # this is a list, so loop over
            for idx, ready_env_id in enumerate(ready_env_ids_R):
                self._current_info_in_all_envs_E[ready_env_id] = last_info_R[idx]
            if self._current_hidden_state_in_all_envs_EH is not None:
                # Need to cast since if it's a Tensor, the assignment might in fact fail if hidden_state_RH is not
                # a tensor as well. This is hard to express with proper typing, even using @overload, so we cheat
                # and hope that if one of the two is a tensor, the other one is as well.
                self._current_hidden_state_in_all_envs_EH = cast(
                    np.ndarray | Batch,
                    self._current_hidden_state_in_all_envs_EH,
                )
                self._current_hidden_state_in_all_envs_EH[ready_env_ids_R] = last_hidden_state_RH
            else:
                self._current_hidden_state_in_all_envs_EH = last_hidden_state_RH

            if (n_step and step_count >= n_step) or (
                n_episode and num_collected_episodes >= n_episode
            ):
                break

        # generate statistics
        self.collect_step += step_count
        self.collect_episode += num_collected_episodes
        collect_time = max(time.time() - start_time, 1e-9)
        self.collect_time += collect_time

        # persist for future collect iterations
        self._ready_env_ids_R = ready_env_ids_R

        return CollectStats.with_autogenerated_stats(
            returns=np.array(episode_returns),
            lens=np.array(episode_lens),
            n_collected_episodes=num_collected_episodes,
            n_collected_steps=step_count,
        )


class StepHookProtocol(Protocol):
    """A protocol for step hooks."""

    def __call__(
        self,
        action_batch: CollectActionBatchProtocol,
        rollout_batch: RolloutBatchProtocol,
    ) -> None:
        """The function to call when the hook is executed."""
        ...


class StepHook(StepHookProtocol, ABC):
    """Marker interface for step hooks.

    All step hooks in Tianshou will inherit from it, but only the corresponding protocol will be
    used in type hints. This makes it possible to discover all hooks in the codebase by looking up
    the hierarchy of this class (or the protocol itself) while still allowing the user to pass
    something like a lambda function as a hook.
    """

    @abstractmethod
    def __call__(
        self,
        action_batch: CollectActionBatchProtocol,
        rollout_batch: RolloutBatchProtocol,
    ) -> None:
        ...


class StepHookAddActionDistribution(StepHook):
    """Adds the action distribution to the collected rollout batch under the field "action_dist".

    The field is also accessible as class variable `ACTION_DIST_KEY`.
    This hook be useful for algorithms that need the previously taken actions for training, like variants of
    imitation learning or DAGGER.
    """

    ACTION_DIST_KEY = "action_dist"

    def __call__(
        self,
        action_batch: CollectActionBatchProtocol,
        rollout_batch: RolloutBatchProtocol,
    ) -> None:
        rollout_batch[self.ACTION_DIST_KEY] = action_batch.dist


class EpisodeRolloutHookProtocol(Protocol):
    """A protocol for hooks (functions) that act on an entire collected episode.

    Can be used to add values to the buffer that are only known after the episode is finished.
    A prime example is something like the MC return to go.
    """

    def __call__(self, episode_batch: EpisodeBatchProtocol) -> dict[str, np.ndarray] | None:
        """Will be called by the collector when an episode is finished.

        If a dictionary is returned, the key-value pairs will be interpreted as new entries
        to be added to the episode batch (inside the buffer). In that case,
        the values should be arrays of the same length as the input `rollout_batch`.

        :param episode_batch: the batch of transitions that belong to the episode.
        :return: an optional dictionary containing new entries (of same len as `rollout_batch`)
            to be added to the buffer.
        """
        ...


class EpisodeRolloutHook(EpisodeRolloutHookProtocol, ABC):
    """Marker interface for episode hooks.

    All episode hooks in Tianshou will inherit from it, but only the corresponding protocol will be
    used in type hints. This makes it possible to discover all hooks in the codebase by looking up
    the hierarchy of this class (or the protocol itself) while still allowing the user to pass
    something like a lambda function as a hook.
    """

    @abstractmethod
    def __call__(self, episode_batch: EpisodeBatchProtocol) -> dict[str, np.ndarray] | None:
        ...


class EpisodeRolloutHookMCReturn(EpisodeRolloutHook):
    """Adds the MC return to go as well as the full episode MC return to the transitions in the buffer.

    The latter will be constant for all transitions in the same episode and simply corresponds to
    the initial MC return to go. Useful for algorithms that rely on the monte carlo returns during training.
    """

    MC_RETURN_TO_GO_KEY = "mc_return_to_go"
    FULL_EPISODE_MC_RETURN_KEY = "full_episode_mc_return"

    class OutputDict(TypedDict):
        mc_return_to_go: np.ndarray
        full_episode_mc_return: np.ndarray

    def __init__(self, gamma: float = 0.99):
        if not 0 <= gamma <= 1:
            raise ValueError(f"Expected 0 <= gamma <= 1, but got {gamma=}.")
        self.gamma = gamma

    def __call__(  # type: ignore[override]
        self,
        episode_batch: RolloutBatchProtocol,
    ) -> "EpisodeRolloutHookMCReturn.OutputDict":
        mc_return_to_go = episode_mc_return_to_go(episode_batch.rew, self.gamma)
        full_episode_mc_return = np.full_like(mc_return_to_go, mc_return_to_go[0])

        return self.OutputDict(
            mc_return_to_go=mc_return_to_go,
            full_episode_mc_return=full_episode_mc_return,
        )


class EpisodeRolloutHookMerged(EpisodeRolloutHook):
    """Combines multiple episode hooks into a single one.

    If all hooks return `None`, this hook will also return `None`.
    """

    def __init__(
        self,
        *episode_rollout_hooks: EpisodeRolloutHookProtocol,
        check_overlapping_keys: bool = True,
    ):
        """
        :param episode_rollout_hooks: the hooks to combine
        :param check_overlapping_keys: whether to check for overlapping keys in the output of the hooks and
            raise a `KeyError` if any are found. Set to `False` to disable this check (can be useful
            if this becomes a performance bottleneck).
        """
        self.episode_rollout_hooks = episode_rollout_hooks
        self.check_overlapping_keys = check_overlapping_keys

    def __call__(self, episode_batch: EpisodeBatchProtocol) -> dict[str, np.ndarray] | None:
        result: dict[str, np.ndarray] = {}
        for rollout_hook in self.episode_rollout_hooks:
            new_entries = rollout_hook(episode_batch)
            if new_entries is None:
                continue

            if self.check_overlapping_keys and (
                duplicated_entries := set(new_entries).difference(result)
            ):
                raise KeyError(
                    f"Combined rollout hook {rollout_hook} leads to previously "
                    f"computed entries that would be overwritten: {duplicated_entries=}. "
                    f"Consider combining hooks which will deliver non-overlapping entries to solve this.",
                )
            result.update(new_entries)
        if not result:
            return None
        return result
