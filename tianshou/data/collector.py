import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Self, cast

import gymnasium as gym
import numpy as np
import torch
from overrides import override

from tianshou.data import (
    Batch,
    CachedReplayBuffer,
    ReplayBuffer,
    ReplayBufferManager,
    SequenceSummaryStats,
    VectorReplayBuffer,
    to_numpy,
)
from tianshou.data.batch import alloc_by_keys_diff
from tianshou.data.types import RolloutBatchProtocol
from tianshou.env import BaseVectorEnv, DummyVectorEnv
from tianshou.policy import BasePolicy


def get_empty_rollout_batch() -> RolloutBatchProtocol:
    """Empty batch, useful for adding new data."""
    result = Batch(
        obs={},
        act={},
        rew={},
        terminated={},
        truncated={},
        done={},
        obs_next={},
        info={},
        policy={},
    )
    return cast(RolloutBatchProtocol, result)


@dataclass(kw_only=True)
class CollectStatsBase:
    """The most basic stats, often used for offline learning."""

    n_collected_episodes: int = 0
    """The number of collected episodes."""
    n_collected_steps: int = 0
    """The number of collected steps."""


@dataclass(kw_only=True)
class CollectStats(CollectStatsBase):
    """A data structure for storing the statistics of rollouts."""

    collect_time: float = 0.0
    """The time for collecting transitions."""
    collect_speed: float = 0.0
    """The speed of collecting (env_step per second)."""
    returns: np.ndarray = field(default_factory=lambda: np.empty(0))
    """The collected episode returns."""
    returns_stat: SequenceSummaryStats | None = None  # can be None if no episode ends while collecting n_step transitions across all workers
    """Stats of the collected returns."""
    lens: np.ndarray = field(default_factory=lambda: np.empty(0))
    """The collected episode lengths."""
    lens_stat: SequenceSummaryStats | None = None  # can be None if no episode ends while collecting n_step transitions across all workers
    """Stats of the collected episode lengths."""

    @classmethod
    def from_collect_output(
        cls,
        episode_count: int,
        step_count: int,
        collect_call_duration: float,
        episode_returns: np.ndarray,
        episode_lens: np.ndarray,
    ) -> Self:
        """Instantiate from variables inside the scope of collector.collect()."""
        return cls(
            n_collected_episodes=episode_count,
            n_collected_steps=step_count,
            collect_time=collect_call_duration,
            collect_speed=step_count / collect_call_duration,
            returns=np.array(episode_returns),
            returns_stat=SequenceSummaryStats.from_sequence(episode_returns)
            if len(episode_returns) > 0
            else None,
            lens=np.array(episode_lens, int),
            lens_stat=SequenceSummaryStats.from_sequence(episode_lens)
            if len(episode_lens) > 0
            else None,
        )


class Collector:
    """Collector enables the policy to interact with different types of envs with exact number of steps or episodes.

    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy` class.
    :param env: a ``gym.Env`` environment or an instance of the
        :class:`~tianshou.env.BaseVectorEnv` class. If the environment is not a VectorEnv,
        it will be converted to a :class:`~tianshou.env.DummyVectorEnv`.
    :param buffer: an instance of the :class:`~tianshou.data.ReplayBuffer` class.
        If set to None, it will not store the data.
    :param function preprocess_fn: a function called before the data has been added to
        the buffer, see issue #42 and :ref:`preprocess_fn`.
    :param exploration_noise: determine whether the action needs to be modified
         with the corresponding policy's exploration noise. If so, "policy.
        exploration_noise(act, batch)" will be called automatically to add the
        exploration noise into action.

    The "preprocess_fn" is a function called before the data has been added to the
    buffer with batch format. It will receive only "obs" and "env_id" when the
    collector resets the environment, and will receive the keys "obs_next", "rew",
    "terminated", "truncated, "info", "policy" and "env_id" in a normal env step.
    Alternatively, it may also accept the keys "obs_next", "rew", "done", "info",
    "policy" and "env_id".
    It returns either a dict or a :class:`~tianshou.data.Batch` with the modified
    keys and values. Examples are in "test/base/test_collector.py".

    .. note::

        Please make sure the given environment has a time limitation if using n_episode
        collect option.

    .. note::

        In past versions of Tianshou, the replay buffer passed to `__init__`
        was automatically reset. This is not done in the current implementation.
    """

    def __init__(
        self,
        policy: BasePolicy,
        env: gym.Env | BaseVectorEnv,
        buffer: ReplayBuffer | None = None,
        preprocess_fn: Callable[..., RolloutBatchProtocol] | None = None,
        exploration_noise: bool = False,
    ) -> None:
        self.env: BaseVectorEnv  # for mypy
        if not isinstance(env, BaseVectorEnv):
            self.env = DummyVectorEnv([lambda: env])
        else:
            self.env = env
        self._validate_env()

        self.exploration_noise = exploration_noise
        self.buffer: ReplayBuffer
        self._assign_buffer(buffer)
        self.policy = policy
        self.preprocess_fn = preprocess_fn
        self._action_space = self.env.action_space

        # Keep default values in sync with the functionality of self.reset!
        # We shouldn't instantiate the fields to None and then call reset in init, since
        # this makes mypy think that the fields can be None...
        self.data = get_empty_rollout_batch()
        self.collect_step, self.collect_episode, self.collect_time = 0, 0, 0.0
        self.reset_env()  # resets envs and sets info and obs attributes in self.data
        self.reset(False)

    @property
    def env_num(self) -> int:
        """Return the number of environments."""
        return len(self.env)

    def get_collect_speed(self) -> float:
        if self.collect_time == 0:
            return 0.0
        return self.collect_step / self.collect_time

    def _validate_env(self) -> None:
        if self.env.is_async:
            raise ValueError(f"Please use {AsyncCollector.__name__} for using async venvs.")

    def _assign_buffer(self, buffer: ReplayBuffer | None) -> None:
        """Check if the buffer matches the constraint."""
        if buffer is None:
            buffer = VectorReplayBuffer(self.env_num, self.env_num)
        # TODO: refactor the code such that these isinstance checks are not necessary
        elif isinstance(buffer, ReplayBufferManager):
            assert buffer.buffer_num >= self.env_num
            if isinstance(buffer, CachedReplayBuffer):
                assert buffer.cached_buffer_num >= self.env_num
        else:
            assert buffer.maxsize > 0
            if self.env_num > 1:
                raise TypeError(
                    f"Cannot use {buffer.__class__.__name__}(size={buffer.maxsize}, ...) to collect "
                    f"{self.env_num} envs, please use a corresponding vectorized buffer instead.",
                )
        self.buffer = buffer

    def reset(
        self,
        reset_buffer: bool = True,
        gym_reset_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Reset the environment, counters, data and, if desired, the buffer.

        :param reset_buffer: if true, reset the replay buffer that is attached
            to the collector.
        :param gym_reset_kwargs: extra keyword arguments to pass into the environment's
            reset function.
        """
        self._reset_data()
        self.reset_env(gym_reset_kwargs)
        if reset_buffer:
            self.reset_buffer()
        self.reset_counters()

    def _reset_data(self) -> None:
        self.data = get_empty_rollout_batch()

    def reset_counters(self) -> None:
        """Reset the collection counting fields."""
        self.collect_step, self.collect_episode, self.collect_time = 0, 0, 0.0

    def reset_buffer(self, keep_statistics: bool = False) -> None:
        """Reset the data buffer."""
        self.buffer.reset(keep_statistics=keep_statistics)

    def reset_env(
        self,
        gym_reset_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Any, dict]:
        """Reset all environments and set `info` and `obs` attributes in `self.data`."""
        gym_reset_kwargs = gym_reset_kwargs if gym_reset_kwargs else {}
        obs, info = self.env.reset(**gym_reset_kwargs)
        if self.preprocess_fn:
            processed_data = self.preprocess_fn(obs=obs, info=info, env_id=np.arange(self.env_num))
            obs = processed_data.get("obs", obs)
            info = processed_data.get("info", info)
        return obs, info

    @staticmethod
    def _reset_state(collected_rollout_batch, id: int | list[int]) -> None:
        """Reset the hidden state: self.data.state[id]."""
        if hasattr(collected_rollout_batch.policy, "hidden_state"):
            state = collected_rollout_batch.policy.hidden_state  # it is a reference
            if isinstance(state, torch.Tensor):
                state[id].zero_()
            elif isinstance(state, np.ndarray):
                state[id] = None if state.dtype == object else 0
            elif isinstance(state, Batch):
                state.empty_(id)

    def _reset_env_with_ids(
        self,
        collected_rollout_batch: RolloutBatchProtocol,
        local_ids: list[int] | np.ndarray,
        global_ids: list[int] | np.ndarray,
        gym_reset_kwargs: dict[str, Any] | None = None,
    ) -> None:
        gym_reset_kwargs = gym_reset_kwargs if gym_reset_kwargs else {}
        obs_reset, info = self.env.reset(global_ids, **gym_reset_kwargs)
        if self.preprocess_fn:
            processed_data = self.preprocess_fn(obs=obs_reset, info=info, env_id=global_ids)
            obs_reset = processed_data.get("obs", obs_reset)
            info = processed_data.get("info", info)
        collected_rollout_batch.info[local_ids] = info  # type: ignore

        collected_rollout_batch.obs_next[local_ids] = obs_reset  # type: ignore

    def _reset_env_with_obs_next_to_obs(
        self,
        gym_reset_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._reset_data()
        self.reset_env(gym_reset_kwargs, set_obs_next_to_obs=True)

    @staticmethod
    def _is_collection_finished(
        n_step: int | None,
        n_episode: int | None,
        current_n_step: int,
        current_n_episode: int,
    ) -> bool:
        """:param n_step: number of steps to take
        :param n_episode: number of episodes to take
        """
        if n_step is None and n_episode is None:
            raise RuntimeError("This should not have happened!")
        if n_step:
            return current_n_step >= n_step
        else:
            return current_n_episode >= n_episode


    def collect(
        self,
        n_step: int | None = None,
        n_episode: int | None = None,
        random: bool = False,
        render: float | None = None,
        no_grad: bool = True,
        gym_reset_kwargs: dict[str, Any] | None = None,
        sample_equal_num_episodes_per_worker: bool = False,
    ) -> CollectStats:
        """Collect a specified number of steps or episode.

        TODO: adjust wording
        To ensure unbiased sampling result with `n_episode` option, this function will
        first collect ``n_episode - env_num`` episodes, then for the last ``env_num``
        episodes, they will be collected evenly from each env.

        :param n_step: how many steps you want to collect.
            Either this or `n_episode` has to be provided.
        :param n_episode: how many episodes you want to collect.
            Either this or `n_step` has to be provided.
        :param random: whether to use random policy for collecting data.
        :param render: the sleep time between rendering consecutive frames.
        :param no_grad: whether to retain gradient in policy.forward().
        :param gym_reset_kwargs: extra keyword arguments to pass into the environment's
            reset function.
        :param sample_equal_num_episodes_per_worker: only used if `n_episode` is specified.
            Whether to sample equal number of episodes
            from each parallel rollout. Otherwise, it is only ensured that at least one episode
            is collected from every env when using `n_episode`.
        """
        self._validate_collect_input(n_episode, n_step, sample_equal_num_episodes_per_worker)

        non_idle_env_ids = np.arange(self.env_num)
        start_time = time.time()

        # collect_call_output = CollectCallOutput()

        step_count = 0
        episode_count = 0
        episode_returns: list[float] = []
        episode_lens: list[int] = []

        cur_rollout_batch = get_empty_rollout_batch()

        # collect can be called multiple times, so the total number of steps/episodes needed for termination
        # is dependent on the current values of the collect_step and collect_episode fields

        while not self._is_collection_finished(
            n_step=n_step,
            n_episode=n_episode,
            current_n_step=step_count,
            current_n_episode=episode_count,
        ):
            if len(cur_rollout_batch) != len(non_idle_env_ids):
                raise RuntimeError(
                    f"The length of the collected_rollout_batch {len(cur_rollout_batch)}) is not equal to the length of non_idle_env_ids"
                    f"{len(non_idle_env_ids)}. This should not happen and could be a bug!",
                )

            # TODO: reduce code duplication with AsyncCollector
            # restore the state: if the last state is None, it won't store
            last_state = cur_rollout_batch.policy.pop("hidden_state", None)

            # Modifies the collected rollout batch
            self.compute_next_action_in_active_workers(
                cur_rollout_batch,
                last_state,
                no_grad,
                random,
                non_idle_env_ids,
            )

            # get bounded and remapped actions first (not saved into buffer)
            action_remap = self.policy.map_action(cur_rollout_batch.act)

            # of len non_idle_env_ids
            # modifies collected_rollout_batch
            is_done_and_non_idle = self.step_env_and_update_data(cur_rollout_batch, action_remap, non_idle_env_ids)

            if render:
                self.env.render()
                if render > 0 and not np.isclose(render, 0):
                    time.sleep(render)

            # of len non_idle_env_ids
            ep_add_at_idx, ep_rew, ep_len, ep_start_idx = self.buffer.add(
                cur_rollout_batch,
                buffer_ids=non_idle_env_ids,
            )

            step_count += len(non_idle_env_ids)

            # handle finished episodes
            if np.any(is_done_and_non_idle):
                non_idle_env_ids_done = non_idle_env_ids[is_done_and_non_idle]

                episode_count += len(non_idle_env_ids_done)
                episode_lens.extend(ep_len[is_done_and_non_idle])
                episode_returns.extend(ep_rew[is_done_and_non_idle])
                # for idx, ret in zip(env_ind_global, ep_rew[env_indices_at_done_local], strict=True):
                #     self.episode_returns_per_env[idx].append(ret)

                env_ids_done = np.where(is_done_and_non_idle)[0]
                if not sample_equal_num_episodes_per_worker:
                    self._reset_env_with_ids(
                        cur_rollout_batch,
                        env_ids_done,
                        non_idle_env_ids_done,
                        gym_reset_kwargs,
                    )
                    for i in env_ids_done:
                        self._reset_state(cur_rollout_batch, i)

                if n_episode:
                    non_idle_env_ids = self._retrieve_non_idle_env_ids_on_done()
                    # todo check if async collector is equivalent to just passing here

            # Move observations one step into the future such that we can compute the next actions
            cur_rollout_batch.obs = cur_rollout_batch.obs_next

        collect_call_duration = max(time.time() - start_time, 1e-9)
        self._update_collection_counters(collect_call_duration, episode_count, step_count)

        # TODO: investigate!!
        if n_episode:
            self.reset_env()

        return CollectStats.from_collect_output(
            step_count=step_count,
            episode_count=episode_count,
            collect_call_duration=collect_call_duration,
            episode_returns=np.array(episode_returns),
            episode_lens=np.array(episode_lens, int),
        )

    # env_ids_non_idle = [25, 12, 23, 13]
    # non_idle_and_done = [True, False, False, True]
    # env_array_ids_non_idle_not_done = [1, 2]
    # env_ids_non_idle_not_done = [12, 23]

    def _retrieve_non_idle_env_ids_on_done(self,
                                           env_ids_non_idle: np.ndarray,
                                           sample_equal_num_episodes_per_worker: bool,
                                           gym_reset_kwargs: dict,
                                           non_idle_and_done: np.ndarray):
        if sample_equal_num_episodes_per_worker:
            env_array_ids_non_idle_not_done = np.where(~non_idle_and_done)[0]
            env_ids_non_idle_not_done = env_ids_non_idle[env_array_ids_non_idle_not_done]
            self.data = self.data[env_array_ids_non_idle_not_done]
            if len(env_array_ids_non_idle_not_done) == 0:
                env_ids_non_idle_not_done = np.arange(self.env_num)
                self._reset_env_with_obs_next_to_obs(gym_reset_kwargs)
                for i in env_ids_non_idle_not_done:
                    self._reset_state(i)
            return env_ids_non_idle_not_done

            # return self.sample_equal_episodes_per_worker_postprocessing_on_done_env(
            #         gym_reset_kwargs,
            #         non_idle_env_ids,
            #         np.where(~is_done_and_non_idle)[0],
            #     )
        else:
            return self.sample_at_least_one_episode_per_worker_postprocessing_on_done_env(
                    episode_count,
                    env_ids_done,
                    n_episode,
                    env_ids_non_idle,
                )


    def sample_at_least_one_episode_per_worker_postprocessing_on_done_env(
        self,
        episode_count: int,
        env_ind_local: np.ndarray,
        n_episode: int,
        ready_env_ids: np.ndarray,
    ) -> np.ndarray:
        """Remove surplus env id from ready_env_ids to record at least one episode per worker."""
        surplus_env_num = len(ready_env_ids) - (n_episode - episode_count)
        if surplus_env_num > 0:
            mask = np.ones_like(ready_env_ids, dtype=bool)
            mask[env_ind_local[:surplus_env_num]] = False
            ready_env_ids = ready_env_ids[mask]
            self.data = self.data[mask]
        return ready_env_ids

    def sample_equal_episodes_per_worker_postprocessing_on_done_env(
        self,
        gym_reset_kwargs: dict[str, Any] | None,
        ready_env_ids: np.ndarray,
        unfinished_ind_local: np.ndarray,
    ) -> np.ndarray:
        """Sample the same number of episodes from each worker."""
        ready_env_ids = ready_env_ids[unfinished_ind_local]
        self.data = self.data[unfinished_ind_local]
        if len(unfinished_ind_local) == 0:
            ready_env_ids = np.arange(self.env_num)
            self._reset_env_with_obs_next_to_obs(gym_reset_kwargs)
            for i in ready_env_ids:
                self._reset_state(i)
        return ready_env_ids

    def step_env_and_update_data(
        self,
        collected_rollout_batch: RolloutBatchProtocol,
        action_remap: np.ndarray,
        ready_env_ids: np.ndarray,
    ) -> np.ndarray:
        """Step the environment one action and update the data accordingly. Return the array of workerwise done signal."""
        obs_next, rew, terminated, truncated, info = self.env.step(
            action_remap,
            ready_env_ids,
        )
        done = np.logical_or(terminated, truncated)
        collected_rollout_batch.update(
            obs_next=obs_next,
            rew=rew,
            terminated=terminated,
            truncated=truncated,
            done=done,
            info=info,
        )
        if self.preprocess_fn:
            collected_rollout_batch.update(
                self.preprocess_fn(
                    obs_next=collected_rollout_batch.obs_next,
                    rew=collected_rollout_batch.rew,
                    done=collected_rollout_batch.done,
                    info=collected_rollout_batch.info,
                    policy=collected_rollout_batch.policy,
                    env_id=ready_env_ids,
                    act=collected_rollout_batch.act,
                ),
            )
        return done

    def compute_next_action_in_active_workers(
        self,
        collected_rollout_batch: RolloutBatchProtocol,
        last_state: Any,  # TODO: type
        no_grad: bool,
        random: bool,
        ready_env_ids: np.ndarray,
    ) -> None:
        """Compute the next action in the active workers and update the data accordingly."""
        if random:
            try:
                act_sample = [self._action_space[i].sample() for i in ready_env_ids]
            except TypeError:  # envpool's action space is not for per-env
                act_sample = [self._action_space.sample() for _ in ready_env_ids]
            act_sample = self.policy.map_action_inverse(act_sample)  # type: ignore
            collected_rollout_batch.update(act=act_sample)
        else:
            if no_grad:
                with torch.no_grad():  # faster than the retain_grad version
                    # collected_rollout_batch.obs will be used by agent to get the result
                    result = self.policy(collected_rollout_batch, last_state)
            else:
                result = self.policy(collected_rollout_batch, last_state)
            # update state / act / policy into self.data
            policy = result.get("policy", Batch())
            if not isinstance(policy, Batch):
                raise RuntimeError(
                    f"The policy result should be a {Batch}, but got {type(policy)}",
                )
            state = result.get("state", None)
            if state is not None:
                policy.hidden_state = state  # save state into buffer
            act = to_numpy(result.act)
            if self.exploration_noise:
                act = self.policy.exploration_noise(act, collected_rollout_batch)
            collected_rollout_batch.update(policy=policy, act=act)

    def _validate_collect_input(
        self,
        n_episode: int | None,
        n_step: int | None,
        sample_equal_num_episodes_per_worker: bool,
    ) -> None:
        """Check that exactly one of n_step or n_episode is specified."""
        if n_step is not None and n_episode is not None:
            raise ValueError(
                f"Only one of n_step or n_episode is allowed in Collector."
                f"collect, but got {n_step=}, {n_episode=}.",
            )

        if n_step is not None:
            if sample_equal_num_episodes_per_worker:
                raise ValueError(
                    "sample_equal_num_episodes_per_worker can only be used if `n_episode` is specified but"
                    "got `n_step` instead.",
                )
            if n_step < 1:
                raise ValueError(f"n_step should be an integer larger than 0, but got {n_step}.")

            if n_step % self.env_num:
                warnings.warn(
                    f"{n_step=} is not a multiple of ({self.env_num=}). "
                    "This may cause extra transitions to be collected into the buffer.",
                )
        elif n_episode is not None:
            if n_episode < 1:
                raise ValueError(
                    f"{n_episode=} should be an integer larger than 0.",
                )
            if n_episode < self.env_num:
                raise ValueError(
                    f"{n_episode=} should be larger than or equal to {self.env_num=} "
                    f"(otherwise you will get idle workers).",
                )
            if sample_equal_num_episodes_per_worker and n_episode % self.env_num != 0:
                raise ValueError(
                    f"{n_episode=} must be a multiple of {self.env_num=} "
                    f"when using sample_equal_num_episodes_per_worker.",
                )
        else:
            raise ValueError(
                f"At least one of {n_step=} and {n_episode=} should be specified as int larger than 0.",
            )

    def _update_collection_counters(
        self,
        collect_call_duration: float,
        episode_count: int,
        step_count: int,
    ) -> None:
        self.collect_step += step_count
        self.collect_episode += episode_count
        self.collect_time += collect_call_duration


class AsyncCollector(Collector):
    """Async Collector handles async vector environment.

    The arguments are exactly the same as :class:`~tianshou.data.Collector`, please
    refer to :class:`~tianshou.data.Collector` for more detailed explanation.
    """

    def __init__(
        self,
        policy: BasePolicy,
        env: BaseVectorEnv,
        buffer: ReplayBuffer | None = None,
        preprocess_fn: Callable[..., RolloutBatchProtocol] | None = None,
        exploration_noise: bool = False,
    ) -> None:
        warnings.warn("Using async setting may collect extra transitions into buffer.")
        super().__init__(
            policy,
            env,
            buffer,
            preprocess_fn,
            exploration_noise,
        )

    @override
    def _validate_env(self) -> None:
        if not self.env.is_async:
            raise ValueError(f"Please use {Collector.__name__} for using non-async envs.")

    def reset_env(
        self,
        gym_reset_kwargs: dict[str, Any] | None = None,
        set_obs_next_to_obs: bool = False,
    ) -> None:
        super().reset_env(gym_reset_kwargs)
        self._ready_env_ids = np.arange(self.env_num)

    def collect(
        self,
        n_step: int | None = None,
        n_episode: int | None = None,
        random: bool = False,
        render: float | None = None,
        no_grad: bool = True,
        gym_reset_kwargs: dict[str, Any] | None = None,
        sample_equal_from_each_env: bool = False,
    ) -> CollectStats:
        """Collect a specified number of step or episode with async env setting.

        This function doesn't collect exactly n_step or n_episode number of
        transitions. Instead, in order to support async setting, it may collect more
        than given n_step or n_episode transitions and save into buffer.

        :param n_step: how many steps you want to collect.
        :param n_episode: how many episodes you want to collect.
        :param random: whether to use random policy for collecting data.
        :param render: the sleep time between rendering consecutive frames.
            Default behaviour is no rendering.
        :param no_grad: whether to retain gradient in policy.forward(). Default
            is no gradient retaining.
        :param gym_reset_kwargs: extra keyword arguments to pass into the environment's
            reset function. Defaults to None (extra keyword arguments)

        .. note::

            One and only one collection number specification is permitted, either
            ``n_step`` or ``n_episode``.

        :return: A dataclass object
        """
        assert (
            sample_equal_from_each_env is False
        ), "AyncCollector does not support sample_equal_from_each_env."
        # collect at least n_step or n_episode
        if n_step is not None:
            assert n_episode is None, (
                "Only one of n_step or n_episode is allowed in Collector."
                f"collect, got n_step={n_step}, n_episode={n_episode}."
            )
            assert n_step > 0
        elif n_episode is not None:
            assert n_episode > 0
        else:
            raise TypeError(
                "Please specify at least one (either n_step or n_episode) "
                "in AsyncCollector.collect().",
            )

        ready_env_ids = self._ready_env_ids

        start_time = time.time()

        step_count = 0
        episode_count = 0
        episode_returns: list[float] = []
        episode_lens: list[int] = []
        episode_start_indices: list[int] = []

        while True:
            whole_data = self.data
            self.data = self.data[ready_env_ids]
            assert (
                len(whole_data) == self.env_num
            )  # major difference            # restore the state: if the last state is None, it won't store
            last_state = self.data.policy.pop("hidden_state", None)

            # get the next action
            if random:
                try:
                    act_sample = [self._action_space[i].sample() for i in ready_env_ids]
                except TypeError:  # envpool's action space is not for per-env
                    act_sample = [self._action_space.sample() for _ in ready_env_ids]
                act_sample = self.policy.map_action_inverse(act_sample)  # type: ignore
                self.data.update(act=act_sample)
            else:
                if no_grad:
                    with torch.no_grad():  # faster than retain_grad version
                        # self.data.obs will be used by agent to get result
                        result = self.policy(self.data, last_state)
                else:
                    result = self.policy(self.data, last_state)
                # update state / act / policy into self.data
                policy = result.get("policy", Batch())
                assert isinstance(policy, Batch)
                state = result.get("state", None)
                if state is not None:
                    policy.hidden_state = state  # save state into buffer
                act = to_numpy(result.act)
                if self.exploration_noise:
                    act = self.policy.exploration_noise(act, self.data)
                self.data.update(policy=policy, act=act)

            # save act/policy before env.step
            try:
                whole_data.act[ready_env_ids] = self.data.act  # type: ignore
                whole_data.policy[ready_env_ids] = self.data.policy
            except ValueError:
                alloc_by_keys_diff(whole_data, self.data, self.env_num, False)
                whole_data[ready_env_ids] = self.data  # lots of overhead

            # get bounded and remapped actions first (not saved into buffer)
            action_remap = self.policy.map_action(self.data.act)
            # step in env
            obs_next, rew, terminated, truncated, info = self.env.step(
                action_remap,
                ready_env_ids,
            )
            done = np.logical_or(terminated, truncated)

            # change self.data here because ready_env_ids has changed
            try:
                ready_env_ids = info["env_id"]
            except Exception:
                ready_env_ids = np.array([i["env_id"] for i in info])
            self.data = whole_data[ready_env_ids]

            self.data.update(
                obs_next=obs_next,
                rew=rew,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
            if self.preprocess_fn:
                try:
                    self.data.update(
                        self.preprocess_fn(
                            obs_next=self.data.obs_next,
                            rew=self.data.rew,
                            terminated=self.data.terminated,
                            truncated=self.data.truncated,
                            info=self.data.info,
                            env_id=ready_env_ids,
                            act=self.data.act,
                        ),
                    )
                except TypeError:
                    self.data.update(
                        self.preprocess_fn(
                            obs_next=self.data.obs_next,
                            rew=self.data.rew,
                            done=self.data.done,
                            info=self.data.info,
                            env_id=ready_env_ids,
                            act=self.data.act,
                        ),
                    )

            if render:
                self.env.render()
                if render > 0 and not np.isclose(render, 0):
                    time.sleep(render)

            # add data into the buffer
            ep_add_at_idx, ep_rew, ep_len, ep_start_idx = self.buffer.add(
                self.data,
                buffer_ids=ready_env_ids,
            )

            # collect statistics
            step_count += len(ready_env_ids)

            if np.any(done):
                env_ind_local = np.where(done)[0]
                env_ind_global = ready_env_ids[env_ind_local]
                episode_count += len(env_ind_local)
                episode_lens.extend(ep_len[env_ind_local])
                episode_returns.extend(ep_rew[env_ind_local])
                episode_start_indices.extend(ep_start_idx[env_ind_local])
                # now we copy obs_next to obs, but since there might be
                # finished episodes, we have to reset finished envs first.
                self._reset_env_with_ids(env_ind_local, env_ind_global, gym_reset_kwargs)
                for i in env_ind_local:
                    self._reset_state(i)

            try:
                # Need to ignore types b/c according to mypy Tensors cannot be indexed
                # by arrays (which they can...)
                whole_data.obs[ready_env_ids] = self.data.obs_next  # type: ignore
                whole_data.rew[ready_env_ids] = self.data.rew
                whole_data.done[ready_env_ids] = self.data.done
                whole_data.info[ready_env_ids] = self.data.info  # type: ignore
            except ValueError:
                alloc_by_keys_diff(whole_data, self.data, self.env_num, False)
                self.data.obs = self.data.obs_next
                # lots of overhead
                whole_data[ready_env_ids] = self.data
            self.data = whole_data

            if (n_step and step_count >= n_step) or (n_episode and episode_count >= n_episode):
                break

        self._ready_env_ids = ready_env_ids

        # generate statistics
        self.collect_step += step_count
        self.collect_episode += episode_count
        collect_time = max(time.time() - start_time, 1e-9)
        self.collect_time += collect_time

        return CollectStats(
            n_collected_episodes=episode_count,
            n_collected_steps=step_count,
            collect_time=collect_time,
            collect_speed=step_count / collect_time,
            returns=np.array(episode_returns),
            returns_stat=SequenceSummaryStats.from_sequence(episode_returns)
            if len(episode_returns) > 0
            else None,
            lens=np.array(episode_lens, int),
            lens_stat=SequenceSummaryStats.from_sequence(episode_lens)
            if len(episode_lens) > 0
            else None,
        )
