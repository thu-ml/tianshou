import time
import warnings
from dataclasses import dataclass
from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch

from tianshou.data import (
    Batch,
    CachedReplayBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
    ReplayBufferManager,
    SequenceSummaryStats,
    VectorReplayBuffer,
    to_numpy,
)
from tianshou.data.batch import alloc_by_keys_diff
from tianshou.data.types import ObsBatchProtocol, RolloutBatchProtocol
from tianshou.env import BaseVectorEnv, DummyVectorEnv
from tianshou.policy import BasePolicy
from tianshou.utils.print import DataclassPPrintMixin


@dataclass(kw_only=True)
class CollectStatsBase(DataclassPPrintMixin):
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
    returns: np.ndarray
    """The collected episode returns."""
    returns_stat: SequenceSummaryStats | None  # can be None if no episode ends during collect step
    """Stats of the collected returns."""
    lens: np.ndarray
    """The collected episode lengths."""
    lens_stat: SequenceSummaryStats | None  # can be None if no episode ends during collect step
    """Stats of the collected episode lengths."""


def _get_empty_batch_for_collect(
    initial_obs: np.ndarray | None = None,
    initial_info: dict | list[dict] | None = None,
) -> Batch:
    """Empty batch with obs and info of current state, useful for adding new data."""
    return Batch(
        obs=initial_obs,
        info=initial_info,
        # collector uses policy to get a potential hidden state. Not
        # specifying it would lead to AttributeError
        # TODO: is this an appropriate mechanism? Policy is certainly not of a type policy, as
        #   hidden_state is retrieved from it with pop
        policy={},
    )


class Collector:
    """Collector enables the policy to interact with different types of envs with exact number of steps or episodes.

    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy` class.
    :param env: a ``gym.Env`` environment or an instance of the
        :class:`~tianshou.env.BaseVectorEnv` class.
    :param buffer: an instance of the :class:`~tianshou.data.ReplayBuffer` class.
        If set to None, will instantiate a :class:`~tianshou.data.VectorReplayBuffer`
        as the default buffer.
    :param exploration_noise: determine whether the action needs to be modified
        with the corresponding policy's exploration noise. If so, "policy.
        exploration_noise(act, batch)" will be called automatically to add the
        exploration noise into action. Default to False.

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
        exploration_noise: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(env, gym.Env) and not hasattr(env, "__len__"):
            warnings.warn("Single environment detected, wrap to DummyVectorEnv.")
            # Unfortunately, mypy seems to ignore the isinstance in lambda, maybe a bug in mypy
            self.env = DummyVectorEnv([lambda: env])
        else:
            self.env = env  # type: ignore
        self.env_num = len(self.env)
        self.exploration_noise = exploration_noise
        self.buffer = self._assign_buffer(buffer)
        self.policy = policy
        self._action_space = self.env.action_space

        self._pre_collect_obs: np.ndarray | None = None
        self._pre_collect_info: list[dict] | None = None
        self._last_hidden_state_of_policy_RH: np.ndarray | torch.Tensor | Batch | None = None
        self._is_closed = False
        self.collect_step, self.collect_episode, self.collect_time = 0, 0, 0.0

    def close(self) -> None:
        """Close the collector and the environment."""
        self.env.close()
        self._pre_collect_obs = None
        self._pre_collect_info = None
        self._is_closed = True

    @property
    def is_closed(self) -> bool:
        """Return True if the collector is closed."""
        return self._is_closed

    def _assign_buffer(self, buffer: ReplayBuffer | None) -> ReplayBuffer:
        """Check if the buffer matches the constraint."""
        if buffer is None:
            buffer = VectorReplayBuffer(self.env_num, self.env_num)
        elif isinstance(buffer, ReplayBufferManager):
            assert buffer.buffer_num >= self.env_num
            if isinstance(buffer, CachedReplayBuffer):
                assert buffer.cached_buffer_num >= self.env_num
        else:  # ReplayBuffer or PrioritizedReplayBuffer
            assert buffer.maxsize > 0
            if self.env_num > 1:
                if isinstance(buffer, ReplayBuffer):
                    buffer_type = "ReplayBuffer"
                    vector_type = "VectorReplayBuffer"
                if isinstance(buffer, PrioritizedReplayBuffer):
                    buffer_type = "PrioritizedReplayBuffer"
                    vector_type = "PrioritizedVectorReplayBuffer"
                raise TypeError(
                    f"Cannot use {buffer_type}(size={buffer.maxsize}, ...) to collect "
                    f"{self.env_num} envs,\n\tplease use {vector_type}(total_size="
                    f"{buffer.maxsize}, buffer_num={self.env_num}, ...) instead.",
                )
        return buffer

    def reset(
        self,
        reset_buffer: bool = True,
        reset_stats: bool = True,
        gym_reset_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Reset the environment, statistics, current data and possibly replay memory.

        :param reset_buffer: if true, reset the replay buffer attached
            to the collector.
        :param reset_stats: if true, reset the statistics attached to the collector.
        :param gym_reset_kwargs: extra keyword arguments to pass into the environment's
            reset function. Defaults to None (extra keyword arguments)
        """
        self._pre_collect_obs, self._pre_collect_info = self.reset_env(gym_reset_kwargs)
        if reset_buffer:
            self.reset_buffer()
        if reset_stats:
            self.reset_stat()
        self._is_closed = False

    def reset_stat(self) -> None:
        """Reset the statistic variables."""
        self.collect_step, self.collect_episode, self.collect_time = 0, 0, 0.0

    def reset_buffer(self, keep_statistics: bool = False) -> None:
        """Reset the data buffer."""
        self.buffer.reset(keep_statistics=keep_statistics)

    def reset_env(
        self,
        gym_reset_kwargs: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict | list[dict]]:
        """Reset all the environments."""
        gym_reset_kwargs = gym_reset_kwargs if gym_reset_kwargs else {}
        obs, info = self.env.reset(**gym_reset_kwargs)
        return obs, info

    def _reset_done_envs_and_hidden_state_of_policy_with_id(
        self,
        local_ids: list[int] | np.ndarray,
        global_ids: list[int] | np.ndarray,
        gym_reset_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Reset the environments in which the trajectory reached done identified
        by the local/ global ids. Also reset the respective hidden state of the policy.
        """
        gym_reset_kwargs = gym_reset_kwargs if gym_reset_kwargs else {}
        obs_reset, info = self.env.reset(global_ids, **gym_reset_kwargs)
        self._pre_collect_obs[local_ids] = obs_reset
        for id in local_ids:
            self._pre_collect_info[id] = info
            if isinstance(self._last_hidden_state_of_policy_RH, torch.Tensor):
                self._last_hidden_state_of_policy_RH[id].zero_()
            elif isinstance(self._last_hidden_state_of_policy_RH, np.ndarray):
                self._last_hidden_state_of_policy_RH[id] = (
                    None if self._last_hidden_state_of_policy_RH.dtype == object else 0
                )
            elif isinstance(self._last_hidden_state_of_policy_RH, Batch):
                self._last_hidden_state_of_policy_RH.empty_(id)

    # TODO: reduce complexity, remove the noqa
    def collect(  # noqa: C901
        self,
        n_step: int | None = None,
        n_episode: int | None = None,
        random: bool = False,
        render: float | None = None,
        no_grad: bool = True,
        reset_before_collect: bool = False,
        gym_reset_kwargs: dict[str, Any] | None = None,
    ) -> CollectStats:
        """Collect a specified number of steps or episodes.

        To ensure an unbiased sampling result with the n_episode option, this function will
        first collect ``n_episode - env_num`` episodes, then for the last ``env_num``
        episodes, they will be collected evenly from each env.

        :param n_step: how many steps you want to collect.
        :param n_episode: how many episodes you want to collect.
        :param random: whether to use random policy for collecting data.
        :param render: the sleep time between rendering consecutive frames.
        :param no_grad: whether to retain gradient in policy.forward().
        :param reset_before_collect: whether to reset the environment before
            collecting data.
            It has only an effect if n_episode is not None, i.e.
             if one wants to collect a fixed number of steps.
            (The collector needs the initial obs and info to function properly.)
        :param gym_reset_kwargs: extra keyword arguments to pass into the environment's
            reset function. Only used if reset_before_collect is True.

        .. note::

            One and only one collection number specification is permitted, either
            ``n_step`` or ``n_episode``.

        :return: A dataclass object
        """
        # Input validation
        assert not self.env.is_async, "Please use AsyncCollector if using async venv."
        if n_step is not None:
            assert n_episode is None, (
                f"Only one of n_step or n_episode is allowed in Collector."
                f"collect, got n_step={n_step}, n_episode={n_episode}."
            )
            assert n_step > 0
            if n_step % self.env_num != 0:
                warnings.warn(
                    f"n_step={n_step} is not a multiple of #env ({self.env_num}), "
                    "which may cause extra transitions collected into the buffer.",
                )
            ready_env_ids_R = np.arange(self.env_num)
        elif n_episode is not None:
            assert n_episode > 0
            if self.env_num > n_episode:
                raise ValueError(
                    f"{n_episode=} should be larger than {self.env_num=} to "
                    f"collect at least one trajectory in each environment.",
                )
            ready_env_ids_R = np.arange(min(self.env_num, n_episode))
        else:
            raise TypeError(
                "Please specify at least one (either n_step or n_episode) "
                "in AsyncCollector.collect().",
            )

        start_time = time.time()

        if n_episode or reset_before_collect:
            self.reset(reset_buffer=False, gym_reset_kwargs=gym_reset_kwargs)

        if self._pre_collect_obs is None or self._pre_collect_info is None:
            raise ValueError(
                "Initial obs and info should not be None. "
                "Either reset the collector (using reset or reset_env) or pass reset_before_collect=True to collect.",
            )

        # get the first obs to be the current obs in the n_step case as
        # episodes as a new call to collect does not restart trajectories
        # (which we also really don't want)
        step_count = 0
        num_collected_episodes = 0
        episode_returns: list[float] = []
        episode_lens: list[int] = []
        episode_start_indices: list[int] = []

        # NAMING CONVENTION (mostly suffixes):
        # episode - An episode means a rollout until done (terminated or truncated). After an episode is completed,
        # the corresponding env is either reset or removed from the ready envs.
        # R - number ready env ids. Note that this might change when envs get idle.
        # This can only happen in n_episode case, see explanation in the corresponding block.
        # O - dimension(s) of observations
        # A - dimension(s) of actions
        # H - dimension(s) of hidden state
        # D - number of envs that reached done in the current collect iteration. Only relevant in n_episode case.

        last_obs_RO, last_info_R = self._pre_collect_obs, self._pre_collect_info
        while True:
            # todo check if we need this when using cur_rollout_batch
            # if len(cur_rollout_batch) != len(ready_env_ids):
            #     raise RuntimeError(
            #         f"The length of the collected_rollout_batch {len(cur_rollout_batch)}) is not equal to the length of ready_env_ids"
            #         f"{len(ready_env_ids)}. This should not happen and could be a bug!",
            #     )
            # restore the state: if the last state is None, it won't store

            # get the next action
            # TODO: maybe extract a method?
            if random:
                try:
                    act_RA = [self._action_space[i].sample() for i in ready_env_ids_R]
                # TODO: test whether envpool env exlicitly
                except TypeError:  # envpool's action space is not for per-env
                    act_RA = [self._action_space.sample() for _ in ready_env_ids_R]
                act_RA = self.policy.map_action_inverse(act_RA)  # type: ignore
                policy_R = Batch()
            else:
                obs_batch = cast(ObsBatchProtocol, Batch(obs=last_obs_RO, info=last_info_R))
                # TODO: this modifies the batch in place, which is not a good idea!
                if no_grad:
                    with torch.no_grad():  # faster than the retain_grad version
                        # cur_rollout_batch.obs is used by the agent to get the result
                        act_batch_RA = self.policy(
                            obs_batch,
                            self._last_hidden_state_of_policy_RH,
                        )
                else:
                    act_batch_RA = self.policy(
                        obs_batch,
                        self._last_hidden_state_of_policy_RH,
                    )

                act_RA = to_numpy(act_batch_RA.act)
                if self.exploration_noise:
                    act_RA = self.policy.exploration_noise(
                        act_RA,
                        obs_batch,
                    )

                # TODO: cleanup the whole policy in batch thing
                # todo policy_R can also be none, check
                policy_R = act_batch_RA.get("policy", Batch())
                if not isinstance(policy_R, Batch):
                    raise RuntimeError(
                        f"The policy result should be a {Batch}, but got {type(policy_R)}",
                    )

                if hidden_state_RH := act_batch_RA.get("state", None) is not None:
                    policy_R.hidden_state = hidden_state_RH  # save state into buffer
                    self._last_hidden_state_of_policy_RH = hidden_state_RH

            normalized_action_R = self.policy.map_action(act_RA)
            obs_next_RO, rew_R, terminated_R, truncated_R, info_R = self.env.step(
                normalized_action_R,
                ready_env_ids_R,
            )
            done_R = np.logical_or(terminated_R, truncated_R)

            rollout_batch = cast(
                RolloutBatchProtocol,
                Batch(
                    obs=last_obs_RO,
                    act=act_RA,
                    policy=policy_R,  # TODO: can we pass None here instead?
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

            # add data into the buffer
            ptr_R, ep_rew_R, ep_len_R, ep_idx_R = self.buffer.add(
                rollout_batch,
                buffer_ids=ready_env_ids_R,
            )

            # collect statistics
            num_collected_episodes += np.sum(done_R)
            step_count += len(ready_env_ids_R)

            if (n_step and step_count >= n_step) or (
                n_episode and num_collected_episodes >= n_episode
            ):
                break

            # Resetting envs that reached done, or removing some of them from the collection if needed (see below)
            if np.any(done_R):
                # TODO: adjust the whole index story, don't use np.where, just slice with boolean arrays
                # D - number of envs that reached done in the rollout above
                env_ind_local_D = np.where(done_R)[0]
                env_ind_global_D = ready_env_ids_R[env_ind_local_D]
                episode_lens.extend(ep_len_R[env_ind_local_D])
                episode_returns.extend(ep_rew_R[env_ind_local_D])
                episode_start_indices.extend(ep_idx_R[env_ind_local_D])
                # now we copy obs_next to obs, but since there might be
                # finished episodes, we have to reset finished envs first.

                # cur_rollout_batch = self._reset_env_with_ids(
                self._reset_done_envs_and_hidden_state_of_policy_with_id(
                    env_ind_local_D,
                    env_ind_global_D,
                    gym_reset_kwargs,
                )

                # HANDLING THE CASE WHEN WE HAVE MORE READY ENVS THAN DESIRED
                # This can only happen if we are collecting a fixed number of episodes
                # If we have more ready envs than there are remaining episodes to collect,
                # we will remove some of them for the next rollout
                # One effect of this is the following: only envs that have completed an episode
                # in the last step can ever be removed from the ready envs.
                # Thus, this guarantees that each env will contribute at least one episode to the
                # collected data (the buffer). This effect was previous called "avoiding bias in selecting environments"
                # However, it is not at all clear whether this is actually useful or necessary.
                # New naming convention:
                # S - number of surplus envs
                # TODO: can the whole block be removed? If we have too many episodes, we could just strip the last ones.
                #   Changing R to R-S highly increases the complexity of the code.
                if n_episode:
                    remaining_episodes_to_collect = n_episode - num_collected_episodes
                    surplus_env_num = len(ready_env_ids_R) - remaining_episodes_to_collect
                    if surplus_env_num > 0:
                        # R becomes R-S here, preparing for next iteration in while loop
                        # Everything that was of length R needs to be filtered and become of length R-S
                        # Note that this won't be the last iteration, as one iteration equals one
                        # step and we still need to collect the remaining episodes to reach the breaking condition

                        # creating the mask
                        env_to_be_ignored_ind_local_S = env_ind_local_D[:surplus_env_num]
                        env_should_remain_R = np.ones_like(ready_env_ids_R, dtype=bool)
                        env_should_remain_R[env_to_be_ignored_ind_local_S] = False
                        # stripping the "idle" indices, shortening the relevant quantities
                        ready_env_ids_R = ready_env_ids_R[env_should_remain_R]
                        obs_next_RO = obs_next_RO[env_should_remain_R]
                        info_R = info_R[env_should_remain_R]
                        if self._last_hidden_state_of_policy_RH is not None:
                            self._last_hidden_state_of_policy_RH = (
                                self._last_hidden_state_of_policy_RH[env_should_remain_R]
                            )
                        # NOTE: no need to strip last_obs_RO or last_inf_R, they are overwritten

            # preparing for next iteration
            # NOTE: cannot happen earlier b/c of possible masking, see above
            last_obs_RO = obs_next_RO
            last_info_R = info_R

        # generate statistics
        self.collect_step += step_count
        self.collect_episode += num_collected_episodes
        collect_time = max(time.time() - start_time, 1e-9)
        self.collect_time += collect_time

        if n_step:
            # persist for future collect iterations
            # todo is this still needed, stepping in the end seems abandoned
            self._pre_collect_obs = last_obs_RO  # type: ignore
            self._pre_collect_info = last_info_R  # type: ignore

        return CollectStats(
            n_collected_episodes=num_collected_episodes,
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
        exploration_noise: bool = False,
    ) -> None:
        # assert env.is_async
        warnings.warn("Using async setting may collect extra transitions into buffer.")
        super().__init__(
            policy,
            env,
            buffer,
            exploration_noise,
        )

    # def reset_env(self, gym_reset_kwargs: dict[str, Any] | None = None) -> None:
    #     obs, info = super().reset_env(gym_reset_kwargs)
    #     self._ready_env_ids = np.arange(self.env_num)

    def collect(
        self,
        n_step: int | None = None,
        n_episode: int | None = None,
        random: bool = False,
        render: float | None = None,
        no_grad: bool = True,
        gym_reset_kwargs: dict[str, Any] | None = None,
    ) -> CollectStats:
        """Collect a specified number of steps or episodes with async env setting.

        This function doesn't collect exactly n_step or n_episode number of
        transitions. Instead, in order to support async setting, it may collect more
        than given n_step or n_episode transitions and save into buffer.

        :param n_step: how many steps you want to collect.
        :param n_episode: how many episodes you want to collect.
        :param random: whether to use random policy for collecting data. Default
            to False.
        :param render: the sleep time between rendering consecutive frames.
            Default to None (no rendering).
        :param no_grad: whether to retain gradient in policy.forward(). Default to
            True (no gradient retaining).
        :param gym_reset_kwargs: extra keyword arguments to pass into the environment's
            reset function. Defaults to None (extra keyword arguments)

        .. note::

            One and only one collection number specification is permitted, either
            ``n_step`` or ``n_episode``.

        :return: A dataclass object
        """
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

        ready_env_ids = np.arange(self.env_num)

        start_time = time.time()

        cur_rollout_batch = _get_empty_batch_for_collect(
            self._pre_collect_obs,
            self._pre_collect_info,
        )
        step_count = 0
        episode_count = 0
        episode_returns: list[float] = []
        episode_lens: list[int] = []
        episode_start_indices: list[int] = []

        while True:
            whole_data = cur_rollout_batch
            cur_rollout_batch = cur_rollout_batch[ready_env_ids]
            assert len(whole_data) == self.env_num  # major difference
            # restore the state: if the last state is None, it won't store
            last_state = cur_rollout_batch.policy.pop("hidden_state", None)

            # get the next action
            if random:
                try:
                    act_sample = [self._action_space[i].sample() for i in ready_env_ids]
                except TypeError:  # envpool's action space is not for per-env
                    act_sample = [self._action_space.sample() for _ in ready_env_ids]
                act_sample = self.policy.map_action_inverse(act_sample)  # type: ignore
                cur_rollout_batch.update(act=act_sample)
            else:
                if no_grad:
                    with torch.no_grad():  # faster than the retain_grad version
                        # cur_rollout_batch.obs will be used by agent to get result
                        result = self.policy(cur_rollout_batch, last_state)
                else:
                    result = self.policy(cur_rollout_batch, last_state)
                # update state / act / policy into cur_rollout_batch
                policy = result.get("policy", Batch())
                assert isinstance(policy, Batch)
                state = result.get("state", None)
                if state is not None:
                    policy.hidden_state = state  # save state into buffer
                act = to_numpy(result.act)
                if self.exploration_noise:
                    act = self.policy.exploration_noise(act, cur_rollout_batch)
                cur_rollout_batch.update(policy=policy, act=act)

            # save act/policy before env.step
            try:
                whole_data.act[ready_env_ids] = cur_rollout_batch.act  # type: ignore
                whole_data.policy[ready_env_ids] = cur_rollout_batch.policy
            except ValueError:
                alloc_by_keys_diff(whole_data, cur_rollout_batch, self.env_num, False)
                whole_data[ready_env_ids] = cur_rollout_batch  # lots of overhead

            # get bounded and remapped actions first (not saved into buffer)
            action_remap = self.policy.map_action(cur_rollout_batch.act)
            # step in env
            obs_next, rew, terminated, truncated, info = self.env.step(
                action_remap,
                ready_env_ids,
            )
            done = np.logical_or(terminated, truncated)

            # change cur_rollout_batch here because ready_env_ids has changed
            try:
                ready_env_ids = info["env_id"]
            except Exception:
                ready_env_ids = np.array([i["env_id"] for i in info])
            cur_rollout_batch = whole_data[ready_env_ids]

            cur_rollout_batch.update(
                obs_next=obs_next,
                rew=rew,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )

            if render:
                self.env.render()
                if render > 0 and not np.isclose(render, 0):
                    time.sleep(render)

            # add data into the buffer
            ptr, ep_rew, ep_len, ep_idx = self.buffer.add(
                cur_rollout_batch,
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
                episode_start_indices.extend(ep_idx[env_ind_local])
                # now we copy obs_next to obs, but since there might be
                # finished episodes, we have to reset finished envs first.
                self._reset_done_envs_and_hidden_state_of_policy_with_id(
                    env_ind_local,
                    env_ind_global,
                    gym_reset_kwargs,
                )

            try:
                # Need to ignore types b/c according to mypy Tensors cannot be indexed
                # by arrays (which they can...)
                whole_data.obs[ready_env_ids] = cur_rollout_batch.obs_next  # type: ignore
                whole_data.rew[ready_env_ids] = cur_rollout_batch.rew
                whole_data.done[ready_env_ids] = cur_rollout_batch.done
                whole_data.info[ready_env_ids] = cur_rollout_batch.info  # type: ignore
            except ValueError:
                alloc_by_keys_diff(whole_data, cur_rollout_batch, self.env_num, False)
                cur_rollout_batch.obs = cur_rollout_batch.obs_next
                # lots of overhead
                whole_data[ready_env_ids] = cur_rollout_batch
            cur_rollout_batch = whole_data

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
