from typing import Any, Callable, Optional, Union

import numpy as np

from tianshou.data import Batch, ReplayBuffer
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import RolloutBatchProtocol


class HERReplayBuffer(ReplayBuffer):
    """Implementation of Hindsight Experience Replay. arXiv:1707.01495.

    HERReplayBuffer is to be used with goal-based environment where the
    observation is a dictionary with keys ``observation``, ``achieved_goal`` and
    ``desired_goal``. Currently support only HER's future strategy, online sampling.

    :param int size: the size of the replay buffer.
    :param compute_reward_fn: a function that takes 2 ``np.array`` arguments,
        ``acheived_goal`` and ``desired_goal``, and returns rewards as ``np.array``.
        The two arguments are of shape (batch_size, ...original_shape) and the returned
        rewards must be of shape (batch_size,).
    :param int horizon: the maximum number of steps in an episode.
    :param int future_k: the 'k' parameter introduced in the paper. In short, there
        will be at most k episodes that are re-written for every 1 unaltered episode
        during the sampling.

    .. seealso::

        Please refer to :class:`~tianshou.data.ReplayBuffer` for other APIs' usage.
    """

    def __init__(
        self,
        size: int,
        compute_reward_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        horizon: int,
        future_k: float = 8.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(size, **kwargs)
        self.horizon = horizon
        self.future_p = 1 - 1 / future_k
        self.compute_reward_fn = compute_reward_fn
        self._original_meta = Batch()
        self._altered_indices = np.array([])

    def _restore_cache(self) -> None:
        """Write cached original meta back to `self._meta`.

        It's called everytime before 'writing', 'sampling' or 'saving' the buffer.
        """
        if not hasattr(self, "_altered_indices"):
            return

        if self._altered_indices.size == 0:
            return
        self._meta[self._altered_indices] = self._original_meta
        # Clean
        self._original_meta = Batch()
        self._altered_indices = np.array([])

    def reset(self, keep_statistics: bool = False) -> None:
        self._restore_cache()
        return super().reset(keep_statistics)

    def save_hdf5(self, path: str, compression: Optional[str] = None) -> None:
        self._restore_cache()
        return super().save_hdf5(path, compression)

    def set_batch(self, batch: RolloutBatchProtocol) -> None:
        self._restore_cache()
        return super().set_batch(batch)

    def update(self, buffer: Union["HERReplayBuffer", "ReplayBuffer"]) -> np.ndarray:
        self._restore_cache()
        return super().update(buffer)

    def add(
        self,
        batch: RolloutBatchProtocol,
        buffer_ids: Optional[Union[np.ndarray, list[int]]] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._restore_cache()
        return super().add(batch, buffer_ids)

    def sample_indices(self, batch_size: int) -> np.ndarray:
        """Get a random sample of index with size = batch_size.

        Return all available indices in the buffer if batch_size is 0; return an \
        empty numpy array if batch_size < 0 or no available index can be sampled. \
        Additionally, some episodes of the sampled transitions will be re-written \
        according to HER.
        """
        self._restore_cache()
        indices = super().sample_indices(batch_size=batch_size)
        self.rewrite_transitions(indices.copy())
        return indices

    def rewrite_transitions(self, indices: np.ndarray) -> None:
        """Re-write the goal of some sampled transitions' episodes according to HER.

        Currently applies only HER's 'future' strategy. The new goals will be written \
        directly to the internal batch data temporarily and will be restored right \
        before the next sampling or when using some of the buffer's method (e.g. \
        `add`, `save_hdf5`, etc.). This is to make sure that n-step returns \
        calculation etc., performs correctly without additional alteration.
        """
        if indices.size == 0:
            return

        # Sort indices keeping chronological order
        indices[indices < self._index] += self.maxsize
        indices = np.sort(indices)
        indices[indices >= self.maxsize] -= self.maxsize

        # Construct episode trajectories
        indices = [indices]
        for _ in range(self.horizon - 1):
            indices.append(self.next(indices[-1]))
        indices = np.stack(indices)

        # Calculate future timestep to use
        current = indices[0]
        terminal = indices[-1]
        episodes_len = (terminal - current + self.maxsize) % self.maxsize
        future_offset = np.random.uniform(size=len(indices[0])) * episodes_len
        future_offset = np.round(future_offset).astype(int)
        future_t = (current + future_offset) % self.maxsize

        # Compute indices
        #   open indices are used to find longest, unique trajectories among
        #   presented episodes
        unique_ep_open_indices = np.sort(np.unique(terminal, return_index=True)[1])
        unique_ep_indices = indices[:, unique_ep_open_indices]
        #   close indices are used to find max future_t among presented episodes
        unique_ep_close_indices = np.hstack([(unique_ep_open_indices - 1)[1:], len(terminal) - 1])
        #   episode indices that will be altered
        her_ep_indices = np.random.choice(
            len(unique_ep_open_indices),
            size=int(len(unique_ep_open_indices) * self.future_p),
            replace=False,
        )

        # Cache original meta
        self._altered_indices = unique_ep_indices.copy()
        self._original_meta = self._meta[self._altered_indices].copy()

        # Copy original obs, ep_rew (and obs_next), and obs of future time step
        ep_obs = self[unique_ep_indices].obs
        # to satisfy mypy
        # TODO: add protocol covering these batches
        assert isinstance(ep_obs, BatchProtocol)
        ep_rew = self[unique_ep_indices].rew
        if self._save_obs_next:
            ep_obs_next = self[unique_ep_indices].obs_next
            # to satisfy mypy
            assert isinstance(ep_obs_next, BatchProtocol)
            future_obs = self[future_t[unique_ep_close_indices]].obs_next
        else:
            future_obs = self[self.next(future_t[unique_ep_close_indices])].obs

        # Re-assign goals and rewards via broadcast assignment
        ep_obs.desired_goal[:, her_ep_indices] = future_obs.achieved_goal[None, her_ep_indices]
        if self._save_obs_next:
            ep_obs_next.desired_goal[:, her_ep_indices] = future_obs.achieved_goal[
                None,
                her_ep_indices,
            ]
            ep_rew[:, her_ep_indices] = self._compute_reward(ep_obs_next)[:, her_ep_indices]
        else:
            tmp_ep_obs_next = self[self.next(unique_ep_indices)].obs
            assert isinstance(tmp_ep_obs_next, BatchProtocol)
            ep_rew[:, her_ep_indices] = self._compute_reward(tmp_ep_obs_next)[:, her_ep_indices]

        # Sanity check
        assert ep_obs.desired_goal.shape[:2] == unique_ep_indices.shape
        assert ep_obs.achieved_goal.shape[:2] == unique_ep_indices.shape
        assert ep_rew.shape == unique_ep_indices.shape

        # Re-write meta
        assert isinstance(self._meta.obs, BatchProtocol)
        self._meta.obs[unique_ep_indices] = ep_obs
        if self._save_obs_next:
            self._meta.obs_next[unique_ep_indices] = ep_obs_next
        self._meta.rew[unique_ep_indices] = ep_rew.astype(np.float32)

    def _compute_reward(self, obs: BatchProtocol, lead_dims: int = 2) -> np.ndarray:
        lead_shape = obs.observation.shape[:lead_dims]
        g = obs.desired_goal.reshape(-1, *obs.desired_goal.shape[lead_dims:])
        ag = obs.achieved_goal.reshape(-1, *obs.achieved_goal.shape[lead_dims:])
        rewards = self.compute_reward_fn(ag, g)
        return rewards.reshape(*lead_shape, *rewards.shape[1:])
