from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np

from tianshou.data import Batch, ReplayBuffer


class HERReplayBuffer(ReplayBuffer):

    def __init__(
        self,
        size: int,
        deconstruct_obs_fn: Callable[[np.ndarray], Batch],
        flatten_obs_fn: Callable[[Batch], np.ndarray],
        compute_reward_fn: Callable[[Batch], np.ndarray],
        horizon: int,
        future_k: float = 8.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(size, **kwargs)
        self.horizon = horizon
        self.future_p = 1 - 1 / future_k
        self.deconstruct_obs_fn = deconstruct_obs_fn
        self.flatten_obs_fn = flatten_obs_fn
        self.compute_reward_fn = compute_reward_fn
        self._original_meta = Batch()
        self._altered_indices = np.array([])

    def _restore_cache(self) -> None:
        """
        Write cached original meta back to self._meta
        Do this everytime before 'writing', 'sampling' or 'saving' the buffer.
        """
        if not hasattr(self, '_altered_indices'):
            return

        if self._altered_indices.size == 0:
            return
        self._meta[self._altered_indices] = self._original_meta
        # Clean
        del self._original_meta, self._altered_indices
        self._original_meta = Batch()
        self._altered_indices = np.array([])

    def reset(self, keep_statistics: bool = False) -> None:
        self._restore_cache()
        return super().reset(keep_statistics)

    def save_hdf5(self, path: str, compression: Optional[str] = None) -> None:
        self._restore_cache()
        return super().save_hdf5(path, compression)

    def set_batch(self, batch: Batch) -> None:
        self._restore_cache()
        return super().set_batch(batch)

    def update(self, buffer: Union["HERReplayBuffer", "ReplayBuffer"]) -> np.ndarray:
        self._restore_cache()
        return super().update(buffer)

    def add(
        self,
        batch: Batch,
        buffer_ids: Optional[Union[np.ndarray, List[int]]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._restore_cache()
        return super().add(batch, buffer_ids)

    def sample_indices(self, batch_size: int) -> np.ndarray:
        """Get a random sample of index with size = batch_size.
        Return all available indices in the buffer if batch_size is 0; return an empty
        numpy array if batch_size < 0 or no available index can be sampled.
        """
        self._restore_cache()
        indices = np.sort(super().sample_indices(batch_size=batch_size))
        self.rewrite_transitions(indices)
        return indices

    def rewrite_transitions(self, indices: np.ndarray) -> None:
        """ Re-write the goal of some sampled transitions' episodes according to HER's
        'future' strategy. The new goals will be written directly to the internal
        batch data temporarily and will be restored right before the next sampling or
        when using some of the buffer's method (such as `add` or `save_hdf5`). This is
        to make sure that n-step returns calculation etc. performs correctly without
        alteration.
        """
        if indices.size == 0:
            return
        # Construct episode trajectories
        indices = [indices]
        for _ in range(self.horizon - 1):
            indices.append(self.next(indices[-1]))
        indices = np.stack(indices)

        # Calculate future timestep to use
        current = indices[0]
        terminal = indices[-1]
        future_offset = np.random.uniform(size=len(indices[0])) * (terminal - current)
        future_offset = future_offset.astype(int)
        future_t = (current + future_offset)

        # Compute indices
        #   open indices are used to find longest, unique trajectories among
        #   presented episodes
        unique_ep_open_indices = np.unique(terminal, return_index=True)[1]
        unique_ep_indices = indices[:, unique_ep_open_indices]
        #   close indices are used to find max future_t among presented episodes
        unique_ep_close_indices = np.hstack(
            [(unique_ep_open_indices - 1)[1:],
             len(terminal) - 1]
        )
        #   episode indices that will be altered
        her_ep_indices = np.random.choice(
            len(unique_ep_open_indices),
            size=int(len(unique_ep_open_indices) * self.future_p),
            replace=False
        )

        # Copy original obs, ep_rew (and obs_next), and obs of future time step
        ep_obs = self._deconstruct_obs(self[unique_ep_indices].obs)
        ep_rew = self[unique_ep_indices].rew
        if self._save_obs_next:
            ep_obs_next = self._deconstruct_obs(self[unique_ep_indices].obs_next)
            future_obs = self._deconstruct_obs(
                self[future_t[unique_ep_close_indices]].obs_next, lead_dims=1
            )
        else:
            future_obs = self._deconstruct_obs(
                self[self.next(future_t[unique_ep_close_indices])].obs, lead_dims=1
            )

        # Re-assign goals and rewards via broadcast assignment
        ep_obs.g[:, her_ep_indices] = future_obs.ag[None, her_ep_indices]
        if self._save_obs_next:
            ep_obs_next.g[:, her_ep_indices] = future_obs.ag[None, her_ep_indices]
            ep_rew[:,
                   her_ep_indices] = self._compute_reward(ep_obs_next)[:,
                                                                       her_ep_indices]
        else:
            tmp_ep_obs_next = self._deconstruct_obs(
                self[self.next(unique_ep_indices)].obs
            )
            ep_rew[:, her_ep_indices] = self._compute_reward(tmp_ep_obs_next
                                                             )[:, her_ep_indices]

        # Sanity check
        assert ep_obs.g.shape[:2] == unique_ep_indices.shape
        assert ep_obs.ag.shape[:2] == unique_ep_indices.shape
        assert ep_rew.shape == unique_ep_indices.shape
        assert np.all(future_t >= indices[0])

        # Cache original meta
        self._altered_indices = unique_ep_indices.copy()
        self._original_meta = self._meta[self._altered_indices].copy()

        # Re-write meta
        self._meta.obs[unique_ep_indices] = self._flatten_obs(ep_obs)
        if self._save_obs_next:
            self._meta.obs_next[unique_ep_indices] = self._flatten_obs(ep_obs_next)
        self._meta.rew[unique_ep_indices] = ep_rew.astype(np.float32)

    # Reshaping obs into (bsz, *shape) instead of (..., *shape) before
    # calling the provided functions.
    def _deconstruct_obs(self, obs: np.ndarray, lead_dims: int = 2) -> Batch:
        lead_shape = obs.shape[:lead_dims]
        flatten_obs = obs.reshape(-1, *obs.shape[lead_dims:])
        de_obs = self.deconstruct_obs_fn(flatten_obs)
        de_obs.o = de_obs.o.reshape(*lead_shape, *de_obs.o.shape[1:])
        de_obs.g = de_obs.g.reshape(*lead_shape, *de_obs.g.shape[1:])
        de_obs.ag = de_obs.ag.reshape(*lead_shape, *de_obs.ag.shape[1:])
        return de_obs

    def _flatten_obs(self, de_obs: Batch, lead_dims: int = 2) -> np.ndarray:
        lead_shape = de_obs.o.shape[:lead_dims]
        de_obs.o = de_obs.o.reshape(-1, *de_obs.o.shape[lead_dims:])
        de_obs.g = de_obs.g.reshape(-1, *de_obs.g.shape[lead_dims:])
        de_obs.ag = de_obs.ag.reshape(-1, *de_obs.ag.shape[lead_dims:])
        flatten_obs = self.flatten_obs_fn(de_obs)
        return flatten_obs.reshape(*lead_shape, *flatten_obs.shape[1:])

    def _compute_reward(self, de_obs: Batch, lead_dims: int = 2) -> np.ndarray:
        lead_shape = de_obs.o.shape[:lead_dims]
        de_obs.o = de_obs.o.reshape(-1, *de_obs.o.shape[lead_dims:])
        de_obs.g = de_obs.g.reshape(-1, *de_obs.g.shape[lead_dims:])
        de_obs.ag = de_obs.ag.reshape(-1, *de_obs.ag.shape[lead_dims:])
        rewards = self.compute_reward_fn(de_obs)
        de_obs.o = de_obs.o.reshape(*lead_shape, *de_obs.o.shape[1:])
        de_obs.g = de_obs.g.reshape(*lead_shape, *de_obs.g.shape[1:])
        de_obs.ag = de_obs.ag.reshape(*lead_shape, *de_obs.ag.shape[1:])
        return rewards.reshape(*lead_shape, *rewards.shape[1:])
