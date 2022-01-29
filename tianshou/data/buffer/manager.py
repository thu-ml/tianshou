from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from numba import njit

from tianshou.data import Batch, PrioritizedReplayBuffer, ReplayBuffer
from tianshou.data.batch import _alloc_by_keys_diff, _create_value


class ReplayBufferManager(ReplayBuffer):
    """ReplayBufferManager contains a list of ReplayBuffer with exactly the same \
    configuration.

    These replay buffers have contiguous memory layout, and the storage space each
    buffer has is a shallow copy of the topmost memory.

    :param buffer_list: a list of ReplayBuffer needed to be handled.

    .. seealso::

        Please refer to :class:`~tianshou.data.ReplayBuffer` for other APIs' usage.
    """

    def __init__(self, buffer_list: List[ReplayBuffer]) -> None:
        self.buffer_num = len(buffer_list)
        self.buffers = np.array(buffer_list, dtype=object)
        offset, size = [], 0
        buffer_type = type(self.buffers[0])
        kwargs = self.buffers[0].options
        for buf in self.buffers:
            assert buf._meta.is_empty()
            assert isinstance(buf, buffer_type) and buf.options == kwargs
            offset.append(size)
            size += buf.maxsize
        self._offset = np.array(offset)
        self._extend_offset = np.array(offset + [size])
        self._lengths = np.zeros_like(offset)
        super().__init__(size=size, **kwargs)
        self._compile()
        self._meta: Batch

    def _compile(self) -> None:
        lens = last = index = np.array([0])
        offset = np.array([0, 1])
        done = np.array([False, False])
        _prev_index(index, offset, done, last, lens)
        _next_index(index, offset, done, last, lens)

    def __len__(self) -> int:
        return int(self._lengths.sum())

    def reset(self, keep_statistics: bool = False) -> None:
        self.last_index = self._offset.copy()
        self._lengths = np.zeros_like(self._offset)
        for buf in self.buffers:
            buf.reset(keep_statistics=keep_statistics)

    def _set_batch_for_children(self) -> None:
        for offset, buf in zip(self._offset, self.buffers):
            buf.set_batch(self._meta[offset:offset + buf.maxsize])

    def set_batch(self, batch: Batch) -> None:
        super().set_batch(batch)
        self._set_batch_for_children()

    def unfinished_index(self) -> np.ndarray:
        return np.concatenate(
            [
                buf.unfinished_index() + offset
                for offset, buf in zip(self._offset, self.buffers)
            ]
        )

    def prev(self, index: Union[int, np.ndarray]) -> np.ndarray:
        if isinstance(index, (list, np.ndarray)):
            return _prev_index(
                np.asarray(index), self._extend_offset, self.done, self.last_index,
                self._lengths
            )
        else:
            return _prev_index(
                np.array([index]), self._extend_offset, self.done, self.last_index,
                self._lengths
            )[0]

    def next(self, index: Union[int, np.ndarray]) -> np.ndarray:
        if isinstance(index, (list, np.ndarray)):
            return _next_index(
                np.asarray(index), self._extend_offset, self.done, self.last_index,
                self._lengths
            )
        else:
            return _next_index(
                np.array([index]), self._extend_offset, self.done, self.last_index,
                self._lengths
            )[0]

    def update(self, buffer: ReplayBuffer) -> np.ndarray:
        """The ReplayBufferManager cannot be updated by any buffer."""
        raise NotImplementedError

    def add(
        self,
        batch: Batch,
        buffer_ids: Optional[Union[np.ndarray, List[int]]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Add a batch of data into ReplayBufferManager.

        Each of the data's length (first dimension) must equal to the length of
        buffer_ids. By default buffer_ids is [0, 1, ..., buffer_num - 1].

        Return (current_index, episode_reward, episode_length, episode_start_index). If
        the episode is not finished, the return value of episode_length and
        episode_reward is 0.
        """
        # preprocess batch
        new_batch = Batch()
        for key in set(self._reserved_keys).intersection(batch.keys()):
            new_batch.__dict__[key] = batch[key]
        batch = new_batch
        assert set(["obs", "act", "rew", "done"]).issubset(batch.keys())
        if self._save_only_last_obs:
            batch.obs = batch.obs[:, -1]
        if not self._save_obs_next:
            batch.pop("obs_next", None)
        elif self._save_only_last_obs:
            batch.obs_next = batch.obs_next[:, -1]
        # get index
        if buffer_ids is None:
            buffer_ids = np.arange(self.buffer_num)
        ptrs, ep_lens, ep_rews, ep_idxs = [], [], [], []
        for batch_idx, buffer_id in enumerate(buffer_ids):
            ptr, ep_rew, ep_len, ep_idx = self.buffers[buffer_id]._add_index(
                batch.rew[batch_idx], batch.done[batch_idx]
            )
            ptrs.append(ptr + self._offset[buffer_id])
            ep_lens.append(ep_len)
            ep_rews.append(ep_rew)
            ep_idxs.append(ep_idx + self._offset[buffer_id])
            self.last_index[buffer_id] = ptr + self._offset[buffer_id]
            self._lengths[buffer_id] = len(self.buffers[buffer_id])
        ptrs = np.array(ptrs)
        try:
            self._meta[ptrs] = batch
        except ValueError:
            batch.rew = batch.rew.astype(float)
            batch.done = batch.done.astype(bool)
            if self._meta.is_empty():
                self._meta = _create_value(  # type: ignore
                    batch, self.maxsize, stack=False)
            else:  # dynamic key pops up in batch
                _alloc_by_keys_diff(self._meta, batch, self.maxsize, False)
            self._set_batch_for_children()
            self._meta[ptrs] = batch
        return ptrs, np.array(ep_rews), np.array(ep_lens), np.array(ep_idxs)

    def sample_indices(self, batch_size: int) -> np.ndarray:
        if batch_size < 0:
            return np.array([], int)
        if self._sample_avail and self.stack_num > 1:
            all_indices = np.concatenate(
                [
                    buf.sample_indices(0) + offset
                    for offset, buf in zip(self._offset, self.buffers)
                ]
            )
            if batch_size == 0:
                return all_indices
            else:
                return np.random.choice(all_indices, batch_size)
        if batch_size == 0:  # get all available indices
            sample_num = np.zeros(self.buffer_num, int)
        else:
            buffer_idx = np.random.choice(
                self.buffer_num, batch_size, p=self._lengths / self._lengths.sum()
            )
            sample_num = np.bincount(buffer_idx, minlength=self.buffer_num)
            # avoid batch_size > 0 and sample_num == 0 -> get child's all data
            sample_num[sample_num == 0] = -1

        return np.concatenate(
            [
                buf.sample_indices(bsz) + offset
                for offset, buf, bsz in zip(self._offset, self.buffers, sample_num)
            ]
        )


class PrioritizedReplayBufferManager(PrioritizedReplayBuffer, ReplayBufferManager):
    """PrioritizedReplayBufferManager contains a list of PrioritizedReplayBuffer with \
    exactly the same configuration.

    These replay buffers have contiguous memory layout, and the storage space each
    buffer has is a shallow copy of the topmost memory.

    :param buffer_list: a list of PrioritizedReplayBuffer needed to be handled.

    .. seealso::

        Please refer to :class:`~tianshou.data.ReplayBuffer` for other APIs' usage.
    """

    def __init__(self, buffer_list: Sequence[PrioritizedReplayBuffer]) -> None:
        ReplayBufferManager.__init__(self, buffer_list)  # type: ignore
        kwargs = buffer_list[0].options
        for buf in buffer_list:
            del buf.weight
        PrioritizedReplayBuffer.__init__(self, self.maxsize, **kwargs)


@njit
def _prev_index(
    index: np.ndarray,
    offset: np.ndarray,
    done: np.ndarray,
    last_index: np.ndarray,
    lengths: np.ndarray,
) -> np.ndarray:
    index = index % offset[-1]
    prev_index = np.zeros_like(index)
    for start, end, cur_len, last in zip(offset[:-1], offset[1:], lengths, last_index):
        mask = (start <= index) & (index < end)
        cur_len = max(1, cur_len)
        if np.sum(mask) > 0:
            subind = index[mask]
            subind = (subind - start - 1) % cur_len
            end_flag = done[subind + start] | (subind + start == last)
            prev_index[mask] = (subind + end_flag) % cur_len + start
    return prev_index


@njit
def _next_index(
    index: np.ndarray,
    offset: np.ndarray,
    done: np.ndarray,
    last_index: np.ndarray,
    lengths: np.ndarray,
) -> np.ndarray:
    index = index % offset[-1]
    next_index = np.zeros_like(index)
    for start, end, cur_len, last in zip(offset[:-1], offset[1:], lengths, last_index):
        mask = (start <= index) & (index < end)
        cur_len = max(1, cur_len)
        if np.sum(mask) > 0:
            subind = index[mask]
            end_flag = done[subind] | (subind == last)
            next_index[mask] = (subind - start + 1 - end_flag) % cur_len + start
    return next_index
