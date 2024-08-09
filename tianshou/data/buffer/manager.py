from collections.abc import Sequence
from typing import Union, cast

import numpy as np
from numba import njit
from overrides import override

from tianshou.data import Batch, HERReplayBuffer, PrioritizedReplayBuffer, ReplayBuffer
from tianshou.data.batch import alloc_by_keys_diff, create_value
from tianshou.data.types import RolloutBatchProtocol


class ReplayBufferManager(ReplayBuffer):
    """ReplayBufferManager contains a list of ReplayBuffer with exactly the same configuration.

    These replay buffers have contiguous memory layout, and the storage space each
    buffer has is a shallow copy of the topmost memory.

    :param buffer_list: a list of ReplayBuffer needed to be handled.

    .. seealso::

        Please refer to :class:`~tianshou.data.ReplayBuffer` for other APIs' usage.
    """

    def __init__(self, buffer_list: list[ReplayBuffer] | list[HERReplayBuffer]) -> None:
        self.buffer_num = len(buffer_list)
        self.buffers = np.array(buffer_list, dtype=object)
        last_index: list[int] = []
        offset, size = [], 0
        buffer_type = type(self.buffers[0])
        kwargs = self.buffers[0].options
        for buf in self.buffers:
            buf = cast(ReplayBuffer, buf)
            assert len(buf._meta.get_keys()) == 0
            assert isinstance(buf, buffer_type)
            assert buf.options == kwargs
            offset.append(size)
            if len(buf.last_index) != 1:
                raise ValueError(
                    f"{self.__class__.__name__} only supports buffers with a single index "
                    f"(non-vector buffers), but got {last_index=}. "
                    f"Did you try to use a {self.__class__.__name__} within a {self.__class__.__name__}?",
                )
            last_index.append(size + buf.last_index[0])
            size += buf.maxsize
        super().__init__(size=size, **kwargs)
        self._offset = np.array(offset)
        self._extend_offset = np.array([*offset, size])
        self._lengths = np.zeros_like(offset)
        self.last_index = np.array(last_index)
        self._compile()
        self._meta: RolloutBatchProtocol

    @property
    @override
    def subbuffer_edges(self) -> np.ndarray:
        return self._extend_offset

    def _compile(self) -> None:
        lens = last = index = np.array([0])
        offset = np.array([0, 1])
        done = np.array([False, False])
        _prev_index(index, offset, done, last, lens)
        _next_index(index, offset, done, last, lens)

    def __len__(self) -> int:
        return int(self._lengths.sum())

    def reset(self, keep_statistics: bool = False) -> None:
        # keep in sync with init!
        self.last_index = self._offset.copy()
        self._lengths = np.zeros_like(self._offset)
        for buf in self.buffers:
            buf.reset(keep_statistics=keep_statistics)

    def _set_batch_for_children(self) -> None:
        for offset, buf in zip(self._offset, self.buffers, strict=True):
            buf.set_batch(self._meta[offset : offset + buf.maxsize])

    def set_batch(self, batch: RolloutBatchProtocol) -> None:
        super().set_batch(batch)
        self._set_batch_for_children()

    def unfinished_index(self) -> np.ndarray:
        return np.concatenate(
            [
                buf.unfinished_index() + offset
                for offset, buf in zip(self._offset, self.buffers, strict=True)
            ],
        )

    def prev(self, index: int | np.ndarray) -> np.ndarray:
        if isinstance(index, list | np.ndarray):
            return _prev_index(
                np.asarray(index),
                self._extend_offset,
                self.done,
                self.last_index,
                self._lengths,
            )
        return _prev_index(
            np.array([index]),
            self._extend_offset,
            self.done,
            self.last_index,
            self._lengths,
        )[0]

    def next(self, index: int | np.ndarray) -> np.ndarray:
        if isinstance(index, list | np.ndarray):
            return _next_index(
                np.asarray(index),
                self._extend_offset,
                self.done,
                self.last_index,
                self._lengths,
            )
        return _next_index(
            np.array([index]),
            self._extend_offset,
            self.done,
            self.last_index,
            self._lengths,
        )[0]

    def update(self, buffer: ReplayBuffer) -> np.ndarray:
        """The ReplayBufferManager cannot be updated by any buffer."""
        raise NotImplementedError

    def add(
        self,
        batch: RolloutBatchProtocol,
        buffer_ids: np.ndarray | list[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Add a batch of data into ReplayBufferManager.

        Each of the data's length (first dimension) must equal to the length of
        buffer_ids. By default buffer_ids is [0, 1, ..., buffer_num - 1].

        Return (current_index, episode_reward, episode_length, episode_start_index). If
        the episode is not finished, the return value of episode_length and
        episode_reward is 0.
        """
        # preprocess batch
        new_batch = Batch()
        for key in set(self._reserved_keys).intersection(batch.get_keys()):
            new_batch.__dict__[key] = batch[key]
        batch = new_batch
        batch.__dict__["done"] = np.logical_or(batch.terminated, batch.truncated)
        assert {"obs", "act", "rew", "terminated", "truncated", "done"}.issubset(batch.get_keys())
        if self._save_only_last_obs:
            batch.obs = batch.obs[:, -1]
        if not self._save_obs_next:
            batch.pop("obs_next", None)
        elif self._save_only_last_obs:
            batch.obs_next = batch.obs_next[:, -1]
        # get index
        if buffer_ids is None:
            buffer_ids = np.arange(self.buffer_num)
        insertion_indxS, ep_lens, ep_returns, ep_idxs = [], [], [], []
        for batch_idx, buffer_id in enumerate(buffer_ids):
            # TODO: don't access private method!
            insertion_index, ep_return, ep_len, ep_start_idx = self.buffers[
                buffer_id
            ]._update_state_pre_add(
                batch.rew[batch_idx],
                batch.done[batch_idx],
            )
            offset_insertion_idx = insertion_index + self._offset[buffer_id]
            offset_ep_start_idx = ep_start_idx + self._offset[buffer_id]
            insertion_indxS.append(offset_insertion_idx)
            ep_lens.append(ep_len)
            ep_returns.append(ep_return)
            ep_idxs.append(offset_ep_start_idx)
            self.last_index[buffer_id] = insertion_index + self._offset[buffer_id]
            self._lengths[buffer_id] = len(self.buffers[buffer_id])
        insertion_indxS = np.array(insertion_indxS)
        try:
            self._meta[insertion_indxS] = batch
        # TODO: don't do this!
        except ValueError:
            batch.rew = batch.rew.astype(float)
            batch.done = batch.done.astype(bool)
            batch.terminated = batch.terminated.astype(bool)
            batch.truncated = batch.truncated.astype(bool)
            if len(self._meta.get_keys()) == 0:
                self._meta = create_value(batch, self.maxsize, stack=False)  # type: ignore
            else:  # dynamic key pops up in batch
                alloc_by_keys_diff(self._meta, batch, self.maxsize, False)
            self._set_batch_for_children()
            self._meta[insertion_indxS] = batch
        return insertion_indxS, np.array(ep_returns), np.array(ep_lens), np.array(ep_idxs)

    def sample_indices(self, batch_size: int | None) -> np.ndarray:
        # TODO: simplify this code
        if batch_size is not None and batch_size < 0:
            # TODO: raise error instead?
            return np.array([], int)
        if self._sample_avail and self.stack_num > 1:
            all_indices = np.concatenate(
                [
                    buf.sample_indices(0) + offset
                    for offset, buf in zip(self._offset, self.buffers, strict=True)
                ],
            )
            if batch_size == 0:
                return all_indices
            if batch_size is None:
                batch_size = len(all_indices)
            return np.random.choice(all_indices, batch_size)
        if batch_size == 0 or batch_size is None:  # get all available indices
            sample_num = np.zeros(self.buffer_num, int)
        else:
            buffer_idx = np.random.choice(
                self.buffer_num,
                batch_size,
                p=self._lengths / self._lengths.sum(),
            )
            sample_num = np.bincount(buffer_idx, minlength=self.buffer_num)
            # avoid batch_size > 0 and sample_num == 0 -> get child's all data
            sample_num[sample_num == 0] = -1

        return np.concatenate(
            [
                buf.sample_indices(int(bsz)) + offset
                for offset, buf, bsz in zip(self._offset, self.buffers, sample_num, strict=True)
            ],
        )


# TODO: unintuitively, the order of inheritance has to stay this way for tests to pass
#  As also described in the todo below, this is a bad design and should be refactored
class PrioritizedReplayBufferManager(PrioritizedReplayBuffer, ReplayBufferManager):
    """PrioritizedReplayBufferManager contains a list of PrioritizedReplayBuffer with exactly the same configuration.

    These replay buffers have contiguous memory layout, and the storage space each
    buffer has is a shallow copy of the topmost memory.

    :param buffer_list: a list of PrioritizedReplayBuffer needed to be handled.

    .. seealso::

        Please refer to :class:`~tianshou.data.ReplayBuffer` for other APIs' usage.
    """

    def __init__(self, buffer_list: Sequence[PrioritizedReplayBuffer]) -> None:
        ReplayBufferManager.__init__(self, buffer_list)  # type: ignore
        # last_index = copy(self.last_index)
        kwargs = buffer_list[0].options
        last_index_from_buffer_manager = self.last_index

        for buf in buffer_list:
            del buf.weight
        PrioritizedReplayBuffer.__init__(self, self.maxsize, **kwargs)

        # TODO: the line below is needed since we now set the last_index of the manager in init
        #  (previously it was only set in reset), and it clashes with multiple inheritance
        #  Initializing the ReplayBufferManager after the PrioritizedReplayBuffer would be a better solution,
        #  but it currently leads to infinite recursion. This kind of multiple inheritance with overlapping
        #  interfaces is evil and we should get rid of it
        self.last_index = last_index_from_buffer_manager


class HERReplayBufferManager(ReplayBufferManager):
    """HERReplayBufferManager contains a list of HERReplayBuffer with exactly the same configuration.

    These replay buffers have contiguous memory layout, and the storage space each
    buffer has is a shallow copy of the topmost memory.

    :param buffer_list: a list of HERReplayBuffer needed to be handled.

    .. seealso::

        Please refer to :class:`~tianshou.data.ReplayBuffer` for other APIs' usage.
    """

    def __init__(self, buffer_list: list[HERReplayBuffer]) -> None:
        super().__init__(buffer_list)

    def _restore_cache(self) -> None:
        for buf in self.buffers:
            buf._restore_cache()

    def save_hdf5(self, path: str, compression: str | None = None) -> None:
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
        buffer_ids: np.ndarray | list[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._restore_cache()
        return super().add(batch, buffer_ids)


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
    # disable B905 until strict=True in zip is implemented in numba
    # https://github.com/numba/numba/issues/8943
    for start, end, cur_len, last in zip(  # noqa: B905
        offset[:-1],
        offset[1:],
        lengths,
        last_index,
    ):
        mask = (start <= index) & (index < end)
        correct_cur_len = max(1, cur_len)
        if np.sum(mask) > 0:
            subind = index[mask]
            subind = (subind - start - 1) % correct_cur_len
            end_flag = done[subind + start] | (subind + start == last)
            prev_index[mask] = (subind + end_flag) % correct_cur_len + start
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
    # disable B905 until strict=True in zip is implemented in numba
    # https://github.com/numba/numba/issues/8943
    for start, end, cur_len, last in zip(  # noqa: B905
        offset[:-1],
        offset[1:],
        lengths,
        last_index,
    ):
        mask = (start <= index) & (index < end)
        correct_cur_len = max(1, cur_len)
        if np.sum(mask) > 0:
            subind = index[mask]
            end_flag = done[subind] | (subind == last)
            next_index[mask] = (subind - start + 1 - end_flag) % correct_cur_len + start
    return next_index
