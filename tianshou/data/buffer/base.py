from collections.abc import Sequence
from typing import Any, ClassVar, Self, TypeVar, cast

import h5py
import numpy as np

from tianshou.data import Batch
from tianshou.data.batch import (
    IndexType,
    alloc_by_keys_diff,
    create_value,
    log,
)
from tianshou.data.types import RolloutBatchProtocol
from tianshou.data.utils.converter import from_hdf5, to_hdf5

TBuffer = TypeVar("TBuffer", bound="ReplayBuffer")


class MalformedBufferError(RuntimeError):
    pass


class ReplayBuffer:
    """:class:`~tianshou.data.ReplayBuffer` stores data generated from interaction between the policy and environment.

    ReplayBuffer can be considered as a specialized form (or management) of Batch. It
    stores all the data in a batch with circular-queue style.

    For the example usage of ReplayBuffer, please check out Section Buffer in
    :doc:`/01_tutorials/01_concepts`.

    :param size: the maximum size of replay buffer.
    :param stack_num: the frame-stack sampling argument, should be greater than or
        equal to 1. Default to 1 (no stacking).
    :param ignore_obs_next: whether to not store obs_next.
    :param save_only_last_obs: only save the last obs/obs_next when it has a shape
        of (timestep, ...) because of temporal stacking.
    :param sample_avail: whether to sample only available indices
        when using the frame-stack sampling method.
    """

    _reserved_keys = (
        "obs",
        "act",
        "rew",
        "terminated",
        "truncated",
        "done",
        "obs_next",
        "info",
        "policy",
    )
    _input_keys = (
        "obs",
        "act",
        "rew",
        "terminated",
        "truncated",
        "obs_next",
        "info",
        "policy",
    )
    _required_keys_for_add: ClassVar[set[str]] = {
        "obs",
        "act",
        "rew",
        "terminated",
        "truncated",
        "done",
    }

    def __init__(
        self,
        size: int,
        stack_num: int = 1,
        ignore_obs_next: bool = False,
        save_only_last_obs: bool = False,
        sample_avail: bool = False,
        **kwargs: Any,  # otherwise PrioritizedVectorReplayBuffer will cause TypeError
    ) -> None:
        # TODO: why do we need this? Just for readout?
        self.options: dict[str, Any] = {
            "stack_num": stack_num,
            "ignore_obs_next": ignore_obs_next,
            "save_only_last_obs": save_only_last_obs,
            "sample_avail": sample_avail,
        }
        super().__init__()
        self.maxsize = int(size)
        assert stack_num > 0, "stack_num should be greater than 0"
        self.stack_num = stack_num
        self._indices = np.arange(size)
        # TODO: remove double negation and different name
        self._save_obs_next = not ignore_obs_next
        self._save_only_last_obs = save_only_last_obs
        self._sample_avail = sample_avail
        self._meta = cast(RolloutBatchProtocol, Batch())

        # Keep in sync with reset!
        self.last_index = np.array([0])
        self._insertion_idx = self._size = 0
        self._ep_return, self._ep_len, self._ep_start_idx = 0.0, 0, 0

    @property
    def subbuffer_edges(self) -> np.ndarray:
        """Edges of contained buffers, mostly needed as part of the VectorReplayBuffer interface.

        For the standard ReplayBuffer it is always [0, maxsize]. Transitions can be added
        to the buffer indefinitely, and one episode can "go over the edge". Having the edges
        available is useful for fishing out whole episodes from the buffer and for input validation.
        """
        return np.array([0, self.maxsize], dtype=int)

    def _get_start_stop_tuples_for_edge_crossing_interval(
        self,
        start: int,
        stop: int,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Assumes that stop < start and retrieves tuples corresponding to the two
        slices that determine the interval within the buffer.

        Example:
        -------
        >>> list(self.subbuffer_edges) == [0, 5, 10]
        >>> start = 4
        >>> stop = 2
        >>> self._get_start_stop_tuples_for_edge_crossing_interval(start, stop)
        ((4, 5), (0, 2))

        The buffer sliced from 4 to 5 and then from 0 to 2 will contain the transitions
        corresponding to the provided start and stop values.
        """
        if stop >= start:
            raise ValueError(
                f"Expected stop < start, but got {start=}, {stop=}. "
                f"For stop larger than start this method should never be called, "
                f"and stop=start should never occur. This can occur either due to an implementation error, "
                f"or due a bad configuration of the buffer that resulted in a single episode being so long that "
                f"it completely filled a subbuffer (of size len(buffer)/degree_of_vectorization). "
                f"Consider either shortening the episode, increasing the size of the buffer, or decreasing the "
                f"degree of vectorization.",
            )
        subbuffer_edges = cast(Sequence[int], self.subbuffer_edges)

        edge_after_start_idx = int(np.searchsorted(subbuffer_edges, start, side="left"))
        """This is the crossed edge"""

        if edge_after_start_idx == 0:
            raise ValueError(
                f"The start value should be larger than the first edge, but got {start=}, {subbuffer_edges[1]=}.",
            )
        edge_after_start = subbuffer_edges[edge_after_start_idx]
        edge_before_stop = subbuffer_edges[edge_after_start_idx - 1]
        """It's the edge before the crossed edge"""

        if edge_before_stop >= stop:
            raise ValueError(
                f"The edge before the crossed edge should be smaller than the stop, but got {edge_before_stop=}, {stop=}.",
            )
        return (start, edge_after_start), (edge_before_stop, stop)

    def get_buffer_indices(self, start: int, stop: int) -> np.ndarray:
        """Get the indices of the transitions in the buffer between start and stop.

        The special thing about this is that stop may actually be smaller than start,
        since one often is interested in a sequence of transitions that goes over a subbuffer edge.

        The main use case for this method is to retrieve an episode from the buffer, in which case
        start is the index of the first transition in the episode and stop is the index where `done` is True + 1.
        This can be done with the following code:

        .. code-block:: python

            episode_indices = buffer.get_buffer_indices(episode_start_index, episode_done_index + 1)
            episode = buffer[episode_indices]

        Even when `start` is smaller than `stop`, it will be validated that they are in the same subbuffer.

        Example:
        --------
        >>> list(buffer.subbuffer_edges) == [0, 5, 10]
        >>> buffer.get_buffer_indices(start=2, stop=4)
        [2, 3]
        >>> buffer.get_buffer_indices(start=4, stop=2)
        [4, 0, 1]
        >>> buffer.get_buffer_indices(start=8, stop=7)
        [8, 9, 5, 6]
        >>> buffer.get_buffer_indices(start=1, stop=6)
        ValueError: Start and stop indices must be within the same subbuffer.
        >>> buffer.get_buffer_indices(start=8, stop=1)
        ValueError: Start and stop indices must be within the same subbuffer.

        :param start: The start index of the interval.
        :param stop: The stop index of the interval.
        :return: The indices of the transitions in the buffer between start and stop.
        """
        start_left_edge = np.searchsorted(self.subbuffer_edges, start, side="right") - 1
        stop_left_edge = np.searchsorted(self.subbuffer_edges, stop - 1, side="right") - 1
        if start_left_edge != stop_left_edge:
            raise ValueError(
                f"Start and stop indices must be within the same subbuffer. "
                f"Got {start=} in subbuffer edge {start_left_edge} and {stop=} in subbuffer edge {stop_left_edge}.",
            )
        if stop > start:
            return np.arange(start, stop, dtype=int)
        else:
            (start, upper_edge), (
                lower_edge,
                stop,
            ) = self._get_start_stop_tuples_for_edge_crossing_interval(
                start,
                stop,
            )
            log.debug(f"{start=}, {upper_edge=}, {lower_edge=}, {stop=}")
            return np.concatenate(
                (np.arange(start, upper_edge, dtype=int), np.arange(lower_edge, stop, dtype=int)),
            )

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        wrapped_batch_repr = self._meta.__repr__()[len(self._meta.__class__.__name__) :]
        return self.__class__.__name__ + wrapped_batch_repr

    def __getattr__(self, key: str) -> Any:
        try:
            return self._meta[key]
        except KeyError as exception:
            raise AttributeError from exception

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def __setattr__(self, key: str, value: Any) -> None:
        assert key not in self._reserved_keys, f"key '{key}' is reserved and cannot be assigned"
        super().__setattr__(key, value)

    def save_hdf5(self, path: str, compression: str | None = None) -> None:
        """Save replay buffer to HDF5 file."""
        with h5py.File(path, "w") as f:
            to_hdf5(self.__dict__, f, compression=compression)

    @classmethod
    def load_hdf5(cls, path: str, device: str | None = None) -> Self:
        """Load replay buffer from HDF5 file."""
        with h5py.File(path, "r") as f:
            buf = cls.__new__(cls)
            buf.__setstate__(from_hdf5(f, device=device))  # type: ignore
        return buf

    @classmethod
    def from_data(
        cls,
        obs: h5py.Dataset,
        act: h5py.Dataset,
        rew: h5py.Dataset,
        terminated: h5py.Dataset,
        truncated: h5py.Dataset,
        done: h5py.Dataset,
        obs_next: h5py.Dataset,
    ) -> Self:
        size = len(obs)
        assert all(
            len(dset) == size for dset in [obs, act, rew, terminated, truncated, done, obs_next]
        ), "Lengths of all hdf5 datasets need to be equal."
        buf = cls(size)
        if size == 0:
            return buf
        batch = Batch(
            obs=obs,
            act=act,
            rew=rew,
            terminated=terminated,
            truncated=truncated,
            done=done,
            obs_next=obs_next,
        )
        batch = cast(RolloutBatchProtocol, batch)
        buf.set_batch(batch)
        buf._size = size
        return buf

    def reset(self, keep_statistics: bool = False) -> None:
        """Clear all the data in replay buffer and episode statistics."""
        # Keep in sync with init!
        self.last_index = np.array([0])
        self._insertion_idx = self._size = self._ep_start_idx = 0
        if not keep_statistics:
            self._ep_return, self._ep_len = 0.0, 0

    # TODO: is this method really necessary? It's kinda dangerous, can accidentally
    #  remove all references to collected data
    def set_batch(self, batch: RolloutBatchProtocol) -> None:
        """Manually choose the batch you want the ReplayBuffer to manage."""
        assert len(batch) == self.maxsize and set(batch.get_keys()).issubset(
            self._reserved_keys,
        ), "Input batch doesn't meet ReplayBuffer's data form requirement."
        self._meta = batch

    def unfinished_index(self) -> np.ndarray:
        """Return the index of unfinished episode."""
        last = (self._insertion_idx - 1) % self._size if self._size else 0
        return np.array([last] if not self.done[last] and self._size else [], int)

    def prev(self, index: int | np.ndarray) -> np.ndarray:
        """Return the index of previous transition.

        The index won't be modified if it is the beginning of an episode.
        """
        index = (index - 1) % self._size
        end_flag = self.done[index] | (index == self.last_index[0])
        return (index + end_flag) % self._size

    def next(self, index: int | np.ndarray) -> np.ndarray:
        """Return the index of next transition.

        The index won't be modified if it is the end of an episode.
        """
        end_flag = self.done[index] | (index == self.last_index[0])
        return (index + (1 - end_flag)) % self._size

    def update(self, buffer: "ReplayBuffer") -> np.ndarray:
        """Move the data from the given buffer to current buffer.

        Return the updated indices. If update fails, return an empty array.
        """
        if len(buffer) == 0 or self.maxsize == 0:
            return np.array([], int)
        stack_num, buffer.stack_num = buffer.stack_num, 1
        from_indices = buffer.sample_indices(0)  # get all available indices
        buffer.stack_num = stack_num
        if len(from_indices) == 0:
            return np.array([], int)
        updated_indices = []
        for _ in range(len(from_indices)):
            updated_indices.append(self._insertion_idx)
            self.last_index[0] = self._insertion_idx
            self._insertion_idx = (self._insertion_idx + 1) % self.maxsize
            self._size = min(self._size + 1, self.maxsize)
        updated_indices = np.array(updated_indices)
        if len(self._meta.get_keys()) == 0:
            self._meta = create_value(buffer._meta, self.maxsize, stack=False)  # type: ignore
        self._meta[updated_indices] = buffer._meta[from_indices]
        return updated_indices

    def _update_state_pre_add(
        self,
        rew: float | np.ndarray,
        done: bool,
    ) -> tuple[int, float, int, int]:
        """Update the buffer's state before adding one data batch.

        Updates the `_size` and `_insertion_idx`, adds the reward and len
        internally maintained `_ep_len` and `_ep_return`. If `done` is `True`,
        will reset `_ep_len` and `_ep_return` to zero, and set `_ep_start_idx` to
        `_insertion_idx`

        Returns a tuple with:
        0. the index at which to insert the next transition,
        1. the episode len (if done=True, otherwise 0)
        2. the episode return (if done=True, otherwise 0)
        3. the episode start index.
        """
        self.last_index[0] = cur_insertion_idx = self._insertion_idx
        self._size = min(self._size + 1, self.maxsize)
        self._insertion_idx = (self._insertion_idx + 1) % self.maxsize

        self._ep_return += rew  # type: ignore
        self._ep_len += 1

        if self._ep_start_idx > len(self):
            raise MalformedBufferError(
                f"Encountered a starting index {self._ep_start_idx} that is outside "
                f"the currently available samples {len(self)=}. "
                f"The buffer is malformed. This might be caused by a bug or by manual modifications of the buffer "
                f"by users.",
            )

        # return 0 for unfinished episodes
        if done:
            ep_return = self._ep_return
            ep_len = self._ep_len
        else:
            if isinstance(self._ep_return, np.ndarray):  # type: ignore[unreachable]
                # TODO: fix this!
                log.error(  # type: ignore[unreachable]
                    f"ep_return should be a scalar but is a numpy array: {self._ep_return.shape=}. "
                    "This doesn't make sense for a ReplayBuffer, but currently tests of CachedReplayBuffer require"
                    "this behavior for some reason. Should be fixed ASAP! "
                    "Returning an array of zeros instead of a scalar zero.",
                )
            ep_return = np.zeros_like(self._ep_return)  # type: ignore
            ep_len = 0

        result = cur_insertion_idx, ep_return, ep_len, self._ep_start_idx

        if done:
            # prepare for next episode collection
            # set return and len to zero, set start idx to next insertion idx
            self._ep_return, self._ep_len, self._ep_start_idx = 0.0, 0, self._insertion_idx
        return result

    def add(
        self,
        batch: RolloutBatchProtocol,
        buffer_ids: np.ndarray | list[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Add a batch of data into replay buffer.

        :param batch: the input data batch. "obs", "act", "rew",
            "terminated", "truncated" are required keys.
        :param buffer_ids: id's of subbuffers, allowed here to be consistent with classes similar to
            :class:`~tianshou.data.buffer.vecbuf.VectorReplayBuffer`. Since the `ReplayBuffer`
            has a single subbuffer, if this is not None, it must be a single element with value 0.
            In that case, the batch is expected to have the shape (1, len(data)).
            Failure to adhere to this will result in a `ValueError`.

        Return `(current_index, episode_return, episode_length, episode_start_index)`. If
        the episode is not finished, the return value of episode_length and
        episode_reward is 0.
        """
        # preprocess and copy batch into a new Batch object to avoid mutating the input
        # TODO: can't we just copy? Why do we need to rely on setting inside __dict__?
        new_batch = Batch()
        for key in batch.get_keys():
            new_batch.__dict__[key] = batch[key]
        batch = new_batch
        batch.__dict__["done"] = np.logical_or(batch.terminated, batch.truncated)

        # has to be done after preprocess batch
        if not self._required_keys_for_add.issubset(
            batch.get_keys(),
        ):
            raise ValueError(
                f"Input batch must have the following keys: {self._required_keys_for_add}",
            )

        batch_is_stacked = False
        """True when instead of passing a batch of shape (len(data)), a batch of shape (1, len(data)) is passed."""

        if buffer_ids is not None:
            if len(buffer_ids) != 1 and buffer_ids[0] != 0:
                raise ValueError(
                    "If `buffer_ids` is not None, it must be a single element with value 0 for the non-vectorized `ReplayBuffer`. "
                    f"Got {buffer_ids=}.",
                )
            if len(batch) != 1:
                raise ValueError(
                    f"If `buffer_ids` is not None, the batch must have the shape (1, len(data)) but got {len(batch)=}.",
                )
            batch_is_stacked = True

        # block dealing with exotic options that are currently only used for atari, see various TODOs about that
        # These options have interactions with the case when buffer_ids is not None
        if self._save_only_last_obs:
            batch.obs = batch.obs[:, -1] if batch_is_stacked else batch.obs[-1]
        if not self._save_obs_next:
            batch.pop("obs_next", None)
        elif self._save_only_last_obs:
            batch.obs_next = batch.obs_next[:, -1] if batch_is_stacked else batch.obs_next[-1]

        if batch_is_stacked:
            rew, done = batch.rew[0], batch.done[0]
        else:
            rew, done = batch.rew, batch.done
        insertion_idx, ep_return, ep_len, ep_start_idx = (
            np.array([x]) for x in self._update_state_pre_add(rew, done)
        )

        # TODO: improve this, don'r rely on try-except, instead process the batch if needed
        try:
            self._meta[insertion_idx] = batch
        except ValueError:
            stack = not batch_is_stacked
            batch.rew = batch.rew.astype(float)
            batch.done = batch.done.astype(bool)
            batch.terminated = batch.terminated.astype(bool)
            batch.truncated = batch.truncated.astype(bool)
            if len(self._meta.get_keys()) == 0:
                self._meta = create_value(batch, self.maxsize, stack)  # type: ignore
            else:  # dynamic key pops up in batch
                alloc_by_keys_diff(self._meta, batch, self.maxsize, stack)
            self._meta[insertion_idx] = batch
        return insertion_idx, ep_return, ep_len, ep_start_idx

    def sample_indices(self, batch_size: int | None) -> np.ndarray:
        """Get a random sample of index with size = batch_size.

        Return all available indices in the buffer if batch_size is 0; return an empty
        numpy array if batch_size < 0 or no available index can be sampled.

        :param batch_size: the number of indices to be sampled. If None, it will be set
            to the length of the buffer (i.e. return all available indices in a
            random order).
        """
        if batch_size is None:
            batch_size = len(self)
        if self.stack_num == 1 or not self._sample_avail:  # most often case
            if batch_size > 0:
                return np.random.choice(self._size, batch_size)
            # TODO: is this behavior really desired?
            if batch_size == 0:  # construct current available indices
                return np.concatenate(
                    [np.arange(self._insertion_idx, self._size), np.arange(self._insertion_idx)],
                )
            return np.array([], int)
        # TODO: raise error on negative batch_size instead?
        if batch_size < 0:
            return np.array([], int)
        # TODO: simplify this code - shouldn't have such a large if-else
        #  with many returns for handling different stack nums.
        #  It is also not clear whether this is really necessary - frame stacking usually is handled
        #  by environment wrappers (e.g. FrameStack) and not by the replay buffer.
        all_indices = prev_indices = np.concatenate(
            [np.arange(self._insertion_idx, self._size), np.arange(self._insertion_idx)],
        )
        for _ in range(self.stack_num - 2):
            prev_indices = self.prev(prev_indices)
        all_indices = all_indices[prev_indices != self.prev(prev_indices)]
        if batch_size > 0:
            return np.random.choice(all_indices, batch_size)
        return all_indices

    def sample(self, batch_size: int | None) -> tuple[RolloutBatchProtocol, np.ndarray]:
        """Get a random sample from buffer with size = batch_size.

        Return all the data in the buffer if batch_size is 0.

        :return: Sample data and its corresponding index inside the buffer.
        """
        indices = self.sample_indices(batch_size)
        return self[indices], indices

    def get(
        self,
        index: int | list[int] | np.ndarray,
        key: str,
        default_value: Any = None,
        # TODO 1: this is only here because of atari, it should never be needed (can be solved with index)
        #  and should be removed
        # TODO 2: does something entirely different from getitem
        # TODO 3: key should not be required
        stack_num: int | None = None,
    ) -> Batch | np.ndarray:
        """Return the stacked result.

        E.g., if you set ``key = "obs", stack_num = 4, index = t``, it returns the
        stacked result as ``[obs[t-3], obs[t-2], obs[t-1], obs[t]]``.

        :param index: the index for getting stacked data.
        :param key: the key to get, should be one of the reserved_keys.
        :param default_value: if the given key's data is not found and default_value is
            set, return this default_value.
        :param stack_num: Default to self.stack_num.
        """
        if key not in self._meta.get_keys() and default_value is not None:
            return default_value
        val = self._meta[key]
        if stack_num is None:
            stack_num = self.stack_num
        try:
            if stack_num == 1:  # the most common case
                return val[index]

            stack = list[Any]()
            indices = np.array(index) if isinstance(index, list) else index
            # NOTE: stack_num > 1, so the range is not empty and indices is turned into
            # np.ndarray by self.prev
            for _ in range(stack_num):
                stack = [val[indices], *stack]
                indices = self.prev(indices)
            indices = cast(np.ndarray, indices)
            if isinstance(val, Batch):
                return Batch.stack(stack, axis=indices.ndim)
            return np.stack(stack, axis=indices.ndim)

        except IndexError as exception:
            if not (isinstance(val, Batch) and len(val.keys()) == 0):
                raise exception  # val != Batch()
            return Batch()

    def __getitem__(self, index: IndexType) -> RolloutBatchProtocol:
        """Return a data batch: self[index].

        If stack_num is larger than 1, return the stacked obs and obs_next with shape
        (batch, len, ...).
        """
        # TODO: this is a seriously problematic hack leading to
        #  buffer[slice] != buffer[np.arange(slice.start, slice.stop)]
        #  Fix asap, high priority!!!
        if isinstance(index, slice):  # change slice to np array
            # buffer[:] will get all available data
            indices = (
                self.sample_indices(0)
                if index == slice(None)
                else self._indices[: len(self)][index]
            )
        else:
            indices = index  # type: ignore
        # raise KeyError first instead of AttributeError,
        # to support np.array([ReplayBuffer()])
        obs = self.get(indices, "obs")
        if self._save_obs_next:
            obs_next = self.get(indices, "obs_next", Batch())
        else:
            obs_next_indices = self.next(indices)
            obs_next = self.get(obs_next_indices, "obs", Batch())
        # TODO: don't do this
        batch_dict = {
            "obs": obs,
            "act": self.act[indices],
            "rew": self.rew[indices],
            "terminated": self.terminated[indices],
            "truncated": self.truncated[indices],
            "done": self.done[indices],
            "obs_next": obs_next,
            "info": self.get(indices, "info", Batch()),
            # TODO: what's the use of this key?
            "policy": self.get(indices, "policy", Batch()),
        }
        # TODO: don't do this, reduce complexity. Why such a big difference between what is returned
        #   and sub-batches of self._meta?
        missing_keys = set(self._meta.get_keys()) - set(self._input_keys)
        for key in missing_keys:
            batch_dict[key] = self._meta[key][indices]
        return cast(RolloutBatchProtocol, Batch(batch_dict))

    def set_array_at_key(
        self,
        seq: np.ndarray,
        key: str,
        index: IndexType | None = None,
        default_value: float | None = None,
    ) -> None:
        self._meta.set_array_at_key(seq, key, index, default_value)

    def hasnull(self) -> bool:
        return self[:].hasnull()

    def isnull(self) -> RolloutBatchProtocol:
        return self[:].isnull()

    def dropnull(self) -> None:
        # TODO: may fail, needs more testing with VectorBuffers
        self._meta = self._meta.dropnull()
        self._size = len(self._meta)
        self._insertion_idx = len(self._meta)
