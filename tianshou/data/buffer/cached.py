from typing import List, Optional, Tuple, Union

import numpy as np

from tianshou.data import Batch, ReplayBuffer, ReplayBufferManager


class CachedReplayBuffer(ReplayBufferManager):
    """CachedReplayBuffer contains a given main buffer and n cached buffers, \
    ``cached_buffer_num * ReplayBuffer(size=max_episode_length)``.

    The memory layout is: ``| main_buffer | cached_buffers[0] | cached_buffers[1] | ...
    | cached_buffers[cached_buffer_num - 1] |``.

    The data is first stored in cached buffers. When an episode is terminated, the data
    will move to the main buffer and the corresponding cached buffer will be reset.

    :param ReplayBuffer main_buffer: the main buffer whose ``.update()`` function
        behaves normally.
    :param int cached_buffer_num: number of ReplayBuffer needs to be created for cached
        buffer.
    :param int max_episode_length: the maximum length of one episode, used in each
        cached buffer's maxsize.

    .. seealso::

        Please refer to :class:`~tianshou.data.ReplayBuffer` for other APIs' usage.
    """

    def __init__(
        self,
        main_buffer: ReplayBuffer,
        cached_buffer_num: int,
        max_episode_length: int,
    ) -> None:
        assert cached_buffer_num > 0 and max_episode_length > 0
        assert isinstance(main_buffer, ReplayBuffer)
        kwargs = main_buffer.options
        buffers = [main_buffer] + [
            ReplayBuffer(max_episode_length, **kwargs)
            for _ in range(cached_buffer_num)
        ]
        super().__init__(buffer_list=buffers)
        self.main_buffer = self.buffers[0]
        self.cached_buffers = self.buffers[1:]
        self.cached_buffer_num = cached_buffer_num

    def add(
        self,
        batch: Batch,
        buffer_ids: Optional[Union[np.ndarray, List[int]]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Add a batch of data into CachedReplayBuffer.

        Each of the data's length (first dimension) must equal to the length of
        buffer_ids. By default the buffer_ids is [0, 1, ..., cached_buffer_num - 1].

        Return (current_index, episode_reward, episode_length, episode_start_index)
        with each of the shape (len(buffer_ids), ...), where (current_index[i],
        episode_reward[i], episode_length[i], episode_start_index[i]) refers to the
        cached_buffer_ids[i]th cached buffer's corresponding episode result.
        """
        if buffer_ids is None:
            buf_arr = np.arange(1, 1 + self.cached_buffer_num)
        else:  # make sure it is np.ndarray
            buf_arr = np.asarray(buffer_ids) + 1
        ptr, ep_rew, ep_len, ep_idx = super().add(batch, buffer_ids=buf_arr)
        # find the terminated episode, move data from cached buf to main buf
        updated_ptr, updated_ep_idx = [], []
        done = np.logical_or(batch.terminated, batch.truncated)
        for buffer_idx in buf_arr[done]:
            index = self.main_buffer.update(self.buffers[buffer_idx])
            if len(index) == 0:  # unsuccessful move, replace with -1
                index = [-1]
            updated_ep_idx.append(index[0])
            updated_ptr.append(index[-1])
            self.buffers[buffer_idx].reset()
            self._lengths[0] = len(self.main_buffer)
            self._lengths[buffer_idx] = 0
            self.last_index[0] = index[-1]
            self.last_index[buffer_idx] = self._offset[buffer_idx]
        ptr[done] = updated_ptr
        ep_idx[done] = updated_ep_idx
        return ptr, ep_rew, ep_len, ep_idx
