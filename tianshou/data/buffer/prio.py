import torch
import numpy as np
from typing import Any, List, Tuple, Union, Optional

from tianshou.data import Batch, SegmentTree, to_numpy, ReplayBuffer


class PrioritizedReplayBuffer(ReplayBuffer):
    """Implementation of Prioritized Experience Replay. arXiv:1511.05952.

    :param float alpha: the prioritization exponent.
    :param float beta: the importance sample soft coefficient.

    .. seealso::

        Please refer to :class:`~tianshou.data.ReplayBuffer` for other APIs' usage.
    """

    def __init__(self, size: int, alpha: float, beta: float, **kwargs: Any) -> None:
        # will raise KeyError in PrioritizedVectorReplayBuffer
        # super().__init__(size, **kwargs)
        ReplayBuffer.__init__(self, size, **kwargs)
        assert alpha > 0.0 and beta >= 0.0
        self._alpha, self._beta = alpha, beta
        self._max_prio = self._min_prio = 1.0
        # save weight directly in this class instead of self._meta
        self.weight = SegmentTree(size)
        self.__eps = np.finfo(np.float32).eps.item()
        self.options.update(alpha=alpha, beta=beta)

    def init_weight(self, index: Union[int, np.ndarray]) -> None:
        self.weight[index] = self._max_prio ** self._alpha

    def update(self, buffer: ReplayBuffer) -> np.ndarray:
        indices = super().update(buffer)
        self.init_weight(indices)

    def add(
        self, batch: Batch, buffer_ids: Optional[Union[np.ndarray, List[int]]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ptr, ep_rew, ep_len, ep_idx = super().add(batch, buffer_ids)
        self.init_weight(ptr)
        return ptr, ep_rew, ep_len, ep_idx

    def sample_index(self, batch_size: int) -> np.ndarray:
        if batch_size > 0 and len(self) > 0:
            scalar = np.random.rand(batch_size) * self.weight.reduce()
            return self.weight.get_prefix_sum_idx(scalar)
        else:
            return super().sample_index(batch_size)

    def get_weight(
        self, index: Union[slice, int, np.integer, np.ndarray]
    ) -> np.ndarray:
        """Get the importance sampling weight.

        The "weight" in the returned Batch is the weight on loss function to de-bias
        the sampling process (some transition tuples are sampled more often so their
        losses are weighted less).
        """
        # important sampling weight calculation
        # original formula: ((p_j/p_sum*N)**(-beta))/((p_min/p_sum*N)**(-beta))
        # simplified formula: (p_j/p_min)**(-beta)
        return (self.weight[index] / self._min_prio) ** (-self._beta)

    def update_weight(
        self, index: np.ndarray, new_weight: Union[np.ndarray, torch.Tensor]
    ) -> None:
        """Update priority weight by index in this buffer.

        :param np.ndarray index: index you want to update weight.
        :param np.ndarray new_weight: new priority weight you want to update.
        """
        weight = np.abs(to_numpy(new_weight)) + self.__eps
        self.weight[index] = weight ** self._alpha
        self._max_prio = max(self._max_prio, weight.max())
        self._min_prio = min(self._min_prio, weight.min())

    def __getitem__(self, index: Union[slice, int, np.integer, np.ndarray]) -> Batch:
        batch = super().__getitem__(index)
        batch.weight = self.get_weight(index)
        return batch
