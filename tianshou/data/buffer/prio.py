from typing import Any, cast

import numpy as np
import torch

from tianshou.data import ReplayBuffer, SegmentTree, to_numpy
from tianshou.data.types import PrioBatchProtocol, RolloutBatchProtocol


class PrioritizedReplayBuffer(ReplayBuffer):
    """Implementation of Prioritized Experience Replay. arXiv:1511.05952.

    :param alpha: the prioritization exponent.
    :param beta: the importance sample soft coefficient.
    :param weight_norm: whether to normalize returned weights with the maximum
        weight value within the batch. Default to True.

    .. seealso::

        Please refer to :class:`~tianshou.data.ReplayBuffer` for other APIs' usage.
    """

    def __init__(
        self,
        size: int,
        alpha: float,
        beta: float,
        weight_norm: bool = True,
        **kwargs: Any,
    ) -> None:
        # will raise KeyError in PrioritizedVectorReplayBuffer
        # super().__init__(size, **kwargs)
        ReplayBuffer.__init__(self, size, **kwargs)
        assert alpha > 0.0
        assert beta >= 0.0
        self._alpha, self._beta = alpha, beta
        self._max_prio = self._min_prio = 1.0
        # save weight directly in this class instead of self._meta
        self.weight = SegmentTree(size)
        self.__eps = np.finfo(np.float32).eps.item()
        self.options.update(alpha=alpha, beta=beta)
        self._weight_norm = weight_norm

    def init_weight(self, index: int | np.ndarray) -> None:
        self.weight[index] = self._max_prio**self._alpha

    def update(self, buffer: ReplayBuffer) -> np.ndarray:
        indices = super().update(buffer)
        self.init_weight(indices)
        return indices

    def add(
        self,
        batch: RolloutBatchProtocol,
        buffer_ids: np.ndarray | list[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ptr, ep_rew, ep_len, ep_idx = super().add(batch, buffer_ids)
        self.init_weight(ptr)
        return ptr, ep_rew, ep_len, ep_idx

    def sample_indices(self, batch_size: int | None) -> np.ndarray:
        if batch_size is not None and batch_size > 0 and len(self) > 0:
            scalar = np.random.rand(batch_size) * self.weight.reduce()
            return self.weight.get_prefix_sum_idx(scalar)  # type: ignore
        return super().sample_indices(batch_size)

    def get_weight(self, index: int | np.ndarray) -> float | np.ndarray:
        """Get the importance sampling weight.

        The "weight" in the returned Batch is the weight on loss function to debias
        the sampling process (some transition tuples are sampled more often so their
        losses are weighted less).
        """
        # important sampling weight calculation
        # original formula: ((p_j/p_sum*N)**(-beta))/((p_min/p_sum*N)**(-beta))
        # simplified formula: (p_j/p_min)**(-beta)
        return (self.weight[index] / self._min_prio) ** (-self._beta)

    def update_weight(self, index: np.ndarray, new_weight: np.ndarray | torch.Tensor) -> None:
        """Update priority weight by index in this buffer.

        :param np.ndarray index: index you want to update weight.
        :param np.ndarray new_weight: new priority weight you want to update.
        """
        weight = np.abs(to_numpy(new_weight)) + self.__eps
        self.weight[index] = weight**self._alpha
        self._max_prio = max(self._max_prio, weight.max())
        self._min_prio = min(self._min_prio, weight.min())

    def __getitem__(self, index: slice | int | list[int] | np.ndarray) -> PrioBatchProtocol:
        if isinstance(index, slice):  # change slice to np array
            # buffer[:] will get all available data
            indices = (
                self.sample_indices(0)
                if index == slice(None)
                else self._indices[: len(self)][index]
            )
        else:
            indices = index  # type: ignore
        batch = super().__getitem__(indices)
        weight = self.get_weight(indices)
        # ref: https://github.com/Kaixhin/Rainbow/blob/master/memory.py L154
        batch.weight = weight / np.max(weight) if self._weight_norm else weight
        return cast(PrioBatchProtocol, batch)

    def sample(self, batch_size: int | None) -> tuple[PrioBatchProtocol, np.ndarray]:
        return cast(tuple[PrioBatchProtocol, np.ndarray], super().sample(batch_size=batch_size))

    def set_beta(self, beta: float) -> None:
        self._beta = beta
