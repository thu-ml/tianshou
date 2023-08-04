import numpy as np
import torch
from torch.utils.data import Subset, TensorDataset
from typing import (
    Callable,
    Iterator,
    Literal,
    Optional,
    Sequence,
    TypeVar,
    Union,
)


def get_batch_boundaries(
    batch_size: int,
    len_data: int,
    last_batch: Literal["drop", "merge", "keep"] = "merge",
):
    """Get the boundaries of batches for a given batch size and data length.

    :param batch_size: the size of each batch
    :param len_data: the length of the data
    :param last_batch: one of "drop", "merge", or "keep".
        - "drop": drop the last batch if it is smaller than batch_size
        - "merge": merge the last batch with the previous batch
        - "keep": keep the last batch as is, even if it is smaller than batch_size
    :return: a numpy array of batch boundaries
    """
    if batch_size >= len_data:
        return np.array([0, len_data])

    batch_boundaries = np.arange(0, len_data + 1, batch_size)
    if len_data % batch_size == 0 or last_batch == "drop":
        return batch_boundaries

    elif last_batch == "merge":
        batch_boundaries[-1] = len_data
    elif last_batch == "keep":
        batch_boundaries = np.append(batch_boundaries, len_data)
    else:
        raise ValueError(
            f"last_batch must be one of 'drop', 'merge', or 'keep', "
            f"but got {last_batch}"
        )
    return batch_boundaries



SupportsBatching = Union[
    TensorDataset,
    Subset,
    torch.Tensor,
    np.ndarray,
    Sequence,
]


T = TypeVar("T")


class BatchDataLoader:
    def __init__(
        self,
        data: SupportsBatching,
        batch_size: int,
        shuffle: bool = False,
        last_batch: Literal["drop", "merge", "keep"] = "merge",
        collate_fn: Optional[Callable[[Union[torch.Tensor, np.ndarray]], T]] = None,
    ) -> None:
        """A simple data loader that returns batches of data.

        :param data: the data to be loaded. If tensor-based, the batches will be
            tensors, otherwise they will be numpy arrays.
        :param batch_size: the size of each batch
        :param shuffle: whether to shuffle the data before batching
        :param last_batch: one of "drop", "merge", or "keep".
            - "drop": drop the last batch if it is smaller than batch_size
            - "merge": merge the last batch with the previous batch
            - "keep": keep the last batch as is, even if it is smaller than batch_size
        :param collate_fn: a function to apply to each batch before returning it
        """
        if isinstance(data, Sequence):
            data = np.array(data)
        # not pretty nor robust, but hopefully this code won't be around for long anyway
        while isinstance(data, Subset):
            data = data.dataset

        self._data = data

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.last_batch = last_batch

        self._boundary_idxs = get_batch_boundaries(
            batch_size, len(data), last_batch=last_batch
        )
        self._num_batches = len(self._boundary_idxs) - 1
        self.collate_fn = collate_fn or (lambda x: x)

    # TODO: the generic annotation here is probably incorrect
    def __iter__(self) -> Iterator[Union[np.ndarray, torch.Tensor, T]]:
        if self.shuffle:
            self._shuffle_data()
        for lower, upper in zip(self._boundary_idxs[:-1], self._boundary_idxs[1:]):
            yield self.collate_fn(self._data[lower:upper])

    def _shuffle_data(self):
        data_type = type(self._data)
        self._data = self._data[np.random.permutation(len(self._data))]

        if issubclass(data_type, TensorDataset):
            # retrieving data from these types changes the type, so we change it back
            self._data = data_type(*self._data)

    def __len__(self) -> int:
        return self._num_batches
