import pytest
import numpy as np

from tianshou.data import Batch


def test_batch():
    batch = Batch(obs=[0], np=np.zeros([3, 4]))
    batch.update(obs=[1])
    assert batch.obs == [1]
    batch.append(batch)
    assert batch.obs == [1, 1]
    assert batch.np.shape == (6, 4)
    assert batch[0].obs == batch[1].obs
    with pytest.raises(IndexError):
        batch[2]


if __name__ == '__main__':
    test_batch()
