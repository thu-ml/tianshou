import pytest
import pickle
import torch
import numpy as np

from tianshou.data import Batch


def test_batch():
    batch = Batch(obs=[0], np=np.zeros([3, 4]))
    batch.obs = [1]
    assert batch.obs == [1]
    batch.append(batch)
    assert batch.obs == [1, 1]
    assert batch.np.shape == (6, 4)
    assert batch[0].obs == batch[1].obs
    with pytest.raises(IndexError):
        batch[2]
    batch.obs = np.arange(5)
    for i, b in enumerate(batch.split(1, shuffle=False)):
        assert b.obs == batch[i].obs
    print(batch)


def test_batch_over_batch():
    batch = Batch(a=[3, 4, 5], b=[4, 5, 6])
    batch2 = Batch(c=[6, 7, 8], b=batch)
    batch2.b.b[-1] = 0
    print(batch2)
    assert batch2.values()[-1] == batch2.c
    assert batch2[-1].b.b == 0


def test_batch_pickle():
    batch = Batch(obs=Batch(a=0.0, c=torch.Tensor([1.0, 2.0])),
                  np=np.zeros([3, 4]))
    batch_pk = pickle.loads(pickle.dumps(batch))
    assert batch.obs.a == batch_pk.obs.a
    assert torch.all(batch.obs.c == batch_pk.obs.c)
    assert np.all(batch.np == batch_pk.np)


def test_batch_from_to_numpy_without_copy():
    batch = Batch(a=np.ones((1,)), b=Batch(c=np.ones((1,))))
    a_mem_addr_orig = batch["a"].__array_interface__['data'][0]
    c_mem_addr_orig = batch["b"]["c"].__array_interface__['data'][0]
    batch.to_torch()
    assert isinstance(batch["a"], torch.Tensor)
    assert isinstance(batch["b"]["c"], torch.Tensor)
    batch.to_numpy()
    a_mem_addr_new = batch["a"].__array_interface__['data'][0]
    c_mem_addr_new = batch["b"]["c"].__array_interface__['data'][0]
    assert a_mem_addr_new == a_mem_addr_orig
    assert c_mem_addr_new == c_mem_addr_orig


if __name__ == '__main__':
    test_batch()
    test_batch_over_batch()
