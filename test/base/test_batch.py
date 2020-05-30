import pytest
import torch
import numpy as np

from tianshou.data import Batch, to_torch


def test_batch():
    batch = Batch(obs=[0], np=np.zeros([3, 4]))
    assert batch.obs == batch["obs"]
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


def test_batch_over_batch_to_torch():
    batch = Batch(
        a=np.ones((1,), dtype=np.float64),
        b=Batch(
            c=np.ones((1,), dtype=np.float64),
            d=torch.ones((1,), dtype=torch.float64)
        )
    )
    batch.to_torch()
    assert isinstance(batch.a, torch.Tensor)
    assert isinstance(batch.b.c, torch.Tensor)
    assert isinstance(batch.b.d, torch.Tensor)
    assert batch.a.dtype == torch.float64
    assert batch.b.c.dtype == torch.float64
    assert batch.b.d.dtype == torch.float64
    batch.to_torch(dtype=torch.float32)
    assert batch.a.dtype == torch.float32
    assert batch.b.c.dtype == torch.float32
    assert batch.b.d.dtype == torch.float32


def test_utils_to_torch():
    batch = Batch(
        a=np.ones((1,), dtype=np.float64),
        b=Batch(
            c=np.ones((1,), dtype=np.float64),
            d=torch.ones((1,), dtype=torch.float64)
        )
    )
    a_torch_float = to_torch(batch.a, dtype=torch.float32)
    assert a_torch_float.dtype == torch.float32
    a_torch_double = to_torch(batch.a, dtype=torch.float64)
    assert a_torch_double.dtype == torch.float64
    batch_torch_float = to_torch(batch, dtype=torch.float32)
    assert batch_torch_float.a.dtype == torch.float32
    assert batch_torch_float.b.c.dtype == torch.float32
    assert batch_torch_float.b.d.dtype == torch.float32


def test_batch_from_to_numpy_without_copy():
    batch = Batch(a=np.ones((1,)), b=Batch(c=np.ones((1,))))
    a_mem_addr_orig = batch.a.__array_interface__['data'][0]
    c_mem_addr_orig = batch.b.c.__array_interface__['data'][0]
    batch.to_torch()
    batch.to_numpy()
    a_mem_addr_new = batch.a.__array_interface__['data'][0]
    c_mem_addr_new = batch.b.c.__array_interface__['data'][0]
    assert a_mem_addr_new == a_mem_addr_orig
    assert c_mem_addr_new == c_mem_addr_orig


if __name__ == '__main__':
    test_batch()
    test_batch_over_batch()
