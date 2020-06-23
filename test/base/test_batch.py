import torch
import copy
import pickle
import pytest
import numpy as np

from tianshou.data import Batch, to_torch


def test_batch():
    batch = Batch(obs=[0], np=np.zeros([3, 4]))
    assert batch.obs == batch["obs"]
    batch.obs = [1]
    assert batch.obs == [1]
    batch.cat_(batch)
    assert batch.obs == [1, 1]
    assert batch.np.shape == (6, 4)
    assert batch[0].obs == batch[1].obs
    batch.obs = np.arange(5)
    for i, b in enumerate(batch.split(1, shuffle=False)):
        if i != 5:
            assert b.obs == batch[i].obs
        else:
            with pytest.raises(AttributeError):
                batch[i].obs
            with pytest.raises(AttributeError):
                b.obs
    print(batch)
    batch_dict = {'b': np.array([1.0]), 'c': 2.0, 'd': torch.Tensor([3.0])}
    batch_item = Batch({'a': [batch_dict]})[0]
    assert isinstance(batch_item.a.b, np.ndarray)
    assert batch_item.a.b == batch_dict['b']
    assert isinstance(batch_item.a.c, float)
    assert batch_item.a.c == batch_dict['c']
    assert isinstance(batch_item.a.d, torch.Tensor)
    assert batch_item.a.d == batch_dict['d']
    batch2 = Batch(a=[{
        'b': np.float64(1.0),
        'c': np.zeros(1),
        'd': Batch(e=np.array(3.0))}])
    assert len(batch2) == 1
    assert list(batch2[1].keys()) == ['a']
    assert len(batch2[-2].a.d.keys()) == 0
    assert len(batch2[-1].keys()) > 0
    assert batch2[0][0].a.c == 0.0
    assert isinstance(batch2[0].a.c, np.ndarray)
    assert isinstance(batch2[0].a.b, np.float64)
    assert isinstance(batch2[0].a.d.e, np.float64)


def test_batch_over_batch():
    batch = Batch(a=[3, 4, 5], b=[4, 5, 6])
    batch2 = Batch({'c': [6, 7, 8], 'b': batch})
    batch2.b.b[-1] = 0
    print(batch2)
    for k, v in batch2.items():
        assert batch2[k] == v
    assert batch2[-1].b.b == 0
    batch2.cat_(Batch(c=[6, 7, 8], b=batch))
    assert batch2.c == [6, 7, 8, 6, 7, 8]
    assert batch2.b.a == [3, 4, 5, 3, 4, 5]
    assert batch2.b.b == [4, 5, 0, 4, 5, 0]
    d = {'a': [3, 4, 5], 'b': [4, 5, 6]}
    batch3 = Batch(c=[6, 7, 8], b=d)
    batch3.cat_(Batch(c=[6, 7, 8], b=d))
    assert batch3.c == [6, 7, 8, 6, 7, 8]
    assert batch3.b.a == [3, 4, 5, 3, 4, 5]
    assert batch3.b.b == [4, 5, 6, 4, 5, 6]
    batch4 = Batch(({'a': {'b': np.array([1.0])}},))
    assert batch4.a.b.ndim == 2
    assert batch4.a.b[0, 0] == 1.0


def test_batch_cat_and_stack():
    b1 = Batch(a=[{'b': np.float64(1.0), 'd': Batch(e=np.array(3.0))}])
    b2 = Batch(a=[{'b': np.float64(4.0), 'd': {'e': np.array(6.0)}}])
    b_cat_out = Batch.cat((b1, b2))
    b_cat_in = copy.deepcopy(b1)
    b_cat_in.cat_(b2)
    assert np.all(b_cat_in.a.d.e == b_cat_out.a.d.e)
    assert np.all(b_cat_in.a.d.e == b_cat_out.a.d.e)
    assert isinstance(b_cat_in.a.d.e, np.ndarray)
    assert b_cat_in.a.d.e.ndim == 1
    b_stack = Batch.stack((b1, b2))
    assert isinstance(b_stack.a.d.e, np.ndarray)
    assert b_stack.a.d.e.ndim == 2


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


def test_batch_pickle():
    batch = Batch(obs=Batch(a=0.0, c=torch.Tensor([1.0, 2.0])),
                  np=np.zeros([3, 4]))
    batch_pk = pickle.loads(pickle.dumps(batch))
    assert batch.obs.a == batch_pk.obs.a
    assert torch.all(batch.obs.c == batch_pk.obs.c)
    assert np.all(batch.np == batch_pk.np)


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
    test_batch_over_batch_to_torch()
    test_utils_to_torch()
    test_batch_pickle()
    test_batch_from_to_numpy_without_copy()
