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
    assert np.allclose(batch.obs, [1, 1])
    assert batch.np.shape == (6, 4)
    assert np.allclose(batch[0].obs, batch[1].obs)
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
    assert Batch().shape == []
    assert batch2.shape[0] == 1
    with pytest.raises(IndexError):
        batch2[-2]
    with pytest.raises(IndexError):
        batch2[1]
    assert batch2[0].shape == []
    with pytest.raises(IndexError):
        batch2[0][0]
    with pytest.raises(TypeError):
        len(batch2[0])
    assert isinstance(batch2[0].a.c, np.ndarray)
    assert isinstance(batch2[0].a.b, np.float64)
    assert isinstance(batch2[0].a.d.e, np.float64)
    batch2_from_list = Batch(list(batch2))
    batch2_from_comp = Batch([e for e in batch2])
    assert batch2_from_list.a.b == batch2.a.b
    assert batch2_from_list.a.c == batch2.a.c
    assert batch2_from_list.a.d.e == batch2.a.d.e
    assert batch2_from_comp.a.b == batch2.a.b
    assert batch2_from_comp.a.c == batch2.a.c
    assert batch2_from_comp.a.d.e == batch2.a.d.e
    for batch_slice in [
            batch2[slice(0, 1)], batch2[:1], batch2[0:]]:
        assert batch_slice.a.b == batch2.a.b
        assert batch_slice.a.c == batch2.a.c
        assert batch_slice.a.d.e == batch2.a.d.e
    batch2_sum = (batch2 + 1.0) * 2
    assert batch2_sum.a.b == (batch2.a.b + 1.0) * 2
    assert batch2_sum.a.c == (batch2.a.c + 1.0) * 2
    assert batch2_sum.a.d.e == (batch2.a.d.e + 1.0) * 2
    batch3 = Batch(a={
        'c': np.zeros(1),
        'd': Batch(e=np.array([0.0]), f=np.array([3.0]))})
    batch3.a.d[0] = {'e': 4.0}
    assert batch3.a.d.e[0] == 4.0
    batch3.a.d[0] = Batch(f=5.0)
    assert batch3.a.d.f[0] == 5.0
    with pytest.raises(KeyError):
        batch3.a.d[0] = Batch(f=5.0, g=0.0)


def test_batch_over_batch():
    batch = Batch(a=[3, 4, 5], b=[4, 5, 6])
    batch2 = Batch({'c': [6, 7, 8], 'b': batch})
    batch2.b.b[-1] = 0
    print(batch2)
    for k, v in batch2.items():
        assert np.all(batch2[k] == v)
    assert batch2[-1].b.b == 0
    batch2.cat_(Batch(c=[6, 7, 8], b=batch))
    assert np.allclose(batch2.c, [6, 7, 8, 6, 7, 8])
    assert np.allclose(batch2.b.a, [3, 4, 5, 3, 4, 5])
    assert np.allclose(batch2.b.b, [4, 5, 0, 4, 5, 0])
    d = {'a': [3, 4, 5], 'b': [4, 5, 6]}
    batch3 = Batch(c=[6, 7, 8], b=d)
    batch3.cat_(Batch(c=[6, 7, 8], b=d))
    assert np.allclose(batch3.c, [6, 7, 8, 6, 7, 8])
    assert np.allclose(batch3.b.a, [3, 4, 5, 3, 4, 5])
    assert np.allclose(batch3.b.b, [4, 5, 6, 4, 5, 6])
    batch4 = Batch(({'a': {'b': np.array([1.0])}},))
    assert batch4.a.b.ndim == 2
    assert batch4.a.b[0, 0] == 1.0
    # advanced slicing
    batch5 = Batch(a=[[1, 2]], b={'c': np.zeros([3, 2, 1])})
    assert batch5.shape == [1, 2]
    with pytest.raises(IndexError):
        batch5[2]
    with pytest.raises(IndexError):
        batch5[:, 3]
    with pytest.raises(IndexError):
        batch5[:, :, -1]
    batch5[:, -1] += 1
    assert np.allclose(batch5.a, [1, 3])
    assert np.allclose(batch5.b.c.squeeze(), [[0, 1]] * 3)


def test_batch_cat_and_stack():
    b1 = Batch(a=[{'b': np.float64(1.0), 'd': Batch(e=np.array(3.0))}])
    b2 = Batch(a=[{'b': np.float64(4.0), 'd': {'e': np.array(6.0)}}])
    b12_cat_out = Batch.cat((b1, b2))
    b12_cat_in = copy.deepcopy(b1)
    b12_cat_in.cat_(b2)
    assert np.all(b12_cat_in.a.d.e == b12_cat_out.a.d.e)
    assert np.all(b12_cat_in.a.d.e == b12_cat_out.a.d.e)
    assert isinstance(b12_cat_in.a.d.e, np.ndarray)
    assert b12_cat_in.a.d.e.ndim == 1
    b12_stack = Batch.stack((b1, b2))
    assert isinstance(b12_stack.a.d.e, np.ndarray)
    assert b12_stack.a.d.e.ndim == 2
    b3 = Batch(a=np.zeros((3, 4)),
               b=torch.ones((2, 5)),
               c=Batch(d=[[1], [2]]))
    b4 = Batch(a=np.ones((3, 4)),
               b=torch.ones((2, 5)),
               c=Batch(d=[[0], [3]]))
    b34_stack = Batch.stack((b3, b4), axis=1)
    assert np.all(b34_stack.a == np.stack((b3.a, b4.a), axis=1))
    assert np.all(b34_stack.c.d == list(map(list, zip(b3.c.d, b4.c.d))))
    b5_dict = np.array([{'a': False, 'b': {'c': 2.0, 'd': 1.0}},
                        {'a': True, 'b': {'c': 3.0}}])
    b5 = Batch(b5_dict)
    assert b5.a[0] == np.array(False) and b5.a[1] == np.array(True)
    assert np.all(b5.b.c == np.stack([e['b']['c'] for e in b5_dict], axis=0))
    assert b5.b.d[0] == b5_dict[0]['b']['d']
    assert b5.b.d[1] == 0.0


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


def test_batch_copy():
    batch = Batch(a=np.array([3, 4, 5]), b=np.array([4, 5, 6]))
    batch2 = Batch({'c': np.array([6, 7, 8]), 'b': batch})
    orig_c_addr = batch2.c.__array_interface__['data'][0]
    orig_b_a_addr = batch2.b.a.__array_interface__['data'][0]
    orig_b_b_addr = batch2.b.b.__array_interface__['data'][0]
    # test with copy=False
    batch3 = Batch(copy=False, **batch2)
    curr_c_addr = batch3.c.__array_interface__['data'][0]
    curr_b_a_addr = batch3.b.a.__array_interface__['data'][0]
    curr_b_b_addr = batch3.b.b.__array_interface__['data'][0]
    assert batch2.c is batch3.c
    assert batch2.b is batch3.b
    assert batch2.b.a is batch3.b.a
    assert batch2.b.b is batch3.b.b
    assert orig_c_addr == curr_c_addr
    assert orig_b_a_addr == curr_b_a_addr
    assert orig_b_b_addr == curr_b_b_addr
    # test with copy=True
    batch3 = Batch(copy=True, **batch2)
    curr_c_addr = batch3.c.__array_interface__['data'][0]
    curr_b_a_addr = batch3.b.a.__array_interface__['data'][0]
    curr_b_b_addr = batch3.b.b.__array_interface__['data'][0]
    assert batch2.c is not batch3.c
    assert batch2.b is not batch3.b
    assert batch2.b.a is not batch3.b.a
    assert batch2.b.b is not batch3.b.b
    assert orig_c_addr != curr_c_addr
    assert orig_b_a_addr != curr_b_a_addr
    assert orig_b_b_addr != curr_b_b_addr


def test_batch_empty():
    b5_dict = np.array([{'a': False, 'b': {'c': 2.0, 'd': 1.0}},
                        {'a': True, 'b': {'c': 3.0}}])
    b5 = Batch(b5_dict)
    b5[1] = Batch.empty(b5[0])
    assert np.allclose(b5.a, [False, False])
    assert np.allclose(b5.b.c, [2, 0])
    assert np.allclose(b5.b.d, [1, 0])
    data = Batch(a=[False, True],
                 b={'c': np.array([2., 'st'], dtype=np.object),
                    'd': [1, None],
                    'e': [2., float('nan')]},
                 c=np.array([1, 3, 4], dtype=np.int),
                 t=torch.tensor([4, 5, 6, 7.]))
    data[-1] = Batch.empty(data[1])
    assert np.allclose(data.c, [1, 3, 0])
    assert np.allclose(data.a, [False, False])
    assert list(data.b.c) == [2.0, None]
    assert list(data.b.d) == [1, None]
    assert np.allclose(data.b.e, [2, 0])
    assert torch.allclose(data.t, torch.tensor([4, 5, 6, 0.]))
    data[0].empty_()  # which will fail in a, b.c, b.d, b.e, c
    assert torch.allclose(data.t, torch.tensor([0., 5, 6, 0]))
    data.empty_(index=0)
    assert np.allclose(data.c, [0, 3, 0])
    assert list(data.b.c) == [None, None]
    assert list(data.b.d) == [None, None]
    assert list(data.b.e) == [0, 0]
    b0 = Batch()
    b0.empty_()
    assert b0.shape == []


def test_batch_numpy_compatibility():
    batch = Batch(a=np.array([[1.0, 2.0], [3.0, 4.0]]),
                  b=Batch(),
                  c=np.array([5.0, 6.0]))
    batch_mean = np.mean(batch)
    assert isinstance(batch_mean, Batch)
    assert sorted(batch_mean.keys()) == ['a', 'b', 'c']
    with pytest.raises(TypeError):
        len(batch_mean)
    assert np.all(batch_mean.a == np.mean(batch.a, axis=0))
    assert batch_mean.c == np.mean(batch.c, axis=0)


if __name__ == '__main__':
    test_batch()
    test_batch_over_batch()
    test_batch_over_batch_to_torch()
    test_utils_to_torch()
    test_batch_pickle()
    test_batch_from_to_numpy_without_copy()
    test_batch_numpy_compatibility()
    test_batch_cat_and_stack()
    test_batch_copy()
    test_batch_empty()
