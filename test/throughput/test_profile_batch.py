import copy
import pickle
import time

import numpy as np
import pytest
import torch

from tianshou.data import Batch


@pytest.fixture(scope="module")
def data():
    print("Initialising data...")
    np.random.seed(0)
    batch_set = [Batch(a=[j for j in np.arange(1e3)],
                       b={'b1': (3.14, 3.14), 'b2': np.arange(1e3)},
                       c=i) for i in np.arange(int(1e4))]
    batch0 = Batch(
        a=np.ones((3, 4), dtype=np.float64),
        b=Batch(
            c=np.ones((1,), dtype=np.float64),
            d=torch.ones((3, 3, 3), dtype=torch.float32),
            e=list(range(3))
        )
    )
    batch1 = copy.deepcopy(batch0)
    batch2 = copy.deepcopy(batch0)
    batch_len = int(1e4)
    batch3 = Batch(obs=[np.arange(20) for _ in np.arange(batch_len)],
                   reward=np.arange(batch_len))
    indexs = np.random.choice(batch_len,
                              size=batch_len//10, replace=False)
    slice_dict = {'obs': [np.arange(20)
                          for _ in np.arange(batch_len//10)],
                  'reward': np.arange(batch_len//10)}
    dict_set = [{'obs': np.arange(20), 'info': "this is info", 'reward': 0}
                for _ in np.arange(1e2)]
    batch4 = Batch(
        a=np.ones((10000, 4), dtype=np.float64),
        b=Batch(
            c=np.ones((1,), dtype=np.float64),
            d=torch.ones((1000, 1000), dtype=torch.float32),
            e=np.arange(1000)
        )
    )

    print("Initialised")
    return {'batch_set': batch_set,
            'batch0': batch0,
            'batch1': batch1,
            'batch2': batch2,
            'batch3': batch3,
            'indexs': indexs,
            'dict_set': dict_set,
            'slice_dict': slice_dict,
            'batch4': batch4
            }


def test_init(data):
    init_time = time.time()
    for _ in np.arange(10):
        _ = Batch(data['batch_set'])
    init_time = time.time() - init_time
    print("batch init time {:.3f} s".format(init_time))


def test_get(data):
    """Test get with item&attr."""
    getitem_time = time.time()
    for _ in np.arange(1e5):
        _ = data['batch3'][data['indexs']]
    getitem_time = time.time() - getitem_time
    print("batch getitem time {:.3f} s".format(getitem_time))

    getattr_time = time.time()
    for _ in np.arange(1e6):
        data['batch3'].get('obs')
        data['batch3'].get('reward')
        _, _ = data['batch3'].obs, data['batch3'].reward
    getattr_time = time.time() - getattr_time
    print("batch getattr time {:.3f} s".format(getattr_time))


def test_set(data):
    """Test set with item&attr."""
    setitem_time = time.time()
    for _ in np.arange(1e4):
        data['batch3'][data['indexs']] = data['slice_dict']
    setitem_time = time.time() - setitem_time
    print("batch setitem time {:.3f} s".format(setitem_time))

    setattr_time = time.time()
    for _ in np.arange(1e4):
        data['batch3'].c = np.arange(1e3)
        data['batch3'].obs = data['dict_set']
    setattr_time = time.time() - setattr_time
    print("batch setattr time {:.3f} s".format(setattr_time))


def test_numpy_torch_convert(data):
    """Test conversion between numpy and torch."""
    convert_time = time.time()
    for _ in np.arange(1e5):
        data['batch4'].to_torch()
        data['batch4'].to_numpy()
    convert_time = time.time() - convert_time
    print("batch numpy torch convert time {:.3f} s".format(convert_time))


def test_pickle(data):
    pickle_time = time.time()
    for _ in np.arange(1e4):
        pickle.loads(pickle.dumps(data['batch4']))
    pickle_time = time.time() - pickle_time
    print("batch load pickle time {:.3f} s".format(pickle_time))


def test_cat(data):
    """Test cat"""
    cat_time = time.time()
    for _ in np.arange(1e4):
        Batch.cat((data['batch0'], data['batch0']))
        data['batch1'].cat_(data['batch0'])
    cat_time = time.time() - cat_time
    print("batch cat time {:.3f} s".format(cat_time))


def test_stack(data):
    """Test stack"""
    stack_time = time.time()
    for _ in np.arange(1e4):
        Batch.stack((data['batch0'], data['batch0']))
        data['batch2'].stack(data['batch0'])
    stack_time = time.time() - stack_time
    print("batch stack time {:.3f} s".format(stack_time))


if __name__ == '__main__':
    pytest.main(["-s", "-k profile_batch", "--durations=0", "-v"])
