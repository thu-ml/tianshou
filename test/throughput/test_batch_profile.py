import copy
import pickle

import numpy as np
import pytest
import torch

from tianshou.data import Batch


@pytest.fixture(scope="module")
def data():
    print("Initializing data...")
    np.random.seed(0)
    batch_set = [
        Batch(
            a=[j for j in np.arange(1e3)],
            b={
                'b1': (3.14, 3.14),
                'b2': np.arange(1e3)
            },
            c=i,
        ) for i in np.arange(int(1e4))
    ]
    batch0 = Batch(
        a=np.ones((3, 4), dtype=np.float64),
        b=Batch(
            c=np.ones((1, ), dtype=np.float64),
            d=torch.ones((3, 3, 3), dtype=torch.float32),
            e=list(range(3)),
        ),
    )
    batchs1 = [copy.deepcopy(batch0) for _ in np.arange(1e4)]
    batchs2 = [copy.deepcopy(batch0) for _ in np.arange(1e4)]
    batch_len = int(1e4)
    batch3 = Batch(
        obs=[np.arange(20) for _ in np.arange(batch_len)], reward=np.arange(batch_len)
    )
    indexs = np.random.choice(batch_len, size=batch_len // 10, replace=False)
    slice_dict = {
        'obs': [np.arange(20) for _ in np.arange(batch_len // 10)],
        'reward': np.arange(batch_len // 10),
    }
    dict_set = [
        {
            'obs': np.arange(20),
            'info': "this is info",
            'reward': 0,
        } for _ in np.arange(1e2)
    ]
    batch4 = Batch(
        a=np.ones((10000, 4), dtype=np.float64),
        b=Batch(
            c=np.ones((1, ), dtype=np.float64),
            d=torch.ones((1000, 1000), dtype=torch.float32),
            e=np.arange(1000),
        ),
    )

    print("Initialized")
    return {
        'batch_set': batch_set,
        'batch0': batch0,
        'batchs1': batchs1,
        'batchs2': batchs2,
        'batch3': batch3,
        'indexs': indexs,
        'dict_set': dict_set,
        'slice_dict': slice_dict,
        'batch4': batch4,
    }


def test_init(data):
    """Test Batch __init__()."""
    for _ in np.arange(10):
        _ = Batch(data['batch_set'])


def test_get_item(data):
    """Test get with item."""
    for _ in np.arange(1e5):
        _ = data['batch3'][data['indexs']]


def test_get_attr(data):
    """Test get with attr."""
    for _ in np.arange(1e6):
        data['batch3'].get('obs')
        data['batch3'].get('reward')
        _, _ = data['batch3'].obs, data['batch3'].reward


def test_set_item(data):
    """Test set with item."""
    for _ in np.arange(1e4):
        data['batch3'][data['indexs']] = data['slice_dict']


def test_set_attr(data):
    """Test set with attr."""
    for _ in np.arange(1e4):
        data['batch3'].c = np.arange(1e3)
        data['batch3'].obs = data['dict_set']


def test_numpy_torch_convert(data):
    """Test conversion between numpy and torch."""
    for _ in np.arange(1e4):  # not sure what's wrong in torch==1.10.0
        data['batch4'].to_torch()
        data['batch4'].to_numpy()


def test_pickle(data):
    for _ in np.arange(1e4):
        pickle.loads(pickle.dumps(data['batch4']))


def test_cat(data):
    """Test cat"""
    for i in range(10000):
        Batch.cat((data['batch0'], data['batch0']))
        data['batchs1'][i].cat_(data['batch0'])


def test_stack(data):
    """Test stack"""
    for i in range(10000):
        Batch.stack((data['batch0'], data['batch0']))
        data['batchs2'][i].stack_([data['batch0']])


if __name__ == '__main__':
    pytest.main(["-s", "-k batch_profile", "--durations=0", "-v"])
