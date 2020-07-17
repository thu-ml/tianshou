import copy
import pickle
import time

import numpy as np
import torch

from tianshou.data import Batch


def test_init():
    batch_set = [Batch(a=[j for j in np.arange(1e3)],
                       b={'b1': (3.14, 3.14), 'b2': np.arange(1e3)},
                       c=i) for i in np.arange(int(1e4))]
    init_time = time.time()
    for _ in np.arange(10):
        _ = Batch(batch_set)
    init_time = time.time() - init_time
    print("batch init time {:.3f} s".format(init_time))


def test_get_set():
    """Test get/set with item&attr."""
    np.random.seed(0)
    batch_len = int(1e4)
    batch = Batch(obs=[np.arange(20) for _ in np.arange(batch_len)],
                  reward=np.arange(batch_len))
    indexs = np.random.choice(batch_len,
                              size=batch_len//10, replace=False)
    getitem_time = time.time()
    for _ in np.arange(1e5):
        _ = batch[indexs]
    getitem_time = time.time() - getitem_time
    print("batch getitem time {:.3f} s".format(getitem_time))

    getattr_time = time.time()
    for _ in np.arange(1e6):
        batch.get('obs')
        batch.get('reward')
        _, _ = batch.obs, batch.reward
    getattr_time = time.time() - getattr_time
    print("batch getattr time {:.3f} s".format(getattr_time))

    setitem_time = time.time()
    for _ in np.arange(1e4):
        batch[indexs] = {'obs': [np.arange(20)
                                 for _ in np.arange(batch_len//10)],
                         'reward': np.arange(batch_len//10)}
    setitem_time = time.time() - setitem_time
    print("batch setitem time {:.3f} s".format(setitem_time))

    dict_set = [{'obs': np.arange(20), 'info': "this is info", 'reward': 0}
                for _ in np.arange(1e2)]
    setattr_time = time.time()
    for _ in np.arange(1e4):
        batch.c = np.arange(1e3)
        batch.obs = dict_set
    setattr_time = time.time() - setattr_time
    print("batch setattr time {:.3f} s".format(setattr_time))


def test_numpy_torch_convert():
    """Test conversion between numpy and torch."""
    batch = Batch(
        a=np.ones((10000, 4), dtype=np.float64),
        b=Batch(
            c=np.ones((1,), dtype=np.float64),
            d=torch.ones((1000, 1000), dtype=torch.float32),
            e=range(1000)
        )
    )
    convert_time = time.time()
    for _ in np.arange(1e5):
        batch.to_torch()
        batch.to_numpy()
    convert_time = time.time() - convert_time
    print("batch numpy torch convert time {:.3f} s".format(convert_time))


def test_pickle():
    pickle_time = time.time()
    batch = Batch(obs=Batch(o1=0.0, o2=torch.zeros(10000, 20, 5),
                            o3=Batch(a=1)),
                  act=np.zeros([10000, 20]))
    for _ in np.arange(1e4):
        pickle.loads(pickle.dumps(batch))
    pickle_time = time.time() - pickle_time
    print("batch load pickle time {:.3f} s".format(pickle_time))


def test_concat_split():
    """Test splict, cat, stack, etc."""
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

    cat_time = time.time()
    for _ in np.arange(1e4):
        Batch.cat((batch0, batch0))
        batch1.cat_(batch0)
    cat_time = time.time() - cat_time
    print("batch cat time {:.3f} s".format(cat_time))

    stack_time = time.time()
    for _ in np.arange(1e4):
        Batch.stack((batch0, batch0))
        batch2.stack(batch0)
    stack_time = time.time() - stack_time
    print("batch stack time {:.3f} s".format(stack_time))


if __name__ == '__main__':
    test_init()
    test_get_set()
    test_numpy_torch_convert()
    test_pickle()
    test_concat_split()
