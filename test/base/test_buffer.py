import torch
import pickle
import pytest
import numpy as np
from timeit import timeit

from tianshou.data import Batch, SegmentTree, \
    ReplayBuffer, ListReplayBuffer, PrioritizedReplayBuffer

if __name__ == '__main__':
    from env import MyTestEnv
else:  # pytest
    from test.base.env import MyTestEnv


def test_replaybuffer(size=10, bufsize=20):
    env = MyTestEnv(size)
    buf = ReplayBuffer(bufsize)
    obs = env.reset()
    action_list = [1] * 5 + [0] * 10 + [1] * 10
    for i, a in enumerate(action_list):
        obs_next, rew, done, info = env.step(a)
        buf.add(obs, a, rew, done, obs_next, info)
        obs = obs_next
        assert len(buf) == min(bufsize, i + 1)
    data, indice = buf.sample(bufsize * 2)
    assert (indice < len(buf)).all()
    assert (data.obs < size).all()
    assert (0 <= data.done).all() and (data.done <= 1).all()
    b = ReplayBuffer(size=10)
    b.add(1, 1, 1, 'str', 1, {'a': 3, 'b': {'c': 5.0}})
    assert b.obs[0] == 1
    assert b.done[0] == 'str'
    assert np.all(b.obs[1:] == 0)
    assert np.all(b.done[1:] == np.array(None))
    assert b.info.a[0] == 3 and b.info.a.dtype == np.integer
    assert np.all(b.info.a[1:] == 0)
    assert b.info.b.c[0] == 5.0 and b.info.b.c.dtype == np.inexact
    assert np.all(b.info.b.c[1:] == 0.0)


def test_ignore_obs_next(size=10):
    # Issue 82
    buf = ReplayBuffer(size, ignore_obs_next=True)
    for i in range(size):
        buf.add(obs={'mask1': np.array([i, 1, 1, 0, 0]),
                     'mask2': np.array([i + 4, 0, 1, 0, 0]),
                     'mask': i},
                act={'act_id': i,
                     'position_id': i + 3},
                rew=i,
                done=i % 3 == 0,
                info={'if': i})
    indice = np.arange(len(buf))
    orig = np.arange(len(buf))
    data = buf[indice]
    data2 = buf[indice]
    assert isinstance(data, Batch)
    assert isinstance(data2, Batch)
    assert np.allclose(indice, orig)
    assert np.allclose(data.obs_next.mask, data2.obs_next.mask)
    assert np.allclose(data.obs_next.mask, [0, 2, 3, 3, 5, 6, 6, 8, 9, 9])
    buf.stack_num = 4
    data = buf[indice]
    data2 = buf[indice]
    assert np.allclose(data.obs_next.mask, data2.obs_next.mask)
    assert np.allclose(data.obs_next.mask, np.array([
        [0, 0, 0, 0], [1, 1, 1, 2], [1, 1, 2, 3], [1, 1, 2, 3],
        [4, 4, 4, 5], [4, 4, 5, 6], [4, 4, 5, 6],
        [7, 7, 7, 8], [7, 7, 8, 9], [7, 7, 8, 9]]))
    assert np.allclose(data.info['if'], data2.info['if'])
    assert np.allclose(data.info['if'], np.array([
        [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 2], [1, 1, 2, 3],
        [4, 4, 4, 4], [4, 4, 4, 5], [4, 4, 5, 6],
        [7, 7, 7, 7], [7, 7, 7, 8], [7, 7, 8, 9]]))
    assert data.obs_next


def test_stack(size=5, bufsize=9, stack_num=4):
    env = MyTestEnv(size)
    buf = ReplayBuffer(bufsize, stack_num=stack_num)
    buf2 = ReplayBuffer(bufsize, stack_num=stack_num, sample_avail=True)
    obs = env.reset(1)
    for i in range(16):
        obs_next, rew, done, info = env.step(1)
        buf.add(obs, 1, rew, done, None, info)
        buf2.add(obs, 1, rew, done, None, info)
        obs = obs_next
        if done:
            obs = env.reset(1)
    indice = np.arange(len(buf))
    assert np.allclose(buf.get(indice, 'obs'), np.expand_dims(
        [[1, 1, 1, 2], [1, 1, 2, 3], [1, 2, 3, 4],
         [1, 1, 1, 1], [1, 1, 1, 2], [1, 1, 2, 3],
         [1, 2, 3, 4], [4, 4, 4, 4], [1, 1, 1, 1]], axis=-1))
    _, indice = buf2.sample(0)
    assert indice.tolist() == [2, 6]
    _, indice = buf2.sample(1)
    assert indice in [2, 6]


def test_priortized_replaybuffer(size=32, bufsize=15):
    env = MyTestEnv(size)
    buf = PrioritizedReplayBuffer(bufsize, 0.5, 0.5)
    obs = env.reset()
    action_list = [1] * 5 + [0] * 10 + [1] * 10
    for i, a in enumerate(action_list):
        obs_next, rew, done, info = env.step(a)
        buf.add(obs, a, rew, done, obs_next, info, np.random.randn() - 0.5)
        obs = obs_next
        data, indice = buf.sample(len(buf) // 2)
        if len(buf) // 2 == 0:
            assert len(data) == len(buf)
        else:
            assert len(data) == len(buf) // 2
        assert len(buf) == min(bufsize, i + 1)
    data, indice = buf.sample(len(buf) // 2)
    buf.update_weight(indice, -data.weight / 2)
    assert np.allclose(
        buf.weight[indice], np.abs(-data.weight / 2) ** buf._alpha)


def test_update():
    buf1 = ReplayBuffer(4, stack_num=2)
    buf2 = ReplayBuffer(4, stack_num=2)
    for i in range(5):
        buf1.add(obs=np.array([i]), act=float(i), rew=i * i,
                 done=i % 2 == 0, info={'incident': 'found'})
    assert len(buf1) > len(buf2)
    buf2.update(buf1)
    assert len(buf1) == len(buf2)
    assert (buf2[0].obs == buf1[1].obs).all()
    assert (buf2[-1].obs == buf1[0].obs).all()


def test_segtree():
    for op, init in zip(['sum', 'max', 'min'], [0., -np.inf, np.inf]):
        realop = getattr(np, op)
        # small test
        actual_len = 8
        tree = SegmentTree(actual_len, op)  # 1-15. 8-15 are leaf nodes
        assert np.all([tree[i] == init for i in range(actual_len)])
        with pytest.raises(IndexError):
            tree[actual_len]
        naive = np.full([actual_len], init)
        for _ in range(1000):
            # random choose a place to perform single update
            index = np.random.randint(actual_len)
            value = np.random.rand()
            naive[index] = value
            tree[index] = value
            for i in range(actual_len):
                for j in range(i + 1, actual_len):
                    ref = realop(naive[i:j])
                    out = tree.reduce(i, j)
                    assert np.allclose(ref, out)
        # batch setitem
        for _ in range(1000):
            index = np.random.choice(actual_len, size=4)
            value = np.random.rand(4)
            naive[index] = value
            tree[index] = value
            assert np.allclose(realop(naive), tree.reduce())
            for i in range(10):
                left = np.random.randint(actual_len)
                right = np.random.randint(left + 1, actual_len + 1)
                assert np.allclose(realop(naive[left:right]),
                                   tree.reduce(left, right))
        # large test
        actual_len = 16384
        tree = SegmentTree(actual_len, op)
        naive = np.full([actual_len], init)
        for _ in range(1000):
            index = np.random.choice(actual_len, size=64)
            value = np.random.rand(64)
            naive[index] = value
            tree[index] = value
            assert np.allclose(realop(naive), tree.reduce())
            for i in range(10):
                left = np.random.randint(actual_len)
                right = np.random.randint(left + 1, actual_len + 1)
                assert np.allclose(realop(naive[left:right]),
                                   tree.reduce(left, right))

    # test prefix-sum-idx
    actual_len = 8
    tree = SegmentTree(actual_len)
    naive = np.random.rand(actual_len)
    tree[np.arange(actual_len)] = naive
    for _ in range(1000):
        scalar = np.random.rand() * naive.sum()
        index = tree.get_prefix_sum_idx(scalar)
        assert naive[:index].sum() <= scalar <= naive[:index + 1].sum()
    # corner case here
    naive = np.ones(actual_len, np.int)
    tree[np.arange(actual_len)] = naive
    for scalar in range(actual_len):
        index = tree.get_prefix_sum_idx(scalar * 1.)
        assert naive[:index].sum() <= scalar <= naive[:index + 1].sum()
    tree = SegmentTree(10)
    tree[np.arange(3)] = np.array([0.1, 0, 0.1])
    assert np.allclose(tree.get_prefix_sum_idx(
        np.array([0, .1, .1 + 1e-6, .2 - 1e-6])), [0, 0, 2, 2])
    with pytest.raises(AssertionError):
        tree.get_prefix_sum_idx(.2)
    # test large prefix-sum-idx
    actual_len = 16384
    tree = SegmentTree(actual_len)
    naive = np.random.rand(actual_len)
    tree[np.arange(actual_len)] = naive
    for _ in range(1000):
        scalar = np.random.rand() * naive.sum()
        index = tree.get_prefix_sum_idx(scalar)
        assert naive[:index].sum() <= scalar <= naive[:index + 1].sum()

    # profile
    if __name__ == '__main__':
        size = 100000
        bsz = 64
        naive = np.random.rand(size)
        tree = SegmentTree(size)
        tree[np.arange(size)] = naive

        def sample_npbuf():
            return np.random.choice(size, bsz, p=naive / naive.sum())

        def sample_tree():
            scalar = np.random.rand(bsz) * tree.reduce()
            return tree.get_prefix_sum_idx(scalar)

        print('npbuf', timeit(sample_npbuf, setup=sample_npbuf, number=1000))
        print('tree', timeit(sample_tree, setup=sample_tree, number=1000))


def test_pickle():
    size = 100
    vbuf = ReplayBuffer(size, stack_num=2)
    lbuf = ListReplayBuffer()
    pbuf = PrioritizedReplayBuffer(size, 0.6, 0.4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rew = torch.tensor([1.]).to(device)
    for i in range(4):
        vbuf.add(obs=Batch(index=np.array([i])), act=0, rew=rew, done=0)
    for i in range(3):
        lbuf.add(obs=Batch(index=np.array([i])), act=1, rew=rew, done=0)
    for i in range(5):
        pbuf.add(obs=Batch(index=np.array([i])),
                 act=2, rew=rew, done=0, weight=np.random.rand())
    # save & load
    _vbuf = pickle.loads(pickle.dumps(vbuf))
    _lbuf = pickle.loads(pickle.dumps(lbuf))
    _pbuf = pickle.loads(pickle.dumps(pbuf))
    assert len(_vbuf) == len(vbuf) and np.allclose(_vbuf.act, vbuf.act)
    assert len(_lbuf) == len(lbuf) and np.allclose(_lbuf.act, lbuf.act)
    assert len(_pbuf) == len(pbuf) and np.allclose(_pbuf.act, pbuf.act)
    # make sure the meta var is identical
    assert _vbuf.stack_num == vbuf.stack_num
    assert np.allclose(_pbuf.weight[np.arange(len(_pbuf))],
                       pbuf.weight[np.arange(len(pbuf))])


if __name__ == '__main__':
    test_replaybuffer()
    test_ignore_obs_next()
    test_stack()
    test_pickle()
    test_segtree()
    test_priortized_replaybuffer()
    test_priortized_replaybuffer(233333, 200000)
    test_update()
