import pytest
import numpy as np
from timeit import timeit

from tianshou.data import Batch, PrioritizedReplayBuffer, \
    ReplayBuffer, SegmentTree

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
    buf = ReplayBuffer(size, ignore_obs_net=True)
    for i in range(size):
        buf.add(obs={'mask1': np.array([i, 1, 1, 0, 0]),
                     'mask2': np.array([i + 4, 0, 1, 0, 0])},
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


def test_stack(size=5, bufsize=9, stack_num=4):
    env = MyTestEnv(size)
    buf = ReplayBuffer(bufsize, stack_num=stack_num)
    buf2 = ReplayBuffer(bufsize, stack_num=stack_num, sample_avail=True)
    obs = env.reset(1)
    for i in range(15):
        obs_next, rew, done, info = env.step(1)
        buf.add(obs, 1, rew, done, None, info)
        buf2.add(obs, 1, rew, done, None, info)
        obs = obs_next
        if done:
            obs = env.reset(1)
    indice = np.arange(len(buf))
    assert np.allclose(buf.get(indice, 'obs'), np.array([
        [1, 1, 1, 2], [1, 1, 2, 3], [1, 2, 3, 4],
        [1, 1, 1, 1], [1, 1, 1, 2], [1, 1, 2, 3],
        [3, 3, 3, 3], [3, 3, 3, 4], [1, 1, 1, 1]]))
    print(buf)
    _, indice = buf2.sample(0)
    assert indice == [2]
    _, indice = buf2.sample(1)
    assert indice.sum() == 2


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
                 done=False, info={'incident': 'found'})
    assert len(buf1) > len(buf2)
    buf2.update(buf1)
    assert len(buf1) == len(buf2)
    assert (buf2[0].obs == buf1[1].obs).all()
    assert (buf2[-1].obs == buf1[0].obs).all()


def test_segtree():
    for op, init in zip(['sum', 'max', 'min'], [0., -np.inf, np.inf]):
        realop = getattr(np, op)
        # small test
        tree = SegmentTree(6, op)  # 1-15. 8-15 are leaf nodes
        actual_len = 8
        assert np.all([tree[i] == init for i in range(actual_len)])
        with pytest.raises(AssertionError):
            tree[-1]
        with pytest.raises(AssertionError):
            tree[actual_len]
        naive = np.zeros([actual_len]) + init
        for _ in range(1000):
            # random choose a place to perform single update
            index = np.random.randint(actual_len)
            value = np.random.rand()
            naive[index] = value
            tree[index] = value
            for i in range(actual_len):
                for j in range(i, actual_len):
                    try:
                        ref = realop(naive[i:j])
                    except ValueError:
                        continue
                    out = tree.reduce(i, j)
                    assert np.allclose(ref, out), (i, j, ref, out)
        # batch setitem
        for _ in range(1000):
            index = np.random.choice(actual_len, size=4)
            value = np.random.rand(4)
            naive[index] = value
            tree[index] = value
            assert np.allclose(realop(naive), tree.reduce())
            for i in range(10):
                left = right = 0
                while left >= right:
                    left = np.random.randint(actual_len)
                    right = np.random.randint(actual_len)
                assert np.allclose(realop(naive[left:right]),
                                   tree.reduce(left, right))
        # large test
        tree = SegmentTree(10000, op)
        actual_len = 16384
        naive = np.zeros([actual_len]) + init
        for _ in range(1000):
            index = np.random.choice(actual_len, size=64)
            value = np.random.rand(64)
            naive[index] = value
            tree[index] = value
            assert np.allclose(realop(naive), tree.reduce())
            for i in range(10):
                left = right = 0
                while left >= right:
                    left = np.random.randint(actual_len)
                    right = np.random.randint(actual_len)
                assert np.allclose(realop(naive[left:right]),
                                   tree.reduce(left, right))

    # test prefix-sum-idx
    tree = SegmentTree(6)
    actual_len = 8
    naive = np.random.rand(actual_len)
    tree[np.arange(actual_len)] = naive
    for _ in range(1000):
        scalar = np.random.rand() * naive.sum()
        index = tree.get_prefix_sum_idx(scalar)
        assert naive[:index].sum() <= scalar < naive[:index + 1].sum()
    # corner case here
    naive = np.ones(actual_len, np.int)
    tree[np.arange(actual_len)] = naive
    for scalar in range(actual_len):
        index = tree.get_prefix_sum_idx(scalar * 1.)
        assert naive[:index].sum() <= scalar < naive[:index + 1].sum()
    # test large prefix-sum-idx
    tree = SegmentTree(10000)
    actual_len = 16384
    naive = np.random.rand(actual_len)
    tree[np.arange(actual_len)] = naive
    for _ in range(1000):
        scalar = np.random.rand() * naive.sum()
        index = tree.get_prefix_sum_idx(scalar)
        assert naive[:index].sum() <= scalar < naive[:index + 1].sum()
    # profile
    if __name__ == '__main__':
        naive = np.zeros(10000)
        tree = SegmentTree(10000)


if __name__ == '__main__':
    test_replaybuffer()
    test_ignore_obs_next()
    test_stack()
    test_segtree()
    test_priortized_replaybuffer(233333, 200000)
    test_update()
