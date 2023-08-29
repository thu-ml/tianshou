import os
import pickle
import tempfile
from timeit import timeit

import h5py
import numpy as np
import pytest
import torch

from tianshou.data import (
    Batch,
    CachedReplayBuffer,
    HERReplayBuffer,
    HERVectorReplayBuffer,
    PrioritizedReplayBuffer,
    PrioritizedVectorReplayBuffer,
    ReplayBuffer,
    SegmentTree,
    VectorReplayBuffer,
)
from tianshou.data.utils.converter import to_hdf5

if __name__ == "__main__":
    from env import MyGoalEnv, MyTestEnv
else:  # pytest
    from test.base.env import MyGoalEnv, MyTestEnv


def test_replaybuffer(size=10, bufsize=20):
    env = MyTestEnv(size)
    buf = ReplayBuffer(bufsize)
    buf.update(buf)
    assert str(buf) == buf.__class__.__name__ + "()"
    obs, _ = env.reset()
    action_list = [1] * 5 + [0] * 10 + [1] * 10
    for i, act in enumerate(action_list):
        obs_next, rew, terminated, truncated, info = env.step(act)
        buf.add(
            Batch(
                obs=obs,
                act=[act],
                rew=rew,
                terminated=terminated,
                truncated=truncated,
                obs_next=obs_next,
                info=info,
            ),
        )
        obs = obs_next
        assert len(buf) == min(bufsize, i + 1)
    assert buf.act.dtype == int
    assert buf.act.shape == (bufsize, 1)
    data, indices = buf.sample(bufsize * 2)
    assert (indices < len(buf)).all()
    assert (data.obs < size).all()
    assert (data.done >= 0).all()
    assert (data.done <= 1).all()
    assert (data.terminated >= 0).all()
    assert (data.terminated <= 1).all()
    assert (data.truncated >= 0).all()
    assert (data.truncated <= 1).all()
    b = ReplayBuffer(size=10)
    # neg bsz should return empty index
    assert b.sample_indices(-1).tolist() == []
    ptr, ep_rew, ep_len, ep_idx = b.add(
        Batch(
            obs=1,
            act=1,
            rew=1,
            terminated=1,
            truncated=0,
            obs_next="str",
            info={"a": 3, "b": {"c": 5.0}},
        ),
    )
    assert b.obs[0] == 1
    assert b.done[0]
    assert b.terminated[0]
    assert not b.truncated[0]
    assert b.obs_next[0] == "str"
    assert np.all(b.obs[1:] == 0)
    assert np.all(b.obs_next[1:] == np.array(None))
    assert b.info.a[0] == 3
    assert b.info.a.dtype == int
    assert np.all(b.info.a[1:] == 0)
    assert b.info.b.c[0] == 5.0
    assert b.info.b.c.dtype == float
    assert np.all(b.info.b.c[1:] == 0.0)
    assert ptr.shape == (1,)
    assert ptr[0] == 0
    assert ep_rew.shape == (1,)
    assert ep_rew[0] == 1
    assert ep_len.shape == (1,)
    assert ep_len[0] == 1
    assert ep_idx.shape == (1,)
    assert ep_idx[0] == 0
    # test extra keys pop up, the buffer should handle it dynamically
    batch = Batch(
        obs=2,
        act=2,
        rew=2,
        terminated=0,
        truncated=0,
        obs_next="str2",
        info={"a": 4, "d": {"e": -np.inf}},
    )
    b.add(batch)
    info_keys = ["a", "b", "d"]
    assert set(b.info.keys()) == set(info_keys)
    assert b.info.a[1] == 4
    assert b.info.b.c[1] == 0
    assert b.info.d.e[1] == -np.inf
    # test batch-style adding method, where len(batch) == 1
    batch.done = [1]
    batch.terminated = [0]
    batch.truncated = [1]
    batch.info.e = np.zeros([1, 4])
    batch = Batch.stack([batch])
    ptr, ep_rew, ep_len, ep_idx = b.add(batch, buffer_ids=[0])
    assert ptr.shape == (1,)
    assert ptr[0] == 2
    assert ep_rew.shape == (1,)
    assert ep_rew[0] == 4
    assert ep_len.shape == (1,)
    assert ep_len[0] == 2
    assert ep_idx.shape == (1,)
    assert ep_idx[0] == 1
    assert set(b.info.keys()) == {*info_keys, "e"}
    assert b.info.e.shape == (b.maxsize, 1, 4)
    with pytest.raises(IndexError):
        b[22]
    # test prev / next
    assert np.all(b.prev(np.array([0, 1, 2])) == [0, 1, 1])
    assert np.all(b.next(np.array([0, 1, 2])) == [0, 2, 2])
    batch.done = [0]
    b.add(batch, buffer_ids=[0])
    assert np.all(b.prev(np.array([0, 1, 2, 3])) == [0, 1, 1, 3])
    assert np.all(b.next(np.array([0, 1, 2, 3])) == [0, 2, 2, 3])


def test_ignore_obs_next(size=10):
    # Issue 82
    buf = ReplayBuffer(size, ignore_obs_next=True)
    for i in range(size):
        buf.add(
            Batch(
                obs={
                    "mask1": np.array([i, 1, 1, 0, 0]),
                    "mask2": np.array([i + 4, 0, 1, 0, 0]),
                    "mask": i,
                },
                act={"act_id": i, "position_id": i + 3},
                rew=i,
                terminated=i % 3 == 0,
                truncated=False,
                info={"if": i},
            ),
        )
    indices = np.arange(len(buf))
    orig = np.arange(len(buf))
    data = buf[indices]
    data2 = buf[indices]
    assert isinstance(data, Batch)
    assert isinstance(data2, Batch)
    assert np.allclose(indices, orig)
    assert np.allclose(data.obs_next.mask, data2.obs_next.mask)
    assert np.allclose(data.obs_next.mask, [0, 2, 3, 3, 5, 6, 6, 8, 9, 9])
    buf.stack_num = 4
    data = buf[indices]
    data2 = buf[indices]
    assert np.allclose(data.obs_next.mask, data2.obs_next.mask)
    assert np.allclose(
        data.obs_next.mask,
        np.array(
            [
                [0, 0, 0, 0],
                [1, 1, 1, 2],
                [1, 1, 2, 3],
                [1, 1, 2, 3],
                [4, 4, 4, 5],
                [4, 4, 5, 6],
                [4, 4, 5, 6],
                [7, 7, 7, 8],
                [7, 7, 8, 9],
                [7, 7, 8, 9],
            ],
        ),
    )
    assert np.allclose(data.info["if"], data2.info["if"])
    assert np.allclose(
        data.info["if"],
        np.array(
            [
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 2],
                [1, 1, 2, 3],
                [4, 4, 4, 4],
                [4, 4, 4, 5],
                [4, 4, 5, 6],
                [7, 7, 7, 7],
                [7, 7, 7, 8],
                [7, 7, 8, 9],
            ],
        ),
    )
    assert data.obs_next


def test_stack(size=5, bufsize=9, stack_num=4, cached_num=3):
    env = MyTestEnv(size)
    buf = ReplayBuffer(bufsize, stack_num=stack_num)
    buf2 = ReplayBuffer(bufsize, stack_num=stack_num, sample_avail=True)
    buf3 = ReplayBuffer(bufsize, stack_num=stack_num, save_only_last_obs=True)
    obs, info = env.reset(options={"state": 1})
    for _ in range(16):
        obs_next, rew, terminated, truncated, info = env.step(1)
        done = terminated or truncated
        buf.add(
            Batch(
                obs=obs,
                act=1,
                rew=rew,
                terminated=terminated,
                truncated=truncated,
                info=info,
            ),
        )
        buf2.add(
            Batch(
                obs=obs,
                act=1,
                rew=rew,
                terminated=terminated,
                truncated=truncated,
                info=info,
            ),
        )
        buf3.add(
            Batch(
                obs=[obs, obs, obs],
                act=1,
                rew=rew,
                terminated=terminated,
                truncated=truncated,
                obs_next=[obs, obs],
                info=info,
            ),
        )
        obs = obs_next
        if done:
            obs, info = env.reset(options={"state": 1})
    indices = np.arange(len(buf))
    assert np.allclose(
        buf.get(indices, "obs")[..., 0],
        [
            [1, 1, 1, 2],
            [1, 1, 2, 3],
            [1, 2, 3, 4],
            [1, 1, 1, 1],
            [1, 1, 1, 2],
            [1, 1, 2, 3],
            [1, 2, 3, 4],
            [4, 4, 4, 4],
            [1, 1, 1, 1],
        ],
    )
    assert np.allclose(buf.get(indices, "obs"), buf3.get(indices, "obs"))
    assert np.allclose(buf.get(indices, "obs"), buf3.get(indices, "obs_next"))
    _, indices = buf2.sample(0)
    assert indices.tolist() == [2, 6]
    _, indices = buf2.sample(1)
    assert indices[0] in [2, 6]
    batch, indices = buf2.sample(-1)  # neg bsz -> no data
    assert indices.tolist() == []
    assert len(batch) == 0
    with pytest.raises(IndexError):
        buf[bufsize * 2]


def test_priortized_replaybuffer(size=32, bufsize=15):
    env = MyTestEnv(size)
    buf = PrioritizedReplayBuffer(bufsize, 0.5, 0.5)
    buf2 = PrioritizedVectorReplayBuffer(bufsize, buffer_num=3, alpha=0.5, beta=0.5)
    obs, info = env.reset()
    action_list = [1] * 5 + [0] * 10 + [1] * 10
    for i, act in enumerate(action_list):
        obs_next, rew, terminated, truncated, info = env.step(act)
        batch = Batch(
            obs=obs,
            act=act,
            rew=rew,
            terminated=terminated,
            truncated=truncated,
            obs_next=obs_next,
            info=info,
            policy=np.random.randn() - 0.5,
        )
        batch_stack = Batch.stack([batch, batch, batch])
        buf.add(Batch.stack([batch]), buffer_ids=[0])
        buf2.add(batch_stack, buffer_ids=[0, 1, 2])
        obs = obs_next
        data, indices = buf.sample(len(buf) // 2)
        if len(buf) // 2 == 0:
            assert len(data) == len(buf)
        else:
            assert len(data) == len(buf) // 2
        assert len(buf) == min(bufsize, i + 1)
        assert len(buf2) == min(bufsize, 3 * (i + 1))
    # check single buffer's data
    assert buf.info.key.shape == (buf.maxsize,)
    assert buf.rew.dtype == float
    assert buf.done.dtype == bool
    assert buf.terminated.dtype == bool
    assert buf.truncated.dtype == bool
    data, indices = buf.sample(len(buf) // 2)
    buf.update_weight(indices, -data.weight / 2)
    assert np.allclose(buf.weight[indices], np.abs(-data.weight / 2) ** buf._alpha)
    # check multi buffer's data
    assert np.allclose(buf2[np.arange(buf2.maxsize)].weight, 1)
    batch, indices = buf2.sample(10)
    buf2.update_weight(indices, batch.weight * 0)
    weight = buf2[np.arange(buf2.maxsize)].weight
    mask = np.isin(np.arange(buf2.maxsize), indices)
    assert np.all(weight[mask] == weight[mask][0])
    assert np.all(weight[~mask] == weight[~mask][0])
    assert weight[~mask][0] < weight[mask][0]
    assert weight[mask][0] <= 1


def test_herreplaybuffer(size=10, bufsize=100, sample_sz=4):
    env_size = size
    env = MyGoalEnv(env_size, array_state=True)

    def compute_reward_fn(ag, g):
        return env.compute_reward_fn(ag, g, {})

    buf = HERReplayBuffer(bufsize, compute_reward_fn=compute_reward_fn, horizon=30, future_k=8)
    buf2 = HERVectorReplayBuffer(
        bufsize,
        buffer_num=3,
        compute_reward_fn=compute_reward_fn,
        horizon=30,
        future_k=8,
    )
    # Apply her on every episodes sampled (Hacky but necessary for deterministic test)
    buf.future_p = 1
    for buf2_buf in buf2.buffers:
        buf2_buf.future_p = 1

    obs, _ = env.reset()
    action_list = [1] * 5 + [0] * 10 + [1] * 10
    for i, act in enumerate(action_list):
        obs_next, rew, terminated, truncated, info = env.step(act)
        batch = Batch(
            obs=obs,
            act=[act],
            rew=rew,
            terminated=terminated,
            truncated=truncated,
            obs_next=obs_next,
            info=info,
        )
        buf.add(batch)
        buf2.add(Batch.stack([batch, batch, batch]), buffer_ids=[0, 1, 2])
        obs = obs_next
        assert len(buf) == min(bufsize, i + 1)
        assert len(buf2) == min(bufsize, 3 * (i + 1))

    batch, indices = buf.sample(sample_sz)

    # Check that goals are the same for the episode (only 1 ep in buffer)
    tmp_indices = indices.copy()
    for _ in range(2 * env_size):
        obs = buf[tmp_indices].obs
        obs_next = buf[tmp_indices].obs_next
        rew = buf[tmp_indices].rew
        g = obs.desired_goal.reshape(sample_sz, -1)[:, 0]
        ag_next = obs_next.achieved_goal.reshape(sample_sz, -1)[:, 0]
        g_next = obs_next.desired_goal.reshape(sample_sz, -1)[:, 0]
        assert np.all(g == g[0])
        assert np.all(g_next == g_next[0])
        assert np.all(rew == (ag_next == g).astype(np.float32))
        tmp_indices = buf.next(tmp_indices)

    # Check that goals are correctly restored
    buf._restore_cache()
    tmp_indices = indices.copy()
    for _ in range(2 * env_size):
        obs = buf[tmp_indices].obs
        obs_next = buf[tmp_indices].obs_next
        g = obs.desired_goal.reshape(sample_sz, -1)[:, 0]
        g_next = obs_next.desired_goal.reshape(sample_sz, -1)[:, 0]
        assert np.all(g == env_size)
        assert np.all(g_next == g_next[0])
        assert np.all(g == g[0])
        tmp_indices = buf.next(tmp_indices)

    # Test vector buffer
    batch, indices = buf2.sample(sample_sz)

    # Check that goals are the same for the episode (only 1 ep in buffer)
    tmp_indices = indices.copy()
    for _ in range(2 * env_size):
        obs = buf2[tmp_indices].obs
        obs_next = buf2[tmp_indices].obs_next
        rew = buf2[tmp_indices].rew
        g = obs.desired_goal.reshape(sample_sz, -1)[:, 0]
        ag_next = obs_next.achieved_goal.reshape(sample_sz, -1)[:, 0]
        g_next = obs_next.desired_goal.reshape(sample_sz, -1)[:, 0]
        assert np.all(g == g_next)
        assert np.all(rew == (ag_next == g).astype(np.float32))
        tmp_indices = buf2.next(tmp_indices)

    # Check that goals are correctly restored
    buf2._restore_cache()
    tmp_indices = indices.copy()
    for _ in range(2 * env_size):
        obs = buf2[tmp_indices].obs
        obs_next = buf2[tmp_indices].obs_next
        g = obs.desired_goal.reshape(sample_sz, -1)[:, 0]
        g_next = obs_next.desired_goal.reshape(sample_sz, -1)[:, 0]
        assert np.all(g == env_size)
        assert np.all(g_next == g_next[0])
        assert np.all(g == g[0])
        tmp_indices = buf2.next(tmp_indices)

    # Test handling cycled indices
    env_size = size
    bufsize = 15
    env = MyGoalEnv(env_size, array_state=False)

    def compute_reward_fn(ag, g):
        return env.compute_reward_fn(ag, g, {})

    buf = HERReplayBuffer(bufsize, compute_reward_fn=compute_reward_fn, horizon=30, future_k=8)
    buf._index = 5  # shifted start index
    buf.future_p = 1
    action_list = [1] * 10
    for ep_len in [5, 10]:
        obs, _ = env.reset()
        for i in range(ep_len):
            act = 1
            obs_next, rew, terminated, truncated, info = env.step(act)
            batch = Batch(
                obs=obs,
                act=[act],
                rew=rew,
                terminated=(i == ep_len - 1),
                truncated=(i == ep_len - 1),
                obs_next=obs_next,
                info=info,
            )
            buf.add(batch)
            obs = obs_next
    batch, indices = buf.sample(0)
    assert np.all(buf[:5].obs.desired_goal == buf[0].obs.desired_goal)
    assert np.all(buf[5:10].obs.desired_goal == buf[5].obs.desired_goal)
    assert np.all(buf[10:].obs.desired_goal == buf[0].obs.desired_goal)  # (same ep)
    assert np.all(buf[0].obs.desired_goal != buf[5].obs.desired_goal)  # (diff ep)

    # Another test case for cycled indices
    env_size = 99
    bufsize = 15
    env = MyGoalEnv(env_size, array_state=False)
    buf = HERReplayBuffer(bufsize, compute_reward_fn=compute_reward_fn, horizon=30, future_k=8)
    buf.future_p = 1
    for x, ep_len in enumerate([10, 20]):
        obs, _ = env.reset()
        for i in range(ep_len):
            act = 1
            obs_next, rew, terminated, truncated, info = env.step(act)
            batch = Batch(
                obs=obs,
                act=[act],
                rew=rew,
                terminated=(i == ep_len - 1),
                truncated=(i == ep_len - 1),
                obs_next=obs_next,
                info=info,
            )
            if x == 1 and obs["observation"] < 10:
                obs = obs_next
                continue
            buf.add(batch)
            obs = obs_next
    buf._restore_cache()
    sample_indices = np.array([10])  # Suppose the sampled indices is [10]
    buf.rewrite_transitions(sample_indices)
    assert int(buf.obs.desired_goal[10][0]) in [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]


def test_update():
    buf1 = ReplayBuffer(4, stack_num=2)
    buf2 = ReplayBuffer(4, stack_num=2)
    for i in range(5):
        buf1.add(
            Batch(
                obs=np.array([i]),
                act=float(i),
                rew=i * i,
                terminated=i % 2 == 0,
                truncated=False,
                info={"incident": "found"},
            ),
        )
    assert len(buf1) > len(buf2)
    buf2.update(buf1)
    assert len(buf1) == len(buf2)
    assert (buf2[0].obs == buf1[1].obs).all()
    assert (buf2[-1].obs == buf1[0].obs).all()
    b = CachedReplayBuffer(ReplayBuffer(10), 4, 5)
    with pytest.raises(NotImplementedError):
        b.update(b)


def test_segtree():
    realop = np.sum
    # small test
    actual_len = 8
    tree = SegmentTree(actual_len)  # 1-15. 8-15 are leaf nodes
    assert len(tree) == actual_len
    assert np.all([tree[i] == 0.0 for i in range(actual_len)])
    with pytest.raises(IndexError):
        tree[actual_len]
    naive = np.zeros([actual_len])
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
                assert np.allclose(ref, out), (ref, out)
    assert np.allclose(tree.reduce(start=1), realop(naive[1:]))
    assert np.allclose(tree.reduce(end=-1), realop(naive[:-1]))
    # batch setitem
    for _ in range(1000):
        index = np.random.choice(actual_len, size=4)
        value = np.random.rand(4)
        naive[index] = value
        tree[index] = value
        assert np.allclose(realop(naive), tree.reduce())
        for _ in range(10):
            left = np.random.randint(actual_len)
            right = np.random.randint(left + 1, actual_len + 1)
            assert np.allclose(realop(naive[left:right]), tree.reduce(left, right))
    # large test
    actual_len = 16384
    tree = SegmentTree(actual_len)
    naive = np.zeros([actual_len])
    for _ in range(1000):
        index = np.random.choice(actual_len, size=64)
        value = np.random.rand(64)
        naive[index] = value
        tree[index] = value
        assert np.allclose(realop(naive), tree.reduce())
        for _ in range(10):
            left = np.random.randint(actual_len)
            right = np.random.randint(left + 1, actual_len + 1)
            assert np.allclose(realop(naive[left:right]), tree.reduce(left, right))

    # test prefix-sum-idx
    actual_len = 8
    tree = SegmentTree(actual_len)
    naive = np.random.rand(actual_len)
    tree[np.arange(actual_len)] = naive
    for _ in range(1000):
        scalar = np.random.rand() * naive.sum()
        index = tree.get_prefix_sum_idx(scalar)
        assert naive[:index].sum() <= scalar <= naive[: index + 1].sum()
    # corner case here
    naive = np.ones(actual_len, int)
    tree[np.arange(actual_len)] = naive
    for scalar in range(actual_len):
        index = tree.get_prefix_sum_idx(scalar * 1.0)
        assert naive[:index].sum() <= scalar <= naive[: index + 1].sum()
    tree = SegmentTree(10)
    tree[np.arange(3)] = np.array([0.1, 0, 0.1])
    assert np.allclose(
        tree.get_prefix_sum_idx(np.array([0, 0.1, 0.1 + 1e-6, 0.2 - 1e-6])),
        [0, 0, 2, 2],
    )
    with pytest.raises(AssertionError):
        tree.get_prefix_sum_idx(0.2)
    # test large prefix-sum-idx
    actual_len = 16384
    tree = SegmentTree(actual_len)
    naive = np.random.rand(actual_len)
    tree[np.arange(actual_len)] = naive
    for _ in range(1000):
        scalar = np.random.rand() * naive.sum()
        index = tree.get_prefix_sum_idx(scalar)
        assert naive[:index].sum() <= scalar <= naive[: index + 1].sum()

    # profile
    if __name__ == "__main__":
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

        print("npbuf", timeit(sample_npbuf, setup=sample_npbuf, number=1000))
        print("tree", timeit(sample_tree, setup=sample_tree, number=1000))


def test_pickle():
    size = 100
    vbuf = ReplayBuffer(size, stack_num=2)
    pbuf = PrioritizedReplayBuffer(size, 0.6, 0.4)
    rew = np.array([1, 1])
    for i in range(4):
        vbuf.add(
            Batch(
                obs=Batch(index=np.array([i])),
                act=0,
                rew=rew,
                terminated=0,
                truncated=0,
            ),
        )
    for i in range(5):
        pbuf.add(
            Batch(
                obs=Batch(index=np.array([i])),
                act=2,
                rew=rew,
                terminated=0,
                truncated=0,
                info=np.random.rand(),
            ),
        )
    # save & load
    _vbuf = pickle.loads(pickle.dumps(vbuf))
    _pbuf = pickle.loads(pickle.dumps(pbuf))
    assert len(_vbuf) == len(vbuf)
    assert np.allclose(_vbuf.act, vbuf.act)
    assert len(_pbuf) == len(pbuf)
    assert np.allclose(_pbuf.act, pbuf.act)
    # make sure the meta var is identical
    assert _vbuf.stack_num == vbuf.stack_num
    assert np.allclose(_pbuf.weight[np.arange(len(_pbuf))], pbuf.weight[np.arange(len(pbuf))])


def test_hdf5():
    size = 100
    buffers = {
        "array": ReplayBuffer(size, stack_num=2),
        "prioritized": PrioritizedReplayBuffer(size, 0.6, 0.4),
    }
    buffer_types = {k: b.__class__ for k, b in buffers.items()}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    info_t = torch.tensor([1.0]).to(device)
    for i in range(4):
        kwargs = {
            "obs": Batch(index=np.array([i])),
            "act": i,
            "rew": np.array([1, 2]),
            "terminated": i % 3 == 2,
            "truncated": False,
            "done": i % 3 == 2,
            "info": {"number": {"n": i, "t": info_t}, "extra": None},
        }
        buffers["array"].add(Batch(kwargs))
        buffers["prioritized"].add(Batch(kwargs))

    # save
    paths = {}
    for k, buf in buffers.items():
        f, path = tempfile.mkstemp(suffix=".hdf5")
        os.close(f)
        buf.save_hdf5(path)
        paths[k] = path

    # load replay buffer
    _buffers = {k: buffer_types[k].load_hdf5(paths[k]) for k in paths}

    # compare
    for k in buffers:
        assert len(_buffers[k]) == len(buffers[k])
        assert np.allclose(_buffers[k].act, buffers[k].act)
        assert _buffers[k].stack_num == buffers[k].stack_num
        assert _buffers[k].maxsize == buffers[k].maxsize
        assert np.all(_buffers[k]._indices == buffers[k]._indices)
    for k in ["array", "prioritized"]:
        assert _buffers[k]._index == buffers[k]._index
        assert isinstance(buffers[k].get(0, "info"), Batch)
        assert isinstance(_buffers[k].get(0, "info"), Batch)
    for k in ["array"]:
        assert np.all(buffers[k][:].info.number.n == _buffers[k][:].info.number.n)
        assert np.all(buffers[k][:].info.extra == _buffers[k][:].info.extra)

    # raise exception when value cannot be pickled
    data = {"not_supported": lambda x: x * x}
    grp = h5py.Group
    with pytest.raises(NotImplementedError):
        to_hdf5(data, grp)
    # ndarray with data type not supported by HDF5 that cannot be pickled
    data = {"not_supported": np.array(lambda x: x * x)}
    grp = h5py.Group
    with pytest.raises(RuntimeError):
        to_hdf5(data, grp)


def test_replaybuffermanager():
    buf = VectorReplayBuffer(20, 4)
    batch = Batch(
        obs=[1, 2, 3],
        act=[1, 2, 3],
        rew=[1, 2, 3],
        terminated=[0, 0, 1],
        truncated=[0, 0, 0],
    )
    ptr, ep_rew, ep_len, ep_idx = buf.add(batch, buffer_ids=[0, 1, 2])
    assert np.all(ep_len == [0, 0, 1])
    assert np.all(ep_rew == [0, 0, 3])
    assert np.all(ptr == [0, 5, 10])
    assert np.all(ep_idx == [0, 5, 10])
    with pytest.raises(NotImplementedError):
        # ReplayBufferManager cannot be updated
        buf.update(buf)
    # sample index / prev / next / unfinished_index
    indices = buf.sample_indices(11000)
    assert np.bincount(indices)[[0, 5, 10]].min() >= 3000  # uniform sample
    batch, indices = buf.sample(0)
    assert np.allclose(indices, [0, 5, 10])
    indices_prev = buf.prev(indices)
    assert np.allclose(indices_prev, indices), indices_prev
    indices_next = buf.next(indices)
    assert np.allclose(indices_next, indices), indices_next
    assert np.allclose(buf.unfinished_index(), [0, 5])
    buf.add(Batch(obs=[4], act=[4], rew=[4], terminated=[1], truncated=[0]), buffer_ids=[3])
    assert np.allclose(buf.unfinished_index(), [0, 5])
    batch, indices = buf.sample(10)
    batch, indices = buf.sample(0)
    assert np.allclose(indices, [0, 5, 10, 15])
    indices_prev = buf.prev(indices)
    assert np.allclose(indices_prev, indices), indices_prev
    indices_next = buf.next(indices)
    assert np.allclose(indices_next, indices), indices_next
    data = np.array([0, 0, 0, 0])
    buf.add(
        Batch(obs=data, act=data, rew=data, terminated=data, truncated=data),
        buffer_ids=[0, 1, 2, 3],
    )
    buf.add(
        Batch(obs=data, act=data, rew=data, terminated=1 - data, truncated=data),
        buffer_ids=[0, 1, 2, 3],
    )
    assert len(buf) == 12
    buf.add(
        Batch(obs=data, act=data, rew=data, terminated=data, truncated=data),
        buffer_ids=[0, 1, 2, 3],
    )
    buf.add(
        Batch(obs=data, act=data, rew=data, terminated=[0, 1, 0, 1], truncated=data),
        buffer_ids=[0, 1, 2, 3],
    )
    assert len(buf) == 20
    indices = buf.sample_indices(120000)
    assert np.bincount(indices).min() >= 5000
    batch, indices = buf.sample(10)
    indices = buf.sample_indices(0)
    assert np.allclose(indices, np.arange(len(buf)))
    # check the actual data stored in buf._meta
    assert np.allclose(
        buf.done,
        [
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            1,
            1,
            0,
            1,
            0,
            0,
            1,
            0,
            1,
            0,
            1,
        ],
    )
    assert np.allclose(
        buf.prev(indices),
        [
            0,
            0,
            1,
            3,
            3,
            5,
            5,
            6,
            8,
            8,
            10,
            11,
            11,
            13,
            13,
            15,
            16,
            16,
            18,
            18,
        ],
    )
    assert np.allclose(
        buf.next(indices),
        [
            1,
            2,
            2,
            4,
            4,
            6,
            7,
            7,
            9,
            9,
            10,
            12,
            12,
            14,
            14,
            15,
            17,
            17,
            19,
            19,
        ],
    )
    assert np.allclose(buf.unfinished_index(), [4, 14])
    ptr, ep_rew, ep_len, ep_idx = buf.add(
        Batch(obs=[1], act=[1], rew=[1], terminated=[1], truncated=[0]),
        buffer_ids=[2],
    )
    assert np.all(ep_len == [3])
    assert np.all(ep_rew == [1])
    assert np.all(ptr == [10])
    assert np.all(ep_idx == [13])
    assert np.allclose(buf.unfinished_index(), [4])
    indices = sorted(buf.sample_indices(0))
    assert np.allclose(indices, np.arange(len(buf)))
    assert np.allclose(
        buf.prev(indices),
        [
            0,
            0,
            1,
            3,
            3,
            5,
            5,
            6,
            8,
            8,
            14,
            11,
            11,
            13,
            13,
            15,
            16,
            16,
            18,
            18,
        ],
    )
    assert np.allclose(
        buf.next(indices),
        [
            1,
            2,
            2,
            4,
            4,
            6,
            7,
            7,
            9,
            9,
            10,
            12,
            12,
            14,
            10,
            15,
            17,
            17,
            19,
            19,
        ],
    )
    # corner case: list, int and -1
    assert buf.prev(-1) == buf.prev([buf.maxsize - 1])[0]
    assert buf.next(-1) == buf.next([buf.maxsize - 1])[0]
    batch = buf._meta
    batch.info = np.ones(buf.maxsize)
    buf.set_batch(batch)
    assert np.allclose(buf.buffers[-1].info, [1] * 5)
    assert buf.sample_indices(-1).tolist() == []
    assert np.array([ReplayBuffer(0, ignore_obs_next=True)]).dtype == object


def test_cachedbuffer():
    buf = CachedReplayBuffer(ReplayBuffer(10), 4, 5)
    assert buf.sample_indices(0).tolist() == []
    # check the normal function/usage/storage in CachedReplayBuffer
    ptr, ep_rew, ep_len, ep_idx = buf.add(
        Batch(obs=[1], act=[1], rew=[1], terminated=[0], truncated=[0]),
        buffer_ids=[1],
    )
    obs = np.zeros(buf.maxsize)
    obs[15] = 1
    indices = buf.sample_indices(0)
    assert np.allclose(indices, [15])
    assert np.allclose(buf.prev(indices), [15])
    assert np.allclose(buf.next(indices), [15])
    assert np.allclose(buf.obs, obs)
    assert np.all(ep_len == [0])
    assert np.all(ep_rew == [0.0])
    assert np.all(ptr == [15])
    assert np.all(ep_idx == [15])
    ptr, ep_rew, ep_len, ep_idx = buf.add(
        Batch(obs=[2], act=[2], rew=[2], terminated=[1], truncated=[0]),
        buffer_ids=[3],
    )
    obs[[0, 25]] = 2
    indices = buf.sample_indices(0)
    assert np.allclose(indices, [0, 15])
    assert np.allclose(buf.prev(indices), [0, 15])
    assert np.allclose(buf.next(indices), [0, 15])
    assert np.allclose(buf.obs, obs)
    assert np.all(ep_len == [1])
    assert np.all(ep_rew == [2.0])
    assert np.all(ptr == [0])
    assert np.all(ep_idx == [0])
    assert np.allclose(buf.unfinished_index(), [15])
    assert np.allclose(buf.sample_indices(0), [0, 15])
    ptr, ep_rew, ep_len, ep_idx = buf.add(
        Batch(obs=[3, 4], act=[3, 4], rew=[3, 4], terminated=[0, 1], truncated=[0, 0]),
        buffer_ids=[3, 1],  # TODO
    )
    assert np.all(ep_len == [0, 2])
    assert np.all(ep_rew == [0, 5.0])
    assert np.all(ptr == [25, 2])
    assert np.all(ep_idx == [25, 1])
    obs[[0, 1, 2, 15, 16, 25]] = [2, 1, 4, 1, 4, 3]
    assert np.allclose(buf.obs, obs)
    assert np.allclose(buf.unfinished_index(), [25])
    indices = buf.sample_indices(0)
    assert np.allclose(indices, [0, 1, 2, 25])
    assert np.allclose(buf.done[indices], [1, 0, 1, 0])
    assert np.allclose(buf.prev(indices), [0, 1, 1, 25])
    assert np.allclose(buf.next(indices), [0, 2, 2, 25])
    indices = buf.sample_indices(10000)
    assert np.bincount(indices)[[0, 1, 2, 25]].min() > 2000  # uniform sample
    # cached buffer with main_buffer size == 0 (no update)
    # used in test_collector
    buf = CachedReplayBuffer(ReplayBuffer(0, sample_avail=True), 4, 5)
    data = np.zeros(4)
    rew = np.ones([4, 4])
    buf.add(Batch(obs=data, act=data, rew=rew, terminated=[0, 0, 1, 1], truncated=[0, 0, 0, 0]))
    buf.add(Batch(obs=data, act=data, rew=rew, terminated=[0, 0, 0, 0], truncated=[0, 0, 0, 0]))
    buf.add(Batch(obs=data, act=data, rew=rew, terminated=[1, 1, 1, 1], truncated=[0, 0, 0, 0]))
    buf.add(Batch(obs=data, act=data, rew=rew, terminated=[0, 0, 0, 0], truncated=[0, 0, 0, 0]))
    ptr, ep_rew, ep_len, ep_idx = buf.add(
        Batch(obs=data, act=data, rew=rew, terminated=[0, 1, 0, 1], truncated=[0, 0, 0, 0]),
    )
    assert np.all(ptr == [1, -1, 11, -1])
    assert np.all(ep_idx == [0, -1, 10, -1])
    assert np.all(ep_len == [0, 2, 0, 2])
    assert np.all(ep_rew == [data, data + 2, data, data + 2])
    assert np.allclose(
        buf.done,
        [
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
        ],
    )
    indices = buf.sample_indices(0)
    assert np.allclose(indices, [0, 1, 10, 11])
    assert np.allclose(buf.prev(indices), [0, 0, 10, 10])
    assert np.allclose(buf.next(indices), [1, 1, 11, 11])


def test_multibuf_stack():
    size = 5
    bufsize = 9
    stack_num = 4
    cached_num = 3
    env = MyTestEnv(size)
    # test if CachedReplayBuffer can handle stack_num + ignore_obs_next
    buf4 = CachedReplayBuffer(
        ReplayBuffer(bufsize, stack_num=stack_num, ignore_obs_next=True),
        cached_num,
        size,
    )
    # test if CachedReplayBuffer can handle corner case:
    # buffer + stack_num + ignore_obs_next + sample_avail
    buf5 = CachedReplayBuffer(
        ReplayBuffer(bufsize, stack_num=stack_num, ignore_obs_next=True, sample_avail=True),
        cached_num,
        size,
    )
    obs, info = env.reset(options={"state": 1})
    for i in range(18):
        obs_next, rew, terminated, truncated, info = env.step(1)
        done = terminated or truncated
        obs_list = np.array([obs + size * i for i in range(cached_num)])
        act_list = [1] * cached_num
        rew_list = [rew] * cached_num
        terminated_list = [terminated] * cached_num
        truncated_list = [truncated] * cached_num
        obs_next_list = -obs_list
        info_list = [info] * cached_num
        batch = Batch(
            obs=obs_list,
            act=act_list,
            rew=rew_list,
            terminated=terminated_list,
            truncated=truncated_list,
            obs_next=obs_next_list,
            info=info_list,
        )
        buf5.add(batch)
        buf4.add(batch)
        assert np.all(buf4.obs == buf5.obs)
        assert np.all(buf4.done == buf5.done)
        assert np.all(buf4.terminated == buf5.terminated)
        assert np.all(buf4.truncated == buf5.truncated)
        obs = obs_next
        if done:
            obs, info = env.reset(options={"state": 1})
    # check the `add` order is correct
    assert np.allclose(
        buf4.obs.reshape(-1),
        [
            12,
            13,
            14,
            4,
            6,
            7,
            8,
            9,
            11,  # main_buffer
            1,
            2,
            3,
            4,
            0,  # cached_buffer[0]
            6,
            7,
            8,
            9,
            0,  # cached_buffer[1]
            11,
            12,
            13,
            14,
            0,  # cached_buffer[2]
        ],
    ), buf4.obs
    assert np.allclose(
        buf4.done,
        [
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            1,
            0,  # main_buffer
            0,
            0,
            0,
            1,
            0,  # cached_buffer[0]
            0,
            0,
            0,
            1,
            0,  # cached_buffer[1]
            0,
            0,
            0,
            1,
            0,  # cached_buffer[2]
        ],
    ), buf4.done
    assert np.allclose(buf4.unfinished_index(), [10, 15, 20])
    indices = sorted(buf4.sample_indices(0))
    assert np.allclose(indices, [*list(range(bufsize)), 9, 10, 14, 15, 19, 20])
    assert np.allclose(
        buf4[indices].obs[..., 0],
        [
            [11, 11, 11, 12],
            [11, 11, 12, 13],
            [11, 12, 13, 14],
            [4, 4, 4, 4],
            [6, 6, 6, 6],
            [6, 6, 6, 7],
            [6, 6, 7, 8],
            [6, 7, 8, 9],
            [11, 11, 11, 11],
            [1, 1, 1, 1],
            [1, 1, 1, 2],
            [6, 6, 6, 6],
            [6, 6, 6, 7],
            [11, 11, 11, 11],
            [11, 11, 11, 12],
        ],
    )
    assert np.allclose(
        buf4[indices].obs_next[..., 0],
        [
            [11, 11, 12, 13],
            [11, 12, 13, 14],
            [11, 12, 13, 14],
            [4, 4, 4, 4],
            [6, 6, 6, 7],
            [6, 6, 7, 8],
            [6, 7, 8, 9],
            [6, 7, 8, 9],
            [11, 11, 11, 12],
            [1, 1, 1, 2],
            [1, 1, 1, 2],
            [6, 6, 6, 7],
            [6, 6, 6, 7],
            [11, 11, 11, 12],
            [11, 11, 11, 12],
        ],
    )
    indices = buf5.sample_indices(0)
    assert np.allclose(sorted(indices), [2, 7])
    assert np.all(np.isin(buf5.sample_indices(100), indices))
    # manually change the stack num
    buf5.stack_num = 2
    for buf in buf5.buffers:
        buf.stack_num = 2
    indices = buf5.sample_indices(0)
    assert np.allclose(sorted(indices), [0, 1, 2, 5, 6, 7, 10, 15, 20])
    batch, _ = buf5.sample(0)
    # test Atari with CachedReplayBuffer, save_only_last_obs + ignore_obs_next
    buf6 = CachedReplayBuffer(
        ReplayBuffer(bufsize, stack_num=stack_num, save_only_last_obs=True, ignore_obs_next=True),
        cached_num,
        size,
    )
    obs = np.random.rand(size, 4, 84, 84)
    buf6.add(
        Batch(
            obs=[obs[2], obs[0]],
            act=[1, 1],
            rew=[0, 0],
            terminated=[0, 1],
            truncated=[0, 0],
            obs_next=[obs[3], obs[1]],
        ),
        buffer_ids=[1, 2],
    )
    assert buf6.obs.shape == (buf6.maxsize, 84, 84)
    assert np.allclose(buf6.obs[0], obs[0, -1])
    assert np.allclose(buf6.obs[14], obs[2, -1])
    assert np.allclose(buf6.obs[19], obs[0, -1])
    assert buf6[0].obs.shape == (4, 84, 84)


def test_multibuf_hdf5():
    size = 100
    buffers = {
        "vector": VectorReplayBuffer(size * 4, 4),
        "cached": CachedReplayBuffer(ReplayBuffer(size), 4, size),
    }
    buffer_types = {k: b.__class__ for k, b in buffers.items()}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    info_t = torch.tensor([1.0]).to(device)
    for i in range(4):
        kwargs = {
            "obs": Batch(index=np.array([i])),
            "act": i,
            "rew": np.array([1, 2]),
            "terminated": i % 3 == 2,
            "truncated": False,
            "done": i % 3 == 2,
            "info": {"number": {"n": i, "t": info_t}, "extra": None},
        }
        buffers["vector"].add(Batch.stack([kwargs, kwargs, kwargs]), buffer_ids=[0, 1, 2])
        buffers["cached"].add(Batch.stack([kwargs, kwargs, kwargs]), buffer_ids=[0, 1, 2])

    # save
    paths = {}
    for k, buf in buffers.items():
        f, path = tempfile.mkstemp(suffix=".hdf5")
        os.close(f)
        buf.save_hdf5(path)
        paths[k] = path

    # load replay buffer
    _buffers = {k: buffer_types[k].load_hdf5(paths[k]) for k in paths}

    # compare
    for k in buffers:
        assert len(_buffers[k]) == len(buffers[k])
        assert np.allclose(_buffers[k].act, buffers[k].act)
        assert _buffers[k].stack_num == buffers[k].stack_num
        assert _buffers[k].maxsize == buffers[k].maxsize
        assert np.all(_buffers[k]._indices == buffers[k]._indices)
    # check shallow copy in VectorReplayBuffer
    for k in ["vector", "cached"]:
        buffers[k].info.number.n[0] = -100
        assert buffers[k].buffers[0].info.number.n[0] == -100
    # check if still behave normally
    for k in ["vector", "cached"]:
        kwargs = {
            "obs": Batch(index=np.array([5])),
            "act": 5,
            "rew": np.array([2, 1]),
            "terminated": False,
            "truncated": False,
            "done": False,
            "info": {"number": {"n": i}, "Timelimit.truncate": True},
        }
        buffers[k].add(Batch.stack([kwargs, kwargs, kwargs, kwargs]))
        act = np.zeros(buffers[k].maxsize)
        if k == "vector":
            act[np.arange(5)] = np.array([0, 1, 2, 3, 5])
            act[np.arange(5) + size] = np.array([0, 1, 2, 3, 5])
            act[np.arange(5) + size * 2] = np.array([0, 1, 2, 3, 5])
            act[size * 3] = 5
        elif k == "cached":
            act[np.arange(9)] = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
            act[np.arange(3) + size] = np.array([3, 5, 2])
            act[np.arange(3) + size * 2] = np.array([3, 5, 2])
            act[np.arange(3) + size * 3] = np.array([3, 5, 2])
            act[size * 4] = 5
        assert np.allclose(buffers[k].act, act)
        info_keys = ["number", "extra", "Timelimit.truncate"]
        assert set(buffers[k].info.keys()) == set(info_keys)

    for path in paths.values():
        os.remove(path)


def test_from_data():
    obs_data = np.ndarray((10, 3, 3), dtype="uint8")
    for i in range(10):
        obs_data[i] = i * np.ones((3, 3), dtype="uint8")
    obs_next_data = np.zeros_like(obs_data)
    obs_next_data[:-1] = obs_data[1:]
    f, path = tempfile.mkstemp(suffix=".hdf5")
    os.close(f)
    with h5py.File(path, "w") as f:
        obs = f.create_dataset("obs", data=obs_data)
        act = f.create_dataset("act", data=np.arange(10, dtype="int32"))
        rew = f.create_dataset("rew", data=np.arange(10, dtype="float32"))
        terminated = f.create_dataset("terminated", data=np.zeros(10, dtype="bool"))
        truncated = f.create_dataset("truncated", data=np.zeros(10, dtype="bool"))
        done = f.create_dataset("done", data=np.zeros(10, dtype="bool"))
        obs_next = f.create_dataset("obs_next", data=obs_next_data)
        buf = ReplayBuffer.from_data(obs, act, rew, terminated, truncated, done, obs_next)
    assert len(buf) == 10
    batch = buf[3]
    assert np.array_equal(batch.obs, 3 * np.ones((3, 3), dtype="uint8"))
    assert batch.act == 3
    assert batch.rew == 3.0
    assert not batch.done
    assert np.array_equal(batch.obs_next, 4 * np.ones((3, 3), dtype="uint8"))
    os.remove(path)


def test_custom_key():
    batch = Batch(
        obs_next=np.array(
            [
                [
                    1.174,
                    -0.1151,
                    -0.609,
                    -0.5205,
                    -0.9316,
                    3.236,
                    -2.418,
                    0.386,
                    0.2227,
                    -0.5117,
                    2.293,
                ],
            ],
        ),
        rew=np.array([4.28125]),
        act=np.array([[-0.3088, -0.4636, 0.4956]]),
        truncated=np.array([False]),
        obs=np.array(
            [
                [
                    1.193,
                    -0.1203,
                    -0.6123,
                    -0.519,
                    -0.9434,
                    3.32,
                    -2.266,
                    0.9116,
                    0.623,
                    0.1259,
                    0.363,
                ],
            ],
        ),
        terminated=np.array([False]),
        done=np.array([False]),
        returns=np.array([74.70343082]),
        info=Batch(),
        policy=Batch(),
    )
    buffer_size = len(batch.rew)
    buffer = ReplayBuffer(buffer_size)
    buffer.add(batch)
    sampled_batch, _ = buffer.sample(1)
    # Check if they have the same keys
    assert set(batch.keys()) == set(
        sampled_batch.keys(),
    ), "Batches have different keys: {} and {}".format(set(batch.keys()), set(sampled_batch.keys()))
    # Compare the values for each key
    for key in batch.keys():
        if isinstance(batch.__dict__[key], np.ndarray) and isinstance(
            sampled_batch.__dict__[key],
            np.ndarray,
        ):
            assert np.allclose(
                batch.__dict__[key],
                sampled_batch.__dict__[key],
            ), f"Value mismatch for key: {key}"
        if isinstance(batch.__dict__[key], Batch) and isinstance(
            sampled_batch.__dict__[key],
            Batch,
        ):
            assert batch.__dict__[key].is_empty()
            assert sampled_batch.__dict__[key].is_empty()


if __name__ == "__main__":
    test_replaybuffer()
    test_ignore_obs_next()
    test_stack()
    test_segtree()
    test_priortized_replaybuffer()
    test_update()
    test_pickle()
    test_hdf5()
    test_replaybuffermanager()
    test_cachedbuffer()
    test_multibuf_stack()
    test_multibuf_hdf5()
    test_from_data()
    test_herreplaybuffer()
    test_custom_key()
