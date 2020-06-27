import numpy as np
from tianshou.data import Batch, ReplayBuffer, PrioritizedReplayBuffer

if __name__ == '__main__':
    from env import MyTestEnv
else:  # pytest
    from test.base.env import MyTestEnv


def test_replaybuffer(size=10, bufsize=20):
    env = MyTestEnv(size)
    buf = ReplayBuffer(bufsize)
    buf2 = ReplayBuffer(bufsize)
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
    assert len(buf) > len(buf2)
    buf2.update(buf)
    assert len(buf) == len(buf2)
    assert buf2[0].obs == buf[5].obs
    assert buf2[-1].obs == buf[4].obs
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
    buf = ReplayBuffer(bufsize, stack_num)
    obs = env.reset(1)
    for i in range(15):
        obs_next, rew, done, info = env.step(1)
        buf.add(obs, 1, rew, done, None, info)
        obs = obs_next
        if done:
            obs = env.reset(1)
    indice = np.arange(len(buf))
    assert np.allclose(buf.get(indice, 'obs'), np.array([
        [1, 1, 1, 2], [1, 1, 2, 3], [1, 2, 3, 4],
        [1, 1, 1, 1], [1, 1, 1, 2], [1, 1, 2, 3],
        [3, 3, 3, 3], [3, 3, 3, 4], [1, 1, 1, 1]]))
    print(buf)


def test_priortized_replaybuffer(size=32, bufsize=15):
    env = MyTestEnv(size)
    buf = PrioritizedReplayBuffer(bufsize, 0.5, 0.5)
    obs = env.reset()
    action_list = [1] * 5 + [0] * 10 + [1] * 10
    for i, a in enumerate(action_list):
        obs_next, rew, done, info = env.step(a)
        buf.add(obs, a, rew, done, obs_next, info, np.random.randn() - 0.5)
        obs = obs_next
        assert np.isclose(np.sum((buf.weight / buf._weight_sum)[:buf._size]),
                          1, rtol=1e-12)
        data, indice = buf.sample(len(buf) // 2)
        if len(buf) // 2 == 0:
            assert len(data) == len(buf)
        else:
            assert len(data) == len(buf) // 2
        assert len(buf) == min(bufsize, i + 1)
        assert np.isclose(buf._weight_sum, (buf.weight).sum())
    data, indice = buf.sample(len(buf) // 2)
    buf.update_weight(indice, -data.weight / 2)
    assert np.isclose(buf.weight[indice], np.power(
        np.abs(-data.weight / 2), buf._alpha)).all()
    assert np.isclose(buf._weight_sum, (buf.weight).sum())


if __name__ == '__main__':
    test_replaybuffer()
    test_ignore_obs_next()
    test_stack()
    test_priortized_replaybuffer(233333, 200000)
