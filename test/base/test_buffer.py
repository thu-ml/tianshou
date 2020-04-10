import numpy as np
from tianshou.data import ReplayBuffer

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
        assert len(buf) == min(bufsize, i + 1), print(len(buf), i)
    data, indice = buf.sample(bufsize * 2)
    assert (indice < len(buf)).all()
    assert (data.obs < size).all()
    assert (0 <= data.done).all() and (data.done <= 1).all()
    assert len(buf) > len(buf2)
    buf2.update(buf)
    assert len(buf) == len(buf2)
    assert buf2[0].obs == buf[5].obs
    assert buf2[-1].obs == buf[4].obs


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
    assert abs(buf.get(indice, 'obs') - np.array([
        [1, 1, 1, 2], [1, 1, 2, 3], [1, 2, 3, 4],
        [1, 1, 1, 1], [1, 1, 1, 2], [1, 1, 2, 3],
        [3, 3, 3, 3], [3, 3, 3, 4], [1, 1, 1, 1]])).sum() < 1e-6
    print(buf)


if __name__ == '__main__':
    test_replaybuffer()
    test_stack()
