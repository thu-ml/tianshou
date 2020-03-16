from tianshou.data import ReplayBuffer
if __name__ == '__main__':
    from test_env import MyTestEnv
else:  # pytest
    from test.test_env import MyTestEnv


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


if __name__ == '__main__':
    test_replaybuffer()
