from tianshou.data import ReplayBuffer
if __name__ == '__main__':
    from test_env import MyTestEnv
else:  # pytest
    from test.test_env import MyTestEnv


def test_replaybuffer(size=10, bufsize=20):
    env = MyTestEnv(size)
    buf = ReplayBuffer(bufsize)
    obs = env.reset()
    action_list = [1] * 5 + [0] * 10 + [1] * 15
    for i, a in enumerate(action_list):
        obs_next, rew, done, info = env.step(a)
        buf.add(obs, a, rew, done, obs_next, info)
        assert len(buf) == min(bufsize, i + 1), print(len(buf), i)
    data, indice = buf.sample(4)
    assert (indice < len(buf)).all()
    assert (data.obs < size).all()
    assert (0 <= data.done).all() and (data.done <= 1).all()


if __name__ == '__main__':
    test_replaybuffer()
