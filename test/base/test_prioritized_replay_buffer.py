import numpy as np
from tianshou.data import PrioritizedReplayBuffer

if __name__ == '__main__':
    from env import MyTestEnv
else:  # pytest
    from test.base.env import MyTestEnv


def test_replaybuffer(size=32, bufsize=15):
    env = MyTestEnv(size)
    buf = PrioritizedReplayBuffer(bufsize, 0.5, 0.5)
    obs = env.reset()
    action_list = [1] * 5 + [0] * 10 + [1] * 10
    for i, a in enumerate(action_list):
        obs_next, rew, done, info = env.step(a)
        buf.add(obs, a, rew, done, obs_next, info, np.random.randn()-0.5)
        obs = obs_next
        assert np.isclose(np.sum((buf.weight/buf._weight_sum)[:buf._size]), 1,
                          rtol=1e-12)
        data, indice = buf.sample(len(buf) // 2)
        if len(buf)//2 == 0:
            assert len(data) == len(buf)
        else:
            assert len(data) == len(buf)//2
        assert len(buf) == min(bufsize, i + 1), print(len(buf), i)
        assert np.isclose(buf._weight_sum, (buf.weight).sum())
    data, indice = buf.sample(len(buf) // 2)
    buf.update_weight(indice, -data.weight/2)
    assert np.isclose(buf.weight[indice], np.power(
        np.abs(-data.weight/2), buf._alpha)).all()
    assert np.isclose(buf._weight_sum, (buf.weight).sum())


if __name__ == "__main__":
    test_replaybuffer(233333, 200000)
    print("pass")
