from tianshou.data import ReplayBuffer
from test.test_env import MyTestEnv


def test_replaybuffer(bufsize=20):
    env = MyTestEnv(10)
    buf = ReplayBuffer(bufsize)
    obs = env.reset()
    action_list = [1] * 5 + [0] * 10 + [1] * 9
    for i, a in enumerate(action_list):
        obs_next, rew, done, info = env.step(a)
        buf.add(obs, a, rew, done, obs_next, info)
        assert len(buf) == min(bufsize, i + 1), print(len(buf), i)
    indice = buf.sample_indice(4)
    data = buf.sample(4)


if __name__ == '__main__':
    test_replaybuffer()