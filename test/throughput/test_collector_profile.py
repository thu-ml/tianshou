import numpy as np
import tqdm

from tianshou.data import AsyncCollector, Batch, Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import BasePolicy

if __name__ == "__main__":
    from env import MyTestEnv
else:  # pytest
    from test.base.env import MyTestEnv


class MyPolicy(BasePolicy):
    def __init__(self, dict_state=False, need_state=True):
        """Mock policy for testing.

        :param bool dict_state: if the observation of the environment is a dict
        :param bool need_state: if the policy needs the hidden state (for RNN)
        """
        super().__init__()
        self.dict_state = dict_state
        self.need_state = need_state

    def forward(self, batch, state=None):
        if self.need_state:
            if state is None:
                state = np.zeros((len(batch.obs), 2))
            else:
                state += 1
        if self.dict_state:
            return Batch(act=np.ones(len(batch.obs["index"])), state=state)
        return Batch(act=np.ones(len(batch.obs)), state=state)

    def learn(self):
        pass


def test_collector_nstep():
    policy = MyPolicy()
    env_fns = [lambda x=i: MyTestEnv(size=x) for i in np.arange(2, 11)]
    dum = DummyVectorEnv(env_fns)
    num = len(env_fns)
    c3 = Collector(policy, dum, VectorReplayBuffer(total_size=40000, buffer_num=num))
    for i in tqdm.trange(1, 400, desc="test step collector n_step"):
        c3.reset()
        result = c3.collect(n_step=i * len(env_fns))
        assert result["n/st"] >= i


def test_collector_nepisode():
    policy = MyPolicy()
    env_fns = [lambda x=i: MyTestEnv(size=x) for i in np.arange(2, 11)]
    dum = DummyVectorEnv(env_fns)
    num = len(env_fns)
    c3 = Collector(policy, dum, VectorReplayBuffer(total_size=40000, buffer_num=num))
    for i in tqdm.trange(1, 400, desc="test step collector n_episode"):
        c3.reset()
        result = c3.collect(n_episode=i)
        assert result["n/ep"] == i
        assert result["n/st"] == len(c3.buffer)


def test_asynccollector():
    env_lens = [2, 3, 4, 5]
    env_fns = [lambda x=i: MyTestEnv(size=x, sleep=0.001, random_sleep=True) for i in env_lens]

    venv = SubprocVectorEnv(env_fns, wait_num=len(env_fns) - 1)
    policy = MyPolicy()
    bufsize = 300
    c1 = AsyncCollector(policy, venv, VectorReplayBuffer(total_size=bufsize * 4, buffer_num=4))
    ptr = [0, 0, 0, 0]
    for n_episode in tqdm.trange(1, 100, desc="test async n_episode"):
        result = c1.collect(n_episode=n_episode)
        assert result["n/ep"] >= n_episode
        # check buffer data, obs and obs_next, env_id
        for i, count in enumerate(np.bincount(result["lens"], minlength=6)[2:]):
            env_len = i + 2
            total = env_len * count
            indices = np.arange(ptr[i], ptr[i] + total) % bufsize
            ptr[i] = (ptr[i] + total) % bufsize
            seq = np.arange(env_len)
            buf = c1.buffer.buffers[i]
            assert np.all(buf.info.env_id[indices] == i)
            assert np.all(buf.obs[indices].reshape(count, env_len) == seq)
            assert np.all(buf.obs_next[indices].reshape(count, env_len) == seq + 1)
    # test async n_step, for now the buffer should be full of data
    for n_step in tqdm.trange(1, 150, desc="test async n_step"):
        result = c1.collect(n_step=n_step)
        assert result["n/st"] >= n_step
        for i in range(4):
            env_len = i + 2
            seq = np.arange(env_len)
            buf = c1.buffer.buffers[i]
            assert np.all(buf.info.env_id == i)
            assert np.all(buf.obs.reshape(-1, env_len) == seq)
            assert np.all(buf.obs_next.reshape(-1, env_len) == seq + 1)


if __name__ == "__main__":
    test_collector_nstep()
    test_collector_nepisode()
    test_asynccollector()
