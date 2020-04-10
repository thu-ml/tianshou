import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import BasePolicy
from tianshou.env import SubprocVectorEnv
from tianshou.data import Collector, Batch, ReplayBuffer

if __name__ == '__main__':
    from env import MyTestEnv
else:  # pytest
    from test.base.env import MyTestEnv


class MyPolicy(BasePolicy):
    """docstring for MyPolicy"""

    def __init__(self):
        super().__init__()

    def forward(self, batch, state=None):
        return Batch(act=np.ones(batch.obs.shape[0]))

    def learn(self):
        pass


def equal(a, b):
    return abs(np.array(a) - np.array(b)).sum() < 1e-6


class Logger(object):
    def __init__(self, writer):
        self.cnt = 0
        self.writer = writer

    def log(self, info):
        self.writer.add_scalar('key', info['key'], global_step=self.cnt)
        self.cnt += 1


def test_collector():
    writer = SummaryWriter('log/collector')
    logger = Logger(writer)
    env_fns = [
        lambda: MyTestEnv(size=2, sleep=0),
        lambda: MyTestEnv(size=3, sleep=0),
        lambda: MyTestEnv(size=4, sleep=0),
        lambda: MyTestEnv(size=5, sleep=0),
    ]

    venv = SubprocVectorEnv(env_fns)
    policy = MyPolicy()
    env = env_fns[0]()
    c0 = Collector(policy, env, ReplayBuffer(size=100, ignore_obs_next=False))
    c0.collect(n_step=3, log_fn=logger.log)
    assert equal(c0.buffer.obs[:3], [0, 1, 0])
    assert equal(c0.buffer[:3].obs_next, [1, 2, 1])
    c0.collect(n_episode=3, log_fn=logger.log)
    assert equal(c0.buffer.obs[:8], [0, 1, 0, 1, 0, 1, 0, 1])
    assert equal(c0.buffer[:8].obs_next, [1, 2, 1, 2, 1, 2, 1, 2])
    c1 = Collector(policy, venv, ReplayBuffer(size=100, ignore_obs_next=False))
    c1.collect(n_step=6)
    assert equal(c1.buffer.obs[:11], [0, 1, 0, 1, 2, 0, 1, 0, 1, 2, 3])
    assert equal(c1.buffer[:11].obs_next, [1, 2, 1, 2, 3, 1, 2, 1, 2, 3, 4])
    c1.collect(n_episode=2)
    assert equal(c1.buffer.obs[11:21], [0, 1, 2, 3, 4, 0, 1, 0, 1, 2])
    assert equal(c1.buffer[11:21].obs_next, [1, 2, 3, 4, 5, 1, 2, 1, 2, 3])
    c2 = Collector(policy, venv, ReplayBuffer(size=100, ignore_obs_next=False))
    c2.collect(n_episode=[1, 2, 2, 2])
    assert equal(c2.buffer.obs_next[:26], [
        1, 2, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 5,
        1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 5])
    c2.reset_env()
    c2.collect(n_episode=[2, 2, 2, 2])
    assert equal(c2.buffer.obs_next[26:54], [
        1, 2, 1, 2, 3, 1, 2, 1, 2, 3, 4, 1, 2, 3, 4, 5,
        1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 5])


if __name__ == '__main__':
    test_collector()
