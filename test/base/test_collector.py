import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import BasePolicy
from tianshou.env import VectorEnv, SubprocVectorEnv
from tianshou.data import Collector, Batch, ReplayBuffer

if __name__ == '__main__':
    from env import MyTestEnv
else:  # pytest
    from test.base.env import MyTestEnv


class MyPolicy(BasePolicy):
    def __init__(self, dict_state=False):
        super().__init__()
        self.dict_state = dict_state

    def forward(self, batch, state=None):
        if self.dict_state:
            return Batch(act=np.ones(batch.obs['index'].shape[0]))
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


def test_collector_with_dict_state():
    env = MyTestEnv(size=5, sleep=0, dict_state=True)
    policy = MyPolicy(dict_state=True)
    c0 = Collector(policy, env, ReplayBuffer(size=100))
    c0.collect(n_step=3)
    c0.collect(n_episode=3)
    env_fns = [
        lambda: MyTestEnv(size=2, sleep=0, dict_state=True),
        lambda: MyTestEnv(size=3, sleep=0, dict_state=True),
        lambda: MyTestEnv(size=4, sleep=0, dict_state=True),
        lambda: MyTestEnv(size=5, sleep=0, dict_state=True),
    ]
    envs = VectorEnv(env_fns)
    c1 = Collector(policy, envs, ReplayBuffer(size=100))
    c1.collect(n_step=10)
    c1.collect(n_episode=[2, 1, 1, 2])
    batch = c1.sample(10)
    print(batch)
    c0.buffer.update(c1.buffer)
    assert equal(c0.buffer[:len(c0.buffer)].obs.index, [
        0., 1., 2., 3., 4., 0., 1., 2., 3., 4., 0., 1., 2., 3., 4., 0., 1.,
        0., 1., 2., 0., 1., 0., 1., 2., 3., 0., 1., 2., 3., 4., 0., 1., 0.,
        1., 2., 0., 1., 0., 1., 2., 3., 0., 1., 2., 3., 4.])
    c2 = Collector(policy, envs, ReplayBuffer(size=100, stack_num=4))
    c2.collect(n_episode=[0, 0, 0, 10])
    batch = c2.sample(10)
    print(batch['obs_next']['index'])


if __name__ == '__main__':
    test_collector()
    test_collector_with_dict_state()
