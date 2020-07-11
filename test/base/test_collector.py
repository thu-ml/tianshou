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
            return Batch(act=np.ones(len(batch.obs['index'])))
        return Batch(act=np.ones(len(batch.obs)))

    def learn(self):
        pass


def preprocess_fn(**kwargs):
    # modify info before adding into the buffer
    # if info is not provided from env, it will be a `Batch()`.
    if not kwargs.get('info', Batch()).is_empty():
        n = len(kwargs['obs'])
        info = kwargs['info']
        for i in range(n):
            info[i].update(rew=kwargs['rew'][i])
        return {'info': info}
        # or: return Batch(info=info)
    else:
        return Batch()


class Logger(object):
    def __init__(self, writer):
        self.cnt = 0
        self.writer = writer

    def log(self, info):
        self.writer.add_scalar(
            'key', np.mean(info['key']), global_step=self.cnt)
        self.cnt += 1


def test_collector():
    writer = SummaryWriter('log/collector')
    logger = Logger(writer)
    env_fns = [lambda x=i: MyTestEnv(size=x, sleep=0) for i in [2, 3, 4, 5]]

    venv = SubprocVectorEnv(env_fns)
    dum = VectorEnv(env_fns)
    policy = MyPolicy()
    env = env_fns[0]()
    c0 = Collector(policy, env, ReplayBuffer(size=100, ignore_obs_next=False),
                   preprocess_fn)
    c0.collect(n_step=3, log_fn=logger.log)
    assert np.allclose(c0.buffer.obs[:3], [0, 1, 0])
    assert np.allclose(c0.buffer[:3].obs_next, [1, 2, 1])
    c0.collect(n_episode=3, log_fn=logger.log)
    assert np.allclose(c0.buffer.obs[:8], [0, 1, 0, 1, 0, 1, 0, 1])
    assert np.allclose(c0.buffer[:8].obs_next, [1, 2, 1, 2, 1, 2, 1, 2])
    c0.collect(n_step=3, random=True)
    c1 = Collector(policy, venv, ReplayBuffer(size=100, ignore_obs_next=False),
                   preprocess_fn)
    c1.collect(n_step=6)
    assert np.allclose(c1.buffer.obs[:11], [0, 1, 0, 1, 2, 0, 1, 0, 1, 2, 3])
    assert np.allclose(c1.buffer[:11].obs_next,
                       [1, 2, 1, 2, 3, 1, 2, 1, 2, 3, 4])
    c1.collect(n_episode=2)
    assert np.allclose(c1.buffer.obs[11:21], [0, 1, 2, 3, 4, 0, 1, 0, 1, 2])
    assert np.allclose(c1.buffer[11:21].obs_next,
                       [1, 2, 3, 4, 5, 1, 2, 1, 2, 3])
    c1.collect(n_episode=3, random=True)
    c2 = Collector(policy, dum, ReplayBuffer(size=100, ignore_obs_next=False),
                   preprocess_fn)
    c2.collect(n_episode=[1, 2, 2, 2])
    assert np.allclose(c2.buffer.obs_next[:26], [
        1, 2, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 5,
        1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 5])
    c2.reset_env()
    c2.collect(n_episode=[2, 2, 2, 2])
    assert np.allclose(c2.buffer.obs_next[26:54], [
        1, 2, 1, 2, 3, 1, 2, 1, 2, 3, 4, 1, 2, 3, 4, 5,
        1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 5])
    c2.collect(n_episode=[1, 1, 1, 1], random=True)


def test_collector_with_dict_state():
    env = MyTestEnv(size=5, sleep=0, dict_state=True)
    policy = MyPolicy(dict_state=True)
    c0 = Collector(policy, env, ReplayBuffer(size=100), preprocess_fn)
    c0.collect(n_step=3)
    c0.collect(n_episode=3)
    env_fns = [lambda x=i: MyTestEnv(size=x, sleep=0, dict_state=True)
               for i in [2, 3, 4, 5]]
    envs = VectorEnv(env_fns)
    c1 = Collector(policy, envs, ReplayBuffer(size=100), preprocess_fn)
    c1.collect(n_step=10)
    c1.collect(n_episode=[2, 1, 1, 2])
    batch = c1.sample(10)
    print(batch)
    c0.buffer.update(c1.buffer)
    assert np.allclose(c0.buffer[:len(c0.buffer)].obs.index, [
        0., 1., 2., 3., 4., 0., 1., 2., 3., 4., 0., 1., 2., 3., 4., 0., 1.,
        0., 1., 2., 0., 1., 0., 1., 2., 3., 0., 1., 2., 3., 4., 0., 1., 0.,
        1., 2., 0., 1., 0., 1., 2., 3., 0., 1., 2., 3., 4.])
    c2 = Collector(policy, envs, ReplayBuffer(size=100, stack_num=4),
                   preprocess_fn)
    c2.collect(n_episode=[0, 0, 0, 10])
    batch = c2.sample(10)
    print(batch['obs_next']['index'])


if __name__ == '__main__':
    test_collector()
    test_collector_with_dict_state()
