import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import BasePolicy
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.data import Collector, Batch, ReplayBuffer

if __name__ == '__main__':
    from env import MyTestEnv
else:  # pytest
    from test.base.env import MyTestEnv


class MyPolicy(BasePolicy):
    def __init__(self, dict_state: bool = False, need_state: bool = True):
        """
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
            return Batch(act=np.ones(len(batch.obs['index'])), state=state)
        return Batch(act=np.ones(len(batch.obs)), state=state)

    def learn(self):
        pass


class Logger:
    def __init__(self, writer):
        self.cnt = 0
        self.writer = writer

    def preprocess_fn(self, **kwargs):
        # modify info before adding into the buffer, and recorded into tfb
        # if info is not provided from env, it will be a ``Batch()``.
        if not kwargs.get('info', Batch()).is_empty():
            n = len(kwargs['obs'])
            info = kwargs['info']
            for i in range(n):
                info[i].update(rew=kwargs['rew'][i])
            self.writer.add_scalar('key', np.mean(
                info['key']), global_step=self.cnt)
            self.cnt += 1
            return Batch(info=info)
            # or: return {'info': info}
        else:
            return Batch()

    @staticmethod
    def single_preprocess_fn(**kwargs):
        # same as above, without tfb
        if not kwargs.get('info', Batch()).is_empty():
            n = len(kwargs['obs'])
            info = kwargs['info']
            for i in range(n):
                info[i].update(rew=kwargs['rew'][i])
            return Batch(info=info)
            # or: return {'info': info}
        else:
            return Batch()


def test_collector():
    writer = SummaryWriter('log/collector')
    logger = Logger(writer)
    env_fns = [lambda x=i: MyTestEnv(size=x, sleep=0) for i in [2, 3, 4, 5]]

    venv = SubprocVectorEnv(env_fns)
    dum = DummyVectorEnv(env_fns)
    policy = MyPolicy()
    env = env_fns[0]()
    c0 = Collector(policy, env, ReplayBuffer(size=100, ignore_obs_next=False),
                   logger.preprocess_fn)
    c0.collect(n_step=3)
    assert np.allclose(c0.buffer.obs[:4],
                       np.expand_dims([0, 1, 0, 1], axis=-1))
    assert np.allclose(c0.buffer[:4].obs_next,
                       np.expand_dims([1, 2, 1, 2], axis=-1))
    c0.collect(n_episode=3)
    assert np.allclose(c0.buffer.obs[:10],
                       np.expand_dims([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], axis=-1))
    assert np.allclose(c0.buffer[:10].obs_next,
                       np.expand_dims([1, 2, 1, 2, 1, 2, 1, 2, 1, 2], axis=-1))
    c0.collect(n_step=3, random=True)
    c1 = Collector(policy, venv, ReplayBuffer(size=100, ignore_obs_next=False),
                   logger.preprocess_fn)
    c1.collect(n_step=6)
    assert np.allclose(c1.buffer.obs[:11], np.expand_dims(
        [0, 1, 0, 1, 2, 0, 1, 0, 1, 2, 3], axis=-1))
    assert np.allclose(c1.buffer[:11].obs_next, np.expand_dims(
        [1, 2, 1, 2, 3, 1, 2, 1, 2, 3, 4], axis=-1))
    c1.collect(n_episode=2)
    assert np.allclose(c1.buffer.obs[11:21],
                       np.expand_dims([0, 1, 2, 3, 4, 0, 1, 0, 1, 2], axis=-1))
    assert np.allclose(c1.buffer[11:21].obs_next,
                       np.expand_dims([1, 2, 3, 4, 5, 1, 2, 1, 2, 3], axis=-1))
    c1.collect(n_episode=3, random=True)
    c2 = Collector(policy, dum, ReplayBuffer(size=100, ignore_obs_next=False),
                   logger.preprocess_fn)
    c2.collect(n_episode=[1, 2, 2, 2])
    assert np.allclose(c2.buffer.obs_next[:26], np.expand_dims([
        1, 2, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 5,
        1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 5], axis=-1))
    c2.reset_env()
    c2.collect(n_episode=[2, 2, 2, 2])
    assert np.allclose(c2.buffer.obs_next[26:54], np.expand_dims([
        1, 2, 1, 2, 3, 1, 2, 1, 2, 3, 4, 1, 2, 3, 4, 5,
        1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 5], axis=-1))
    c2.collect(n_episode=[1, 1, 1, 1], random=True)


def test_collector_with_async():
    env_lens = [2, 3, 4, 5]
    writer = SummaryWriter('log/async_collector')
    logger = Logger(writer)
    env_fns = [lambda x=i: MyTestEnv(size=x, sleep=0.1, random_sleep=True)
               for i in env_lens]

    venv = SubprocVectorEnv(env_fns, wait_num=len(env_fns) - 1)
    policy = MyPolicy()
    c1 = Collector(policy, venv,
                   ReplayBuffer(size=1000, ignore_obs_next=False),
                   logger.preprocess_fn)
    c1.collect(n_episode=10)
    # check if the data in the buffer is chronological
    # i.e. data in the buffer are full episodes, and each episode is
    # returned by the same environment
    env_id = c1.buffer.info['env_id']
    size = len(c1.buffer)
    obs = c1.buffer.obs[:size]
    done = c1.buffer.done[:size]
    obs_ground_truth = []
    i = 0
    while i < size:
        # i is the start of an episode
        if done[i]:
            # this episode has one transition
            assert env_lens[env_id[i]] == 1
            i += 1
            continue
        j = i
        while True:
            j += 1
            # in one episode, the environment id is the same
            assert env_id[j] == env_id[i]
            if done[j]:
                break
        j = j + 1  # j is the start of the next episode
        assert j - i == env_lens[env_id[i]]
        obs_ground_truth += list(range(j - i))
        i = j
    obs_ground_truth = np.expand_dims(
        np.array(obs_ground_truth), axis=-1)
    assert np.allclose(obs, obs_ground_truth)


def test_collector_with_dict_state():
    env = MyTestEnv(size=5, sleep=0, dict_state=True)
    policy = MyPolicy(dict_state=True)
    c0 = Collector(policy, env, ReplayBuffer(size=100),
                   Logger.single_preprocess_fn)
    c0.collect(n_step=3)
    c0.collect(n_episode=2)
    env_fns = [lambda x=i: MyTestEnv(size=x, sleep=0, dict_state=True)
               for i in [2, 3, 4, 5]]
    envs = DummyVectorEnv(env_fns)
    envs.seed(666)
    obs = envs.reset()
    assert not np.isclose(obs[0]['rand'], obs[1]['rand'])
    c1 = Collector(policy, envs, ReplayBuffer(size=100),
                   Logger.single_preprocess_fn)
    c1.seed(0)
    c1.collect(n_step=10)
    c1.collect(n_episode=[2, 1, 1, 2])
    batch, _ = c1.buffer.sample(10)
    print(batch)
    c0.buffer.update(c1.buffer)
    assert np.allclose(c0.buffer[:len(c0.buffer)].obs.index, np.expand_dims([
        0., 1., 2., 3., 4., 0., 1., 2., 3., 4., 0., 1., 2., 3., 4., 0., 1.,
        0., 1., 2., 0., 1., 0., 1., 2., 3., 0., 1., 2., 3., 4., 0., 1., 0.,
        1., 2., 0., 1., 0., 1., 2., 3., 0., 1., 2., 3., 4.], axis=-1))
    c2 = Collector(policy, envs, ReplayBuffer(size=100, stack_num=4),
                   Logger.single_preprocess_fn)
    c2.collect(n_episode=[0, 0, 0, 10])
    batch, _ = c2.buffer.sample(10)


def test_collector_with_ma():
    def reward_metric(x):
        return x.sum()
    env = MyTestEnv(size=5, sleep=0, ma_rew=4)
    policy = MyPolicy()
    c0 = Collector(policy, env, ReplayBuffer(size=100),
                   Logger.single_preprocess_fn, reward_metric=reward_metric)
    # n_step=3 will collect a full episode
    r = c0.collect(n_step=3)['rew']
    assert np.asanyarray(r).size == 1 and r == 4.
    r = c0.collect(n_episode=2)['rew']
    assert np.asanyarray(r).size == 1 and r == 4.
    env_fns = [lambda x=i: MyTestEnv(size=x, sleep=0, ma_rew=4)
               for i in [2, 3, 4, 5]]
    envs = DummyVectorEnv(env_fns)
    c1 = Collector(policy, envs, ReplayBuffer(size=100),
                   Logger.single_preprocess_fn, reward_metric=reward_metric)
    r = c1.collect(n_step=10)['rew']
    assert np.asanyarray(r).size == 1 and r == 4.
    r = c1.collect(n_episode=[2, 1, 1, 2])['rew']
    assert np.asanyarray(r).size == 1 and r == 4.
    batch, _ = c1.buffer.sample(10)
    print(batch)
    c0.buffer.update(c1.buffer)
    obs = np.array(np.expand_dims([
        0., 1., 2., 3., 4., 0., 1., 2., 3., 4., 0., 1., 2., 3., 4., 0., 1.,
        0., 1., 2., 0., 1., 0., 1., 2., 3., 0., 1., 2., 3., 4., 0., 1., 0.,
        1., 2., 0., 1., 0., 1., 2., 3., 0., 1., 2., 3., 4.], axis=-1))
    assert np.allclose(c0.buffer[:len(c0.buffer)].obs, obs)
    rew = [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1,
           0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0,
           0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    assert np.allclose(c0.buffer[:len(c0.buffer)].rew,
                       [[x] * 4 for x in rew])
    c2 = Collector(policy, envs, ReplayBuffer(size=100, stack_num=4),
                   Logger.single_preprocess_fn, reward_metric=reward_metric)
    r = c2.collect(n_episode=[0, 0, 0, 10])['rew']
    assert np.asanyarray(r).size == 1 and r == 4.
    batch, _ = c2.buffer.sample(10)


if __name__ == '__main__':
    test_collector()
    test_collector_with_dict_state()
    test_collector_with_ma()
    test_collector_with_async()
