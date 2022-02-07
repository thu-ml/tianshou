import numpy as np
import pytest
import tqdm
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import (
    AsyncCollector,
    Batch,
    CachedReplayBuffer,
    Collector,
    PrioritizedReplayBuffer,
    ReplayBuffer,
    VectorReplayBuffer,
)
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import BasePolicy

if __name__ == '__main__':
    from env import MyTestEnv, NXEnv
else:  # pytest
    from test.base.env import MyTestEnv, NXEnv


class MyPolicy(BasePolicy):

    def __init__(self, dict_state=False, need_state=True):
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
        # if obs && env_id exist -> reset
        # if obs_next/rew/done/info/env_id exist -> normal step
        if 'rew' in kwargs:
            info = kwargs['info']
            info.rew = kwargs['rew']
            if 'key' in info.keys():
                self.writer.add_scalar('key', np.mean(info.key), global_step=self.cnt)
            self.cnt += 1
            return Batch(info=info)
        else:
            return Batch()

    @staticmethod
    def single_preprocess_fn(**kwargs):
        # same as above, without tfb
        if 'rew' in kwargs:
            info = kwargs['info']
            info.rew = kwargs['rew']
            return Batch(info=info)
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
    c0 = Collector(policy, env, ReplayBuffer(size=100), logger.preprocess_fn)
    c0.collect(n_step=3)
    assert len(c0.buffer) == 3
    assert np.allclose(c0.buffer.obs[:4, 0], [0, 1, 0, 0])
    assert np.allclose(c0.buffer[:].obs_next[..., 0], [1, 2, 1])
    c0.collect(n_episode=3)
    assert len(c0.buffer) == 8
    assert np.allclose(c0.buffer.obs[:10, 0], [0, 1, 0, 1, 0, 1, 0, 1, 0, 0])
    assert np.allclose(c0.buffer[:].obs_next[..., 0], [1, 2, 1, 2, 1, 2, 1, 2])
    c0.collect(n_step=3, random=True)
    c1 = Collector(
        policy, venv, VectorReplayBuffer(total_size=100, buffer_num=4),
        logger.preprocess_fn
    )
    c1.collect(n_step=8)
    obs = np.zeros(100)
    obs[[0, 1, 25, 26, 50, 51, 75, 76]] = [0, 1, 0, 1, 0, 1, 0, 1]

    assert np.allclose(c1.buffer.obs[:, 0], obs)
    assert np.allclose(c1.buffer[:].obs_next[..., 0], [1, 2, 1, 2, 1, 2, 1, 2])
    c1.collect(n_episode=4)
    assert len(c1.buffer) == 16
    obs[[2, 3, 27, 52, 53, 77, 78, 79]] = [0, 1, 2, 2, 3, 2, 3, 4]
    assert np.allclose(c1.buffer.obs[:, 0], obs)
    assert np.allclose(
        c1.buffer[:].obs_next[..., 0],
        [1, 2, 1, 2, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 5]
    )
    c1.collect(n_episode=4, random=True)
    c2 = Collector(
        policy, dum, VectorReplayBuffer(total_size=100, buffer_num=4),
        logger.preprocess_fn
    )
    c2.collect(n_episode=7)
    obs1 = obs.copy()
    obs1[[4, 5, 28, 29, 30]] = [0, 1, 0, 1, 2]
    obs2 = obs.copy()
    obs2[[28, 29, 30, 54, 55, 56, 57]] = [0, 1, 2, 0, 1, 2, 3]
    c2obs = c2.buffer.obs[:, 0]
    assert np.all(c2obs == obs1) or np.all(c2obs == obs2)
    c2.reset_env()
    c2.reset_buffer()
    assert c2.collect(n_episode=8)['n/ep'] == 8
    obs[[4, 5, 28, 29, 30, 54, 55, 56, 57]] = [0, 1, 0, 1, 2, 0, 1, 2, 3]
    assert np.all(c2.buffer.obs[:, 0] == obs)
    c2.collect(n_episode=4, random=True)

    # test corner case
    with pytest.raises(TypeError):
        Collector(policy, dum, ReplayBuffer(10))
    with pytest.raises(TypeError):
        Collector(policy, dum, PrioritizedReplayBuffer(10, 0.5, 0.5))
    with pytest.raises(TypeError):
        c2.collect()

    # test NXEnv
    for obs_type in ["array", "object"]:
        envs = SubprocVectorEnv(
            [lambda i=x: NXEnv(i, obs_type) for x in [5, 10, 15, 20]]
        )
        c3 = Collector(policy, envs, VectorReplayBuffer(total_size=100, buffer_num=4))
        c3.collect(n_step=6)
        assert c3.buffer.obs.dtype == object


def test_collector_with_async():
    env_lens = [2, 3, 4, 5]
    writer = SummaryWriter('log/async_collector')
    logger = Logger(writer)
    env_fns = [
        lambda x=i: MyTestEnv(size=x, sleep=0.001, random_sleep=True) for i in env_lens
    ]

    venv = SubprocVectorEnv(env_fns, wait_num=len(env_fns) - 1)
    policy = MyPolicy()
    bufsize = 60
    c1 = AsyncCollector(
        policy, venv, VectorReplayBuffer(total_size=bufsize * 4, buffer_num=4),
        logger.preprocess_fn
    )
    ptr = [0, 0, 0, 0]
    for n_episode in tqdm.trange(1, 30, desc="test async n_episode"):
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
    for n_step in tqdm.trange(1, 15, desc="test async n_step"):
        result = c1.collect(n_step=n_step)
        assert result["n/st"] >= n_step
        for i in range(4):
            env_len = i + 2
            seq = np.arange(env_len)
            buf = c1.buffer.buffers[i]
            assert np.all(buf.info.env_id == i)
            assert np.all(buf.obs.reshape(-1, env_len) == seq)
            assert np.all(buf.obs_next.reshape(-1, env_len) == seq + 1)
    with pytest.raises(TypeError):
        c1.collect()


def test_collector_with_dict_state():
    env = MyTestEnv(size=5, sleep=0, dict_state=True)
    policy = MyPolicy(dict_state=True)
    c0 = Collector(policy, env, ReplayBuffer(size=100), Logger.single_preprocess_fn)
    c0.collect(n_step=3)
    c0.collect(n_episode=2)
    assert len(c0.buffer) == 10
    env_fns = [
        lambda x=i: MyTestEnv(size=x, sleep=0, dict_state=True) for i in [2, 3, 4, 5]
    ]
    envs = DummyVectorEnv(env_fns)
    envs.seed(666)
    obs = envs.reset()
    assert not np.isclose(obs[0]['rand'], obs[1]['rand'])
    c1 = Collector(
        policy, envs, VectorReplayBuffer(total_size=100, buffer_num=4),
        Logger.single_preprocess_fn
    )
    c1.collect(n_step=12)
    result = c1.collect(n_episode=8)
    assert result['n/ep'] == 8
    lens = np.bincount(result['lens'])
    assert result['n/st'] == 21 and np.all(lens == [0, 0, 2, 2, 2, 2]) or \
        result['n/st'] == 20 and np.all(lens == [0, 0, 3, 1, 2, 2])
    batch, _ = c1.buffer.sample(10)
    c0.buffer.update(c1.buffer)
    assert len(c0.buffer) in [42, 43]
    if len(c0.buffer) == 42:
        assert np.all(
            c0.buffer[:].obs.index[..., 0] == [
                0,
                1,
                2,
                3,
                4,
                0,
                1,
                2,
                3,
                4,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                2,
                0,
                1,
                2,
                0,
                1,
                2,
                3,
                0,
                1,
                2,
                3,
                0,
                1,
                2,
                3,
                4,
                0,
                1,
                2,
                3,
                4,
            ]
        ), c0.buffer[:].obs.index[..., 0]
    else:
        assert np.all(
            c0.buffer[:].obs.index[..., 0] == [
                0,
                1,
                2,
                3,
                4,
                0,
                1,
                2,
                3,
                4,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                2,
                0,
                1,
                2,
                0,
                1,
                2,
                0,
                1,
                2,
                3,
                0,
                1,
                2,
                3,
                0,
                1,
                2,
                3,
                4,
                0,
                1,
                2,
                3,
                4,
            ]
        ), c0.buffer[:].obs.index[..., 0]
    c2 = Collector(
        policy, envs, VectorReplayBuffer(total_size=100, buffer_num=4, stack_num=4),
        Logger.single_preprocess_fn
    )
    c2.collect(n_episode=10)
    batch, _ = c2.buffer.sample(10)


def test_collector_with_ma():
    env = MyTestEnv(size=5, sleep=0, ma_rew=4)
    policy = MyPolicy()
    c0 = Collector(policy, env, ReplayBuffer(size=100), Logger.single_preprocess_fn)
    # n_step=3 will collect a full episode
    rew = c0.collect(n_step=3)['rews']
    assert len(rew) == 0
    rew = c0.collect(n_episode=2)['rews']
    assert rew.shape == (2, 4) and np.all(rew == 1)
    env_fns = [lambda x=i: MyTestEnv(size=x, sleep=0, ma_rew=4) for i in [2, 3, 4, 5]]
    envs = DummyVectorEnv(env_fns)
    c1 = Collector(
        policy, envs, VectorReplayBuffer(total_size=100, buffer_num=4),
        Logger.single_preprocess_fn
    )
    rew = c1.collect(n_step=12)['rews']
    assert rew.shape == (2, 4) and np.all(rew == 1), rew
    rew = c1.collect(n_episode=8)['rews']
    assert rew.shape == (8, 4) and np.all(rew == 1)
    batch, _ = c1.buffer.sample(10)
    print(batch)
    c0.buffer.update(c1.buffer)
    assert len(c0.buffer) in [42, 43]
    if len(c0.buffer) == 42:
        rew = [
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            0,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
        ]
    else:
        rew = [
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            0,
            1,
            0,
            0,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
        ]
    assert np.all(c0.buffer[:].rew == [[x] * 4 for x in rew])
    assert np.all(c0.buffer[:].done == rew)
    c2 = Collector(
        policy, envs, VectorReplayBuffer(total_size=100, buffer_num=4, stack_num=4),
        Logger.single_preprocess_fn
    )
    rew = c2.collect(n_episode=10)['rews']
    assert rew.shape == (10, 4) and np.all(rew == 1)
    batch, _ = c2.buffer.sample(10)


def test_collector_with_atari_setting():
    reference_obs = np.zeros([6, 4, 84, 84])
    for i in range(6):
        reference_obs[i, 3, np.arange(84), np.arange(84)] = i
        reference_obs[i, 2, np.arange(84)] = i
        reference_obs[i, 1, :, np.arange(84)] = i
        reference_obs[i, 0] = i

    # atari single buffer
    env = MyTestEnv(size=5, sleep=0, array_state=True)
    policy = MyPolicy()
    c0 = Collector(policy, env, ReplayBuffer(size=100))
    c0.collect(n_step=6)
    c0.collect(n_episode=2)
    assert c0.buffer.obs.shape == (100, 4, 84, 84)
    assert c0.buffer.obs_next.shape == (100, 4, 84, 84)
    assert len(c0.buffer) == 15
    obs = np.zeros_like(c0.buffer.obs)
    obs[np.arange(15)] = reference_obs[np.arange(15) % 5]
    assert np.all(obs == c0.buffer.obs)

    c1 = Collector(policy, env, ReplayBuffer(size=100, ignore_obs_next=True))
    c1.collect(n_episode=3)
    assert np.allclose(c0.buffer.obs, c1.buffer.obs)
    with pytest.raises(AttributeError):
        c1.buffer.obs_next
    assert np.all(reference_obs[[1, 2, 3, 4, 4] * 3] == c1.buffer[:].obs_next)

    c2 = Collector(
        policy, env,
        ReplayBuffer(size=100, ignore_obs_next=True, save_only_last_obs=True)
    )
    c2.collect(n_step=8)
    assert c2.buffer.obs.shape == (100, 84, 84)
    obs = np.zeros_like(c2.buffer.obs)
    obs[np.arange(8)] = reference_obs[[0, 1, 2, 3, 4, 0, 1, 2], -1]
    assert np.all(c2.buffer.obs == obs)
    assert np.allclose(
        c2.buffer[:].obs_next, reference_obs[[1, 2, 3, 4, 4, 1, 2, 2], -1]
    )

    # atari multi buffer
    env_fns = [
        lambda x=i: MyTestEnv(size=x, sleep=0, array_state=True) for i in [2, 3, 4, 5]
    ]
    envs = DummyVectorEnv(env_fns)
    c3 = Collector(policy, envs, VectorReplayBuffer(total_size=100, buffer_num=4))
    c3.collect(n_step=12)
    result = c3.collect(n_episode=9)
    assert result["n/ep"] == 9 and result["n/st"] == 23
    assert c3.buffer.obs.shape == (100, 4, 84, 84)
    obs = np.zeros_like(c3.buffer.obs)
    obs[np.arange(8)] = reference_obs[[0, 1, 0, 1, 0, 1, 0, 1]]
    obs[np.arange(25, 34)] = reference_obs[[0, 1, 2, 0, 1, 2, 0, 1, 2]]
    obs[np.arange(50, 58)] = reference_obs[[0, 1, 2, 3, 0, 1, 2, 3]]
    obs[np.arange(75, 85)] = reference_obs[[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]]
    assert np.all(obs == c3.buffer.obs)
    obs_next = np.zeros_like(c3.buffer.obs_next)
    obs_next[np.arange(8)] = reference_obs[[1, 2, 1, 2, 1, 2, 1, 2]]
    obs_next[np.arange(25, 34)] = reference_obs[[1, 2, 3, 1, 2, 3, 1, 2, 3]]
    obs_next[np.arange(50, 58)] = reference_obs[[1, 2, 3, 4, 1, 2, 3, 4]]
    obs_next[np.arange(75, 85)] = reference_obs[[1, 2, 3, 4, 5, 1, 2, 3, 4, 5]]
    assert np.all(obs_next == c3.buffer.obs_next)
    c4 = Collector(
        policy, envs,
        VectorReplayBuffer(
            total_size=100,
            buffer_num=4,
            stack_num=4,
            ignore_obs_next=True,
            save_only_last_obs=True
        )
    )
    c4.collect(n_step=12)
    result = c4.collect(n_episode=9)
    assert result["n/ep"] == 9 and result["n/st"] == 23
    assert c4.buffer.obs.shape == (100, 84, 84)
    obs = np.zeros_like(c4.buffer.obs)
    slice_obs = reference_obs[:, -1]
    obs[np.arange(8)] = slice_obs[[0, 1, 0, 1, 0, 1, 0, 1]]
    obs[np.arange(25, 34)] = slice_obs[[0, 1, 2, 0, 1, 2, 0, 1, 2]]
    obs[np.arange(50, 58)] = slice_obs[[0, 1, 2, 3, 0, 1, 2, 3]]
    obs[np.arange(75, 85)] = slice_obs[[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]]
    assert np.all(c4.buffer.obs == obs)
    obs_next = np.zeros([len(c4.buffer), 4, 84, 84])
    ref_index = np.array(
        [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            1,
            2,
            2,
            1,
            2,
            2,
            1,
            2,
            3,
            3,
            1,
            2,
            3,
            3,
            1,
            2,
            3,
            4,
            4,
            1,
            2,
            3,
            4,
            4,
        ]
    )
    obs_next[:, -1] = slice_obs[ref_index]
    ref_index -= 1
    ref_index[ref_index < 0] = 0
    obs_next[:, -2] = slice_obs[ref_index]
    ref_index -= 1
    ref_index[ref_index < 0] = 0
    obs_next[:, -3] = slice_obs[ref_index]
    ref_index -= 1
    ref_index[ref_index < 0] = 0
    obs_next[:, -4] = slice_obs[ref_index]
    assert np.all(obs_next == c4.buffer[:].obs_next)

    buf = ReplayBuffer(100, stack_num=4, ignore_obs_next=True, save_only_last_obs=True)
    c5 = Collector(policy, envs, CachedReplayBuffer(buf, 4, 10))
    result_ = c5.collect(n_step=12)
    assert len(buf) == 5 and len(c5.buffer) == 12
    result = c5.collect(n_episode=9)
    assert result["n/ep"] == 9 and result["n/st"] == 23
    assert len(buf) == 35
    assert np.all(
        buf.obs[:len(buf)] == slice_obs[[
            0, 1, 0, 1, 2, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 0, 1, 2, 0, 1, 0, 1,
            2, 3, 0, 1, 2, 0, 1, 2, 3, 4
        ]]
    )
    assert np.all(
        buf[:].obs_next[:, -1] == slice_obs[[
            1, 1, 1, 2, 2, 1, 1, 1, 2, 3, 3, 1, 2, 3, 4, 4, 1, 1, 1, 2, 2, 1, 1, 1, 2,
            3, 3, 1, 2, 2, 1, 2, 3, 4, 4
        ]]
    )
    assert len(buf) == len(c5.buffer)

    # test buffer=None
    c6 = Collector(policy, envs)
    result1 = c6.collect(n_step=12)
    for key in ["n/ep", "n/st", "rews", "lens"]:
        assert np.allclose(result1[key], result_[key])
    result2 = c6.collect(n_episode=9)
    for key in ["n/ep", "n/st", "rews", "lens"]:
        assert np.allclose(result2[key], result[key])


if __name__ == '__main__':
    test_collector()
    test_collector_with_dict_state()
    test_collector_with_ma()
    test_collector_with_atari_setting()
    test_collector_with_async()
