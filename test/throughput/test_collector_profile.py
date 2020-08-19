import gym
import numpy as np
import pytest
from gym.spaces.discrete import Discrete
from gym.utils import seeding

from tianshou.data import Batch, Collector, ReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import BasePolicy


class SimpleEnv(gym.Env):
    """A simplest example of self-defined env, used to minimize
    data collect time and profile collector."""

    def __init__(self):
        self.action_space = Discrete(200)
        self._fake_data = np.ones((10, 10, 1))
        self.seed(0)
        self.reset()

    def reset(self):
        self._index = 0
        self.done = np.random.randint(3, high=200)
        return {'observable': np.zeros((10, 10, 1)),
                'hidden': self._index}

    def step(self, action):
        if self._index == self.done:
            raise ValueError('step after done !!!')
        self._index += 1
        return {'observable': self._fake_data, 'hidden': self._index}, -1, \
            self._index == self.done, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class SimplePolicy(BasePolicy):
    """A simplest example of self-defined policy, used
    to minimize data collect time."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def learn(self, batch, **kwargs):
        return super().learn(batch, **kwargs)

    def forward(self, batch, state=None, **kwargs):
        return Batch(act=np.array([30] * len(batch)), state=None, logits=None)


@pytest.fixture(scope="module")
def data():
    np.random.seed(0)
    env = SimpleEnv()
    env.seed(0)
    env_vec = DummyVectorEnv(
        [lambda: SimpleEnv() for _ in range(100)])
    env_vec.seed(np.random.randint(1000, size=100).tolist())
    env_subproc = SubprocVectorEnv(
        [lambda: SimpleEnv() for _ in range(8)])
    env_subproc.seed(np.random.randint(1000, size=100).tolist())
    env_subproc_init = SubprocVectorEnv(
        [lambda: SimpleEnv() for _ in range(8)])
    env_subproc_init.seed(np.random.randint(1000, size=100).tolist())
    buffer = ReplayBuffer(50000)
    policy = SimplePolicy()
    collector = Collector(policy, env, ReplayBuffer(50000))
    collector_vec = Collector(policy, env_vec, ReplayBuffer(50000))
    collector_subproc = Collector(policy, env_subproc, ReplayBuffer(50000))
    return {
        "env": env,
        "env_vec": env_vec,
        "env_subproc": env_subproc,
        "env_subproc_init": env_subproc_init,
        "policy": policy,
        "buffer": buffer,
        "collector": collector,
        "collector_vec": collector_vec,
        "collector_subproc": collector_subproc,
    }


def test_init(data):
    for _ in range(5000):
        Collector(data["policy"], data["env"], data["buffer"])


def test_reset(data):
    for _ in range(5000):
        data["collector"].reset()


def test_collect_st(data):
    for _ in range(50):
        data["collector"].collect(n_step=1000)


def test_collect_ep(data):
    for _ in range(50):
        data["collector"].collect(n_episode=10)


def test_sample(data):
    for _ in range(5000):
        data["collector"].sample(256)


def test_init_vec_env(data):
    for _ in range(5000):
        Collector(data["policy"], data["env_vec"], data["buffer"])


def test_reset_vec_env(data):
    for _ in range(5000):
        data["collector_vec"].reset()


def test_collect_vec_env_st(data):
    for _ in range(50):
        data["collector_vec"].collect(n_step=1000)


def test_collect_vec_env_ep(data):
    for _ in range(50):
        data["collector_vec"].collect(n_episode=10)


def test_sample_vec_env(data):
    for _ in range(5000):
        data["collector_vec"].sample(256)


def test_init_subproc_env(data):
    for _ in range(5000):
        Collector(data["policy"], data["env_subproc_init"], data["buffer"])


def test_reset_subproc_env(data):
    for _ in range(5000):
        data["collector_subproc"].reset()


def test_collect_subproc_env_st(data):
    for _ in range(50):
        data["collector_subproc"].collect(n_step=1000)


def test_collect_subproc_env_ep(data):
    for _ in range(50):
        data["collector_subproc"].collect(n_episode=10)


def test_sample_subproc_env(data):
    for _ in range(5000):
        data["collector_subproc"].sample(256)


if __name__ == '__main__':
    pytest.main(["-s", "-k collector_profile", "--durations=0", "-v"])
