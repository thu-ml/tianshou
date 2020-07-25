import gym
import numpy as np
import pytest
from gym.spaces.discrete import Discrete
from gym.utils import seeding

from tianshou.data import Batch, Collector, ReplayBuffer
from tianshou.env import VectorEnv
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
        return {'observable': self._fake_data, 'hidden': self._index},
        -1, self._index == self.done, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class SimplePolicy(BasePolicy):
    """A simplest example of self-defined policy, used
    to minimize data collect time."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._fake_return = Batch(act=np.array([30]), state=None)

    def learn(self, batch, **kwargs):
        return super().learn(batch, **kwargs)

    def forward(self, batch, state=None, **kwargs):
        return Batch(act=np.array([30]*len(batch)), state=None, logits=None)


@pytest.fixture(scope="module")
def data():
    np.random.seed(0)
    env = SimpleEnv()
    env.seed(0)
    envs = VectorEnv(
        [lambda: SimpleEnv() for _ in range(100)])
    envs.seed(np.random.randint(1000, size=100).tolist())
    buffer = ReplayBuffer(50000)
    policy = SimplePolicy()
    collector = Collector(policy, env, ReplayBuffer(50000))
    collector_multi = Collector(policy, envs, ReplayBuffer(50000))
    return{
        "env": env,
        "policy": policy,
        "envs": envs,
        "buffer": buffer,
        "collector": collector,
        "collector_multi": collector_multi
    }


def test_init(data):
    for _ in range(1000):
        c = Collector(data["policy"], data["env"], data["buffer"])
        c.close()


def test_reset(data):
    for _ in range(1000):
        data["collector"].reset()


def test_collect_st(data):
    for _ in range(10):
        data["collector"].collect(n_step=1000)


def test_collect_ep(data):
    for _ in range(10):
        data["collector"].collect(n_episode=10)


def test_sample(data):
    for _ in range(1000):
        data["collector"].sample(256)


def test_init_multi_env(data):
    for _ in range(1000):
        c = Collector(data["policy"], data["envs"], data["buffer"])
        c.close()


def test_reset_multi_env(data):
    for _ in range(1000):
        data["collector_multi"].reset()


def test_collect_multi_env_st(data):
    for _ in range(10):
        data["collector_multi"].collect(n_step=1000)


def test_collect_multi_env_ep(data):
    for _ in range(10):
        data["collector_multi"].collect(n_episode=10)


def test_sample_multi_env(data):
    for _ in range(1000):
        data["collector_multi"].sample(256)


if __name__ == '__main__':
    pytest.main(["-s", "-k collector_profile", "--durations=0", "-v"])
