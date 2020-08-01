import numpy as np
import pytest

from tianshou.data import (ListReplayBuffer, PrioritizedReplayBuffer,
                           ReplayBuffer)


@pytest.fixture(scope="module")
def data():
    np.random.seed(0)
    obs = {'observable': np.random.rand(
        100, 100), 'hidden': np.random.randint(1000, size=200)}
    info = {'policy': "dqn", 'base': np.arange(10)}
    add_data = {'obs': obs, 'rew': 1., 'act': np.random.rand(30),
                'done': False, 'obs_next': obs, 'info': info}
    buffer = ReplayBuffer(int(1e3), stack_num=100)
    buffer2 = ReplayBuffer(int(1e4), stack_num=100)
    indexes = np.random.choice(int(1e3), size=3, replace=False)
    return{
        'add_data': add_data,
        'buffer': buffer,
        'buffer2': buffer2,
        'slice': slice(-3000, -1000, 2),
        'indexes': indexes
    }


def test_init():
    for _ in np.arange(1e5):
        _ = ReplayBuffer(1e5)
        _ = PrioritizedReplayBuffer(
            size=int(1e5), alpha=0.5,
            beta=0.5, repeat_sample=True)
        _ = ListReplayBuffer()


def test_add(data):
    buffer = data['buffer']
    for _ in np.arange(1e5):
        buffer.add(**data['add_data'])


def test_update(data):
    buffer = data['buffer']
    buffer2 = data['buffer2']
    for _ in np.arange(1e2):
        buffer2.update(buffer)


def test_getitem_slice(data):
    Slice = data['Slice']
    buffer = data['buffer']
    for _ in np.arange(1e3):
        _ = buffer[Slice]


def test_getitem_indexes(data):
    indexes = data['indexes']
    buffer = data['buffer']
    for _ in np.arange(1e2):
        _ = buffer[indexes]


def test_get(data):
    indexes = data['indexes']
    buffer = data['buffer']
    for _ in np.arange(3e2):
        buffer.get(indexes, 'obs')
        buffer.get(indexes, 'rew')
        buffer.get(indexes, 'done')
        buffer.get(indexes, 'info')


def test_sample(data):
    buffer = data['buffer']
    for _ in np.arange(1e1):
        buffer.sample(int(1e2))


if __name__ == '__main__':
    pytest.main(["-s", "-k buffer_profile", "--durations=0", "-v"])
