import numpy as np
from tianshou.data import Batch, ReplayBuffer, PrioritizedReplayBuffer, ListReplayBuffer
import pytest
@pytest.fixture(scope="module")
def data():
    pass


def test_init():
    for _ in range(1e5):
        _ = ReplayBuffer(1e5)
        

def test_add():
    pass

def test_update():
    pass

def test_getitem():
    pass

def test_get():
    pass

def test_sample():
    pass
    


if __name__ == '__main__':
    pytest.main(["-s", "-k buffer_profile", "--durations=0", "-v"])





