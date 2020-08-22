import torch
import numpy as np

from tianshou.utils import MovAvg
from tianshou.exploration import GaussianNoise, OUNoise


def test_noise():
    noise = GaussianNoise()
    size = (3, 4, 5)
    assert np.allclose(noise(size).shape, size)
    noise = OUNoise()
    noise.reset()
    assert np.allclose(noise(size).shape, size)


def test_moving_average():
    stat = MovAvg(10)
    assert np.allclose(stat.get(), 0)
    assert np.allclose(stat.mean(), 0)
    assert np.allclose(stat.std() ** 2, 0)
    stat.add(torch.tensor([1]))
    stat.add(np.array([2]))
    stat.add([3, 4])
    stat.add(5.)
    assert np.allclose(stat.get(), 3)
    assert np.allclose(stat.mean(), 3)
    assert np.allclose(stat.std() ** 2, 2)


if __name__ == '__main__':
    test_noise()
    test_moving_average()
