import numpy as np

from tianshou.policy.base import calculate_episode_returns


def test_calculate_discounted_returns():
    assert np.all(
        calculate_episode_returns([1, 1, 1], 0.9) == np.array([0.9 ** 2 + 0.9 + 1, 0.9 + 1, 1]),
    )
    assert calculate_episode_returns([1, 2, 3], 0.5)[0] == 1 + 0.5 * (2 + 0.5 * 3)
