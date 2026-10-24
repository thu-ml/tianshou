import numpy as np
import pytest
import torch
from gymnasium import spaces

from tianshou.utils.torch_utils import create_uniform_action_dist


@pytest.mark.parametrize("shape", [(2, 2), (2, 1, 3), (3,)])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_uniform_box_preserves_action_shape(shape: tuple[int, ...], batch_size: int) -> None:
    low = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    high = low + 2
    action_space = spaces.Box(low=low, high=high)

    distribution = create_uniform_action_dist(action_space, batch_size)

    assert distribution.batch_shape == (batch_size, *shape)
    for index in range(batch_size):
        torch.testing.assert_close(distribution.low[index], torch.from_numpy(low))
        torch.testing.assert_close(distribution.high[index], torch.from_numpy(high))
        torch.testing.assert_close(distribution.mean[index], torch.from_numpy(low + 1))
    samples = distribution.sample(torch.Size((5,)))
    assert samples.shape == (5, batch_size, *shape)
    assert torch.all(samples >= distribution.low)
    assert torch.all(samples < distribution.high)
