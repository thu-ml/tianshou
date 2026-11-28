from typing import Any, cast

import gymnasium as gym
import numpy as np
import pytest
import torch

from tianshou.algorithm import BDQN
from tianshou.algorithm.modelfree.bdqn import BDQNPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Batch, PrioritizedReplayBuffer
from tianshou.utils.net.common import BranchingNet
from tianshou.utils.torch_utils import policy_within_training_step


class _BranchingQ(torch.nn.Module):
    num_branches = 2
    action_per_branch = 2

    def __init__(self) -> None:
        super().__init__()
        self.offsets = torch.nn.Parameter(torch.tensor([[2.0, 0.0], [0.0, 6.0]]))

    def forward(
        self, obs: np.ndarray, state: Any = None, info: Any = None
    ) -> tuple[torch.Tensor, Any]:
        return torch.as_tensor(obs).reshape(-1, 1, 1) + self.offsets, state


@pytest.mark.parametrize(
    ("ignore_obs_next", "targets"),
    [
        (False, [4.0, 7.0, 1.0, 9.0]),
        (True, [4.0, 1.0, 1.0, 1.0]),
        (False, [4.0]),
    ],
)
def test_bdqn_targets(ignore_obs_next: bool, targets: list[float]) -> None:
    buffer = PrioritizedReplayBuffer(4, alpha=0.6, beta=0.4, ignore_obs_next=ignore_obs_next)
    # The greedy branch values average to obs_next + 4.
    for i, obs_next in enumerate([2, 8, 10, 12][: len(targets)]):
        buffer.add(
            Batch(
                obs=[2 * i],
                obs_next=[obs_next],
                act=[0, 1],
                rew=1.0,
                terminated=i == 2,
                truncated=i == 1,
            )
        )
    model = cast(BranchingNet, _BranchingQ())
    policy = BDQNPolicy(model=model, action_space=gym.spaces.Discrete(2))
    algorithm = BDQN(policy=policy, optim=AdamOptimizerFactory(), gamma=0.5)
    batch, indices = buffer.sample(0)
    returns = algorithm._preprocess_batch(batch, buffer, indices).returns
    expected = torch.tensor(targets)[:, None, None].expand(-1, 2, 2)
    torch.testing.assert_close(returns, expected)

    # Write one priority per transition, even for a single multi-branch sample.
    with policy_within_training_step(policy):
        algorithm.update(buffer, sample_size=0)
