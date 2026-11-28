from typing import Any, cast

import gymnasium as gym
import numpy as np
import pytest
import torch

from tianshou.algorithm import BDQN
from tianshou.algorithm.modelfree.bdqn import BDQNPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Batch, ReplayBuffer
from tianshou.utils.net.common import BranchingNet

NUM_BRANCHES = 2
ACTIONS_PER_BRANCH = 3
Q_VALUE = 10.0


class _ConstantBranchingQ(torch.nn.Module):
    """Branching Q-network whose Q(s, branch, a) is the same constant everywhere."""

    num_branches = NUM_BRANCHES
    action_per_branch = ACTIONS_PER_BRANCH

    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(
        self, obs: np.ndarray, state: Any = None, info: Any = None
    ) -> tuple[torch.Tensor, Any]:
        q = torch.full((len(obs), NUM_BRANCHES, ACTIONS_PER_BRANCH), Q_VALUE)
        return q + 0 * self.dummy, state


@pytest.mark.parametrize("gamma", [0.5, 0.9, 0.99])
def test_bdqn_targets_use_gamma_and_bootstrap_unless_terminated(gamma: float) -> None:
    buffer = ReplayBuffer(8)
    # continuing, truncated (time limit), terminated, continuing but unfinished (last in buffer)
    endings = [(False, False), (False, True), (True, False), (False, False)]
    for terminated, truncated in endings:
        buffer.add(
            Batch(
                obs=np.zeros(1),
                act=np.zeros(NUM_BRANCHES, dtype=int),
                rew=1.0,
                terminated=terminated,
                truncated=truncated,
                obs_next=np.zeros(1),
                info={},
            )
        )
    algorithm = BDQN(
        policy=BDQNPolicy(
            model=cast(BranchingNet, _ConstantBranchingQ()),
            action_space=gym.spaces.Discrete(ACTIONS_PER_BRANCH),
        ),
        optim=AdamOptimizerFactory(),
        gamma=gamma,
    )
    batch, indices = buffer.sample(0)

    returns = algorithm._preprocess_batch(batch, buffer, indices).returns.numpy()

    expected = 1.0 + gamma * Q_VALUE * (1.0 - buffer.terminated[indices])
    assert returns.shape == (len(indices), NUM_BRANCHES, ACTIONS_PER_BRANCH)
    np.testing.assert_allclose(returns, np.broadcast_to(expected[:, None, None], returns.shape))
