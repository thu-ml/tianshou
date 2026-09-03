from typing import cast

import numpy as np
import torch
from gymnasium.spaces import Discrete

from tianshou.algorithm.modelfree.c51 import C51, C51Policy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Batch, ReplayBuffer
from tianshou.data.types import RolloutBatchProtocol


class FixedDistributionModel(torch.nn.Module):
    def __init__(self, probabilities: torch.Tensor) -> None:
        super().__init__()
        self.probabilities = torch.nn.Parameter(probabilities)
        self.batch_sizes: list[int] = []

    def forward(
        self,
        obs: torch.Tensor | np.ndarray,
        state: object | None = None,
        info: object | None = None,
    ) -> tuple[torch.Tensor, None]:
        obs_tensor = torch.as_tensor(obs)
        if torch.isnan(obs_tensor).any():
            raise AssertionError("terminal observations must not be evaluated")
        self.batch_sizes.append(len(obs_tensor))
        return self.probabilities.expand(len(obs_tensor), -1, -1), None


def make_algorithm(
    probabilities: torch.Tensor,
    n_step: int = 1,
) -> tuple[C51, FixedDistributionModel]:
    model = FixedDistributionModel(probabilities)
    policy = C51Policy(
        model=model,
        action_space=Discrete(2),
        num_atoms=5,
        v_min=-2.0,
        v_max=2.0,
    )
    algorithm = C51(
        policy=policy,
        optim=AdamOptimizerFactory(lr=1e-3),
        gamma=0.9,
        n_step_return_horizon=n_step,
    )
    return algorithm, model


def test_terminal_target_projects_reward_without_evaluating_obs_next() -> None:
    probabilities = torch.tensor(
        [
            [[0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0]],
        ],
    )
    algorithm, model = make_algorithm(probabilities)
    batch = Batch(
        obs_next=np.array([[np.nan]], dtype=np.float32),
        info=np.array([None], dtype=object),
        terminated=np.array([True]),
        truncated=np.array([False]),
        returns=torch.full((1, 5), 0.5),
        act=np.array([0]),
    )

    target_dist = algorithm._target_dist(batch)

    torch.testing.assert_close(
        target_dist,
        torch.tensor([[0.0, 0.0, 0.5, 0.5, 0.0]]),
    )
    assert model.batch_sizes == []


def test_mixed_terminal_and_bootstrapped_targets_preserve_row_alignment() -> None:
    probabilities = torch.tensor(
        [
            [[0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0]],
        ],
    )
    algorithm, model = make_algorithm(probabilities)
    batch = Batch(
        obs_next=np.array([[np.nan], [1.0]], dtype=np.float32),
        info=np.array([None, None], dtype=object),
        terminated=np.array([True, False]),
        truncated=np.array([False, False]),
        returns=torch.tensor(
            [
                [0.5, 0.5, 0.5, 0.5, 0.5],
                [0.0, 0.5, 1.0, 1.5, 2.0],
            ],
        ),
        act=np.array([0, 0]),
    )

    target_dist = algorithm._target_dist(batch)
    support = algorithm.policy.support.detach()
    next_dist = probabilities[0, 0]
    nonterminal_support = batch.returns[1:2].clamp(
        algorithm.policy.v_min,
        algorithm.policy.v_max,
    )
    expected_nonterminal = (
        (
            1
            - (nonterminal_support.unsqueeze(1) - support.view(1, -1, 1)).abs() / algorithm.delta_z
        ).clamp(0, 1)
        * next_dist.unsqueeze(0)
    ).sum(-1)

    torch.testing.assert_close(target_dist[0], torch.tensor([0.0, 0.0, 0.5, 0.5, 0.0]))
    torch.testing.assert_close(target_dist[1], expected_nonterminal[0])
    assert model.batch_sizes == [1]


def test_n_step_terminal_mask_uses_endpoint_and_keeps_truncations_bootstrappable() -> None:
    def make_buffer(terminated_at_second_step: bool) -> ReplayBuffer:
        buffer = ReplayBuffer(size=3)
        for step in range(2):
            buffer.add(
                cast(
                    RolloutBatchProtocol,
                    Batch(
                        obs=np.array([step], dtype=np.float32),
                        act=0,
                        rew=1.0,
                        terminated=step == 1 and terminated_at_second_step,
                        truncated=step == 1 and not terminated_at_second_step,
                        obs_next=np.array([step + 1], dtype=np.float32),
                        info={},
                    ),
                ),
            )
        return buffer

    probabilities = torch.full((1, 2, 5), 0.2)
    terminal_algorithm, _ = make_algorithm(probabilities, n_step=3)
    terminal_buffer = make_buffer(terminated_at_second_step=True)
    terminal_indices = np.array([0])
    terminal_batch = terminal_algorithm._preprocess_batch(
        terminal_buffer[terminal_indices],
        terminal_buffer,
        terminal_indices,
    )
    assert np.array_equal(terminal_batch.c51_terminal, np.array([True]))

    truncated_algorithm, _ = make_algorithm(probabilities, n_step=3)
    truncated_buffer = make_buffer(terminated_at_second_step=False)
    truncated_batch = truncated_algorithm._preprocess_batch(
        truncated_buffer[terminal_indices],
        truncated_buffer,
        terminal_indices,
    )
    assert np.array_equal(truncated_batch.c51_terminal, np.array([False]))
