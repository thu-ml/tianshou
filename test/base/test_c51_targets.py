from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch

from tianshou.algorithm import C51, RainbowDQN
from tianshou.algorithm.modelfree.c51 import C51Policy
from tianshou.algorithm.modelfree.reinforce import LossSequenceTrainingStats
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Batch, ReplayBuffer
from tianshou.utils.torch_utils import policy_within_training_step


class _CategoricalTable(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        probabilities = torch.tensor(
            [
                [[0.3, 0.4, 0.3], [0.2, 0.6, 0.2]],
                [[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]],
                [[0.2, 0.6, 0.2], [0.1, 0.2, 0.7]],
                [[0.6, 0.3, 0.1], [0.1, 0.3, 0.6]],
                [[0.2, 0.2, 0.6], [0.5, 0.3, 0.2]],
                [[0.5, 0.4, 0.1], [0.2, 0.3, 0.5]],
                [[0.1, 0.5, 0.4], [0.6, 0.2, 0.2]],
            ]
        )
        self.logits = torch.nn.Parameter(probabilities.log())

    def forward(
        self, obs: np.ndarray, state: Any = None, info: Any = None
    ) -> tuple[torch.Tensor, Any]:
        indices = torch.as_tensor(obs, dtype=torch.long).flatten()
        return self.logits[indices].softmax(-1), state


def _project_distribution(
    probabilities: np.ndarray, support: np.ndarray, reward: float, discount: float
) -> np.ndarray:
    """Deposit each Bellman atom into its two neighboring support points."""
    projected = np.zeros_like(support)
    for probability, atom in zip(probabilities, support, strict=True):
        position = np.clip(reward + discount * atom, support[0], support[-1])
        coordinate = (position - support[0]) / (support[1] - support[0])
        lower = int(np.floor(coordinate))
        upper = int(np.ceil(coordinate))
        if lower == upper:
            projected[lower] += probability
        else:
            projected[lower] += probability * (upper - coordinate)
            projected[upper] += probability * (coordinate - lower)
    return projected


@pytest.mark.parametrize("algorithm_class", [C51, RainbowDQN])
@pytest.mark.parametrize("n_step", [1, 3])
@pytest.mark.parametrize("target_update_freq", [0, 100])
@pytest.mark.parametrize("ending", ["continuing", "terminated", "truncated", "unfinished"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_categorical_nstep_target(
    algorithm_class: type[C51],
    n_step: int,
    target_update_freq: int,
    ending: str,
    wrapped: bool,
) -> None:
    model = _CategoricalTable()
    policy = C51Policy(
        model=model, action_space=gym.spaces.Discrete(2), num_atoms=3, v_min=-1, v_max=1
    )
    algorithm = algorithm_class(
        policy=policy,
        optim=AdamOptimizerFactory(lr=0),
        gamma=0.5,
        n_step_return_horizon=n_step,
        target_update_freq=target_update_freq,
    )
    online = model.logits.detach().softmax(-1).numpy()
    target = online
    if target_update_freq:
        # The target network prefers the opposite action. Double Q must still select
        # the action using the online network, at the same n-step observation.
        assert algorithm.model_old is not None
        with torch.no_grad():
            next(algorithm.model_old.parameters()).copy_(model.logits.flip(-1))
        target = online[..., ::-1]
        algorithm._iter = 1  # keep the distinct target network for this update

    states = [0, 1, 2, 3]
    next_states = [1, 2, 3, 4]
    rewards = [0.2, 0.4, -0.3, 0.1]
    if ending in ("terminated", "truncated"):
        states[2:] = [4, 5]
        next_states[2:] = [5, 6]
    if ending == "unfinished":
        states, next_states, rewards = states[:2], next_states[:2], rewards[:2]
    buffer = ReplayBuffer(len(states))
    if wrapped:
        for _ in range(2):
            buffer.add(
                Batch(obs=[6], act=0, rew=99.0, terminated=True, truncated=False, obs_next=[6])
            )
    indices = []
    for i, (state, next_state, reward) in enumerate(zip(states, next_states, rewards, strict=True)):
        index, _, _, _ = buffer.add(
            Batch(
                obs=[state],
                act=i % 2,
                rew=reward,
                terminated=ending == "terminated" and i == 1,
                truncated=ending == "truncated" and i == 1,
                obs_next=[next_state],
            )
        )
        indices.append(int(index[0]))

    support = np.array([-1.0, 0.0, 1.0])
    expected = []
    for start in range(len(states)):
        reward_sum = 0.0
        discount = 1.0
        for end in range(start, min(start + n_step, len(states))):
            reward_sum += discount * rewards[end]
            discount *= 0.5
            if ending in ("terminated", "truncated") and end == 1:
                if ending == "terminated":
                    discount = 0.0
                break
        next_state = next_states[end]
        action = (online[next_state] @ support).argmax()
        expected.append(
            _project_distribution(target[next_state, action], support, reward_sum, discount)
        )

    batch = buffer[np.array(indices)]
    assert isinstance(batch.obs, np.ndarray)
    original_obs = batch.obs.copy()
    batch = algorithm._preprocess_batch(batch, buffer, np.array(indices))
    with torch.no_grad():
        actual = algorithm._target_dist(batch).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    np.testing.assert_array_equal(batch.obs, original_obs)
    # The actual public update must train the original state/action pairs against
    # those targets, including when the replay ring wraps or an episode ends early.
    expected_loss = np.mean(
        [
            -(distribution * np.log(online[state, i % 2] + 1e-8)).sum()
            for i, (state, distribution) in enumerate(zip(states, expected, strict=True))
        ]
    )
    with policy_within_training_step(policy):
        stats = algorithm.update(buffer=buffer, sample_size=0)
    assert isinstance(stats, LossSequenceTrainingStats)
    assert stats.loss == pytest.approx(expected_loss, abs=1e-6)
    np.testing.assert_array_equal(buffer[np.array(indices)].obs_next[:, 0], next_states)
