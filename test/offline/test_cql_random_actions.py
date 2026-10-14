"""Regression tests for the uniform random-action stream of CQL's conservative regularizer."""

import math
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch

from tianshou.algorithm import CQL
from tianshou.algorithm.modelfree.sac import SACPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Batch
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ContinuousActorProbabilistic, ContinuousCritic

OBS_DIM, BATCH_SIZE, NUM_REPEAT = 4, 16, 10


def make_cql(low: float, high: float, d: int, calibrated: bool = False) -> CQL:
    actor_net = Net(state_shape=(OBS_DIM,), action_shape=(d,), hidden_sizes=[8])
    critic_net = Net(state_shape=(OBS_DIM,), action_shape=(d,), hidden_sizes=[8], concat=True)
    actor = ContinuousActorProbabilistic(
        preprocess_net=actor_net, action_shape=(d,), unbounded=True, conditioned_sigma=True
    )
    space = gym.spaces.Box(low=-1.0, high=1.0, shape=(d,), dtype=np.float32)
    return CQL(
        policy=SACPolicy(actor=actor, action_scaling=False, action_space=space),
        policy_optim=AdamOptimizerFactory(lr=1e-3),
        critic=ContinuousCritic(preprocess_net=critic_net),
        critic_optim=AdamOptimizerFactory(lr=1e-3),
        min_action=low,
        max_action=high,
        num_repeat_actions=NUM_REPEAT,
        calibrated=calibrated,
    )


def make_batch(d: int) -> Batch:
    rng = np.random.default_rng(0)
    n = BATCH_SIZE
    return Batch(
        obs=rng.normal(size=(n, OBS_DIM)).astype(np.float32),
        act=rng.uniform(-1, 1, size=(n, d)).astype(np.float32),
        rew=rng.normal(size=n).astype(np.float32),
        obs_next=rng.normal(size=(n, OBS_DIM)).astype(np.float32),
        done=np.zeros(n, dtype=bool),
        calibration_returns=rng.normal(size=n).astype(np.float32),
    )


@pytest.mark.parametrize(
    ("low", "high", "d"),
    [(-1.0, 1.0, 3), (-0.4, 0.4, 17), (-3.0, 1.0, 1), (-2.0, 2.0, 1)],
)
def test_random_actions_are_uniform_in_the_box_and_weighted_by_its_density(
    low: float, high: float, d: int
) -> None:
    algo = make_cql(low, high, d)
    calls = []
    original = algo._calc_random_values

    def spy(obs: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v1, v2 = original(obs, act)
        calls.append((act.detach().clone(), (v1 - algo.critic(obs, act)).detach()))
        return v1, v2

    algo._calc_random_values = spy  # type: ignore[method-assign]
    algo._update_with_batch(make_batch(d))

    ((act, offset),) = calls
    assert act.shape == (BATCH_SIZE * NUM_REPEAT, d)
    # regression: all samples used to be the single point `(max_action, ..., max_action)`
    assert len(torch.unique(act, dim=0)) == len(act)
    assert (act >= low).all()
    assert (act <= high).all()
    # the values are importance weights Q(s, a) - log(1/volume) = Q(s, a) + d * log(high - low)
    torch.testing.assert_close(offset, torch.full_like(offset, d * math.log(high - low)))


@pytest.mark.parametrize("calibrated", [False, True])
def test_logsumexp_pools_all_samples_of_a_state(
    calibrated: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The log-sum-exp must see (batch_size, 3 * num_repeat_actions, 1), not (batch_size * num_repeat, 3)."""
    algo = make_cql(-1.0, 1.0, 3, calibrated=calibrated)
    shapes: list[tuple[int, ...]] = []
    original = torch.logsumexp

    def recording_logsumexp(x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        shapes.append(tuple(x.shape))
        return original(x, *args, **kwargs)

    monkeypatch.setattr(torch, "logsumexp", recording_logsumexp)
    algo._update_with_batch(make_batch(3))
    assert shapes == [(BATCH_SIZE, 3 * NUM_REPEAT, 1)] * 2  # one per critic
