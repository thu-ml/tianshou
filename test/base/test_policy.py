import gymnasium as gym
import numpy as np
import pytest
import torch
from torch.distributions import Categorical, Distribution, Independent, Normal

from tianshou.data import Batch
from tianshou.policy import BasePolicy, PPOPolicy
from tianshou.policy.base import RandomActionPolicy, episode_mc_return_to_go
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils.net.discrete import Actor

obs_shape = (5,)


def _to_hashable(x: np.ndarray | int) -> int | tuple[list]:
    return x if isinstance(x, int) else tuple(x.tolist())


def test_calculate_discounted_returns() -> None:
    assert np.all(
        episode_mc_return_to_go([1, 1, 1], 0.9) == np.array([0.9**2 + 0.9 + 1, 0.9 + 1, 1]),
    )
    assert episode_mc_return_to_go([1, 2, 3], 0.5)[0] == 1 + 0.5 * (2 + 0.5 * 3)


@pytest.fixture(params=["continuous", "discrete"])
def policy(request: pytest.FixtureRequest) -> PPOPolicy:
    action_type = request.param
    action_space: gym.spaces.Box | gym.spaces.Discrete
    actor: Actor | ActorProb
    if action_type == "continuous":
        action_space = gym.spaces.Box(low=-1, high=1, shape=(3,))
        actor = ActorProb(
            Net(state_shape=obs_shape, hidden_sizes=[64, 64], action_shape=action_space.shape),
            action_shape=action_space.shape,
        )

        def dist_fn(loc_scale: tuple[torch.Tensor, torch.Tensor]) -> Distribution:
            loc, scale = loc_scale
            return Independent(Normal(loc, scale), 1)

    elif action_type == "discrete":
        action_space = gym.spaces.Discrete(3)
        actor = Actor(
            Net(state_shape=obs_shape, hidden_sizes=[64, 64], action_shape=action_space.n),
            action_shape=action_space.n,
        )
        dist_fn = Categorical
    else:
        raise ValueError(f"Unknown action type: {action_type}")

    critic = Critic(
        Net(obs_shape, hidden_sizes=[64, 64]),
    )

    actor_critic = ActorCritic(actor, critic)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=1e-3)

    policy: BasePolicy
    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        dist_fn=dist_fn,
        optim=optim,
        action_space=action_space,
        action_scaling=False,
    )
    policy.eval()
    return policy


class TestPolicyBasics:
    def test_get_action(self, policy: PPOPolicy) -> None:
        policy.is_within_training_step = False
        sample_obs = torch.randn(obs_shape)
        policy.deterministic_eval = False
        actions = [policy.compute_action(sample_obs) for _ in range(10)]
        assert all(policy.action_space.contains(a) for a in actions)

        # check that the actions are different in non-deterministic mode
        assert len(set(map(_to_hashable, actions))) > 1

        policy.deterministic_eval = True
        actions = [policy.compute_action(sample_obs) for _ in range(10)]
        # check that the actions are the same in deterministic mode
        assert len(set(map(_to_hashable, actions))) == 1

    @staticmethod
    def test_random_policy_discrete_actions() -> None:
        action_space = gym.spaces.Discrete(3)
        policy = RandomActionPolicy(action_space=action_space)

        # forward of actor returns discrete probabilities, in compliance with the overall discrete actor
        action_probs = policy.actor(np.zeros((10, 2)))[0]
        assert np.allclose(action_probs, 1 / 3 * np.ones((10, 3)))

        actions = []
        for _ in range(10):
            action = policy.compute_action(np.array([0]))
            assert action_space.contains(action)
            actions.append(action)

        # not all actions are the same
        assert len(set(actions)) > 1

        # test batched forward
        action_batch = policy(Batch(obs=np.zeros((10, 2))))
        assert action_batch.act.shape == (10,)
        assert len(set(action_batch.act.tolist())) > 1

    @staticmethod
    def test_random_policy_continuous_actions() -> None:
        action_space = gym.spaces.Box(low=-1, high=1, shape=(3,))
        policy = RandomActionPolicy(action_space=action_space)

        actions = []
        for _ in range(10):
            action = policy.compute_action(np.array([0]))
            assert action_space.contains(action)
            actions.append(action)

        # not all actions are the same
        assert len(set(map(_to_hashable, actions))) > 1

        # test batched forward
        action_batch = policy(Batch(obs=np.zeros((10, 2))))
        assert action_batch.act.shape == (10, 3)
        assert len(set(map(_to_hashable, action_batch.act))) > 1
