import gymnasium as gym
import numpy as np
import pytest
import torch
from torch.distributions import Categorical, Independent, Normal

from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils.net.discrete import Actor

obs_shape = (5,)


def _to_hashable(x: np.ndarray | int):
    return x if isinstance(x, int) else tuple(x.tolist())


@pytest.fixture(params=["continuous", "discrete"])
def policy(request):
    action_type = request.param
    if action_type == "continuous":
        action_space = gym.spaces.Box(low=-1, high=1, shape=(3,))
        actor = ActorProb(
            Net(state_shape=obs_shape, hidden_sizes=[64, 64], action_shape=action_space.shape),
            action_shape=action_space.shape,
        )
        dist_fn = lambda *logits: Independent(Normal(*logits), 1)
    elif action_type == "discrete":
        action_space = gym.spaces.Discrete(3)
        actor = Actor(
            Net(state_shape=obs_shape, hidden_sizes=[64, 64], action_shape=action_space.n),
            action_shape=action_space.n,
        )
        dist_fn = lambda logits: Categorical(logits=logits)
    else:
        raise ValueError(f"Unknown action type: {action_type}")

    critic = Critic(
        Net(obs_shape, hidden_sizes=[64, 64]),
    )

    actor_critic = ActorCritic(actor, critic)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=1e-3)

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
    def test_get_action(self, policy):
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
