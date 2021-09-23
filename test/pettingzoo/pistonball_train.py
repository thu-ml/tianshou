from pettingzoo.butterfly import pistonball_v4
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.env import DummyVectorEnv
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import RandomPolicy, MultiAgentPolicyManager
import numpy as np
import torch


def get_env():
    return PettingZooEnv(pistonball_v4.env(continuous=False))


train_envs = DummyVectorEnv([get_env for _ in range(10)])
# test_envs = DummyVectorEnv([get_env for _ in range(100)])

# seed
np.random.seed(1626)
torch.manual_seed(1626)
train_envs.seed(1626)
# test_envs.seed(1626)

policy = MultiAgentPolicyManager([RandomPolicy() for _ in range(len(get_env().agents))], get_env())

# collector
train_collector = Collector(policy, train_envs,
        VectorReplayBuffer(6, len(train_envs)),
        exploration_noise=True)
# test_collector = Collector(policy, test_envs, exploration_noise=True)
# policy.set_eps(1)
train_collector.collect(n_step=640, render=0.0001)
