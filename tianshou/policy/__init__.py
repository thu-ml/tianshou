from tianshou.policy.base import BasePolicy
from tianshou.policy.dqn import DQNPolicy
from tianshou.policy.pg import PGPolicy
from tianshou.policy.a2c import A2CPolicy
from tianshou.policy.ddpg import DDPGPolicy

__all__ = [
    'BasePolicy',
    'DQNPolicy',
    'PGPolicy',
    'A2CPolicy',
    'DDPGPolicy'
]
