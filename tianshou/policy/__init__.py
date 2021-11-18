"""Policy package."""
# isort:skip_file

from tianshou.policy.base import BasePolicy
from tianshou.policy.random import RandomPolicy
from tianshou.policy.modelfree.dqn import DQNPolicy
from tianshou.policy.modelfree.c51 import C51Policy
from tianshou.policy.modelfree.rainbow import RainbowPolicy
from tianshou.policy.modelfree.qrdqn import QRDQNPolicy
from tianshou.policy.modelfree.iqn import IQNPolicy
from tianshou.policy.modelfree.fqf import FQFPolicy
from tianshou.policy.modelfree.pg import PGPolicy
from tianshou.policy.modelfree.a2c import A2CPolicy
from tianshou.policy.modelfree.npg import NPGPolicy
from tianshou.policy.modelfree.ddpg import DDPGPolicy
from tianshou.policy.modelfree.ppo import PPOPolicy
from tianshou.policy.modelfree.trpo import TRPOPolicy
from tianshou.policy.modelfree.td3 import TD3Policy
from tianshou.policy.modelfree.sac import SACPolicy
from tianshou.policy.modelfree.discrete_sac import DiscreteSACPolicy
from tianshou.policy.imitation.base import ImitationPolicy
from tianshou.policy.imitation.discrete_bcq import DiscreteBCQPolicy
from tianshou.policy.imitation.discrete_cql import DiscreteCQLPolicy
from tianshou.policy.imitation.discrete_crr import DiscreteCRRPolicy
from tianshou.policy.modelbased.psrl import PSRLPolicy
from tianshou.policy.multiagent.mapolicy import MultiAgentPolicyManager

__all__ = [
    "BasePolicy",
    "RandomPolicy",
    "DQNPolicy",
    "C51Policy",
    "RainbowPolicy",
    "QRDQNPolicy",
    "IQNPolicy",
    "FQFPolicy",
    "PGPolicy",
    "A2CPolicy",
    "NPGPolicy",
    "DDPGPolicy",
    "PPOPolicy",
    "TRPOPolicy",
    "TD3Policy",
    "SACPolicy",
    "DiscreteSACPolicy",
    "ImitationPolicy",
    "DiscreteBCQPolicy",
    "DiscreteCQLPolicy",
    "DiscreteCRRPolicy",
    "PSRLPolicy",
    "MultiAgentPolicyManager",
]
