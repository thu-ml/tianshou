"""Policy package."""
# isort:skip_file

from tianshou.policy.base import Algorithm, TrainingStats
from tianshou.policy.modelfree.pg import Reinforce
from tianshou.policy.modelfree.dqn import DeepQLearning
from tianshou.policy.modelfree.ddpg import DDPG

from tianshou.policy.random import MARLRandomPolicy
from tianshou.policy.modelfree.bdqn import BranchingDuelingQNetwork
from tianshou.policy.modelfree.c51 import C51
from tianshou.policy.modelfree.rainbow import RainbowPolicy
from tianshou.policy.modelfree.qrdqn import QRDQN
from tianshou.policy.modelfree.iqn import IQNPolicy
from tianshou.policy.modelfree.fqf import FQFPolicy
from tianshou.policy.modelfree.a2c import A2C
from tianshou.policy.modelfree.npg import NPG
from tianshou.policy.modelfree.ppo import PPO
from tianshou.policy.modelfree.trpo import TRPO
from tianshou.policy.modelfree.td3 import TD3
from tianshou.policy.modelfree.sac import SAC
from tianshou.policy.modelfree.redq import REDQ
from tianshou.policy.modelfree.discrete_sac import DiscreteSAC
from tianshou.policy.imitation.base import ImitationPolicy
from tianshou.policy.imitation.bcq import BCQPolicy
from tianshou.policy.imitation.cql import CQLPolicy
from tianshou.policy.imitation.td3_bc import TD3BCPolicy
from tianshou.policy.imitation.discrete_bcq import DiscreteBCQPolicy
from tianshou.policy.imitation.discrete_cql import DiscreteCQLPolicy
from tianshou.policy.imitation.discrete_crr import DiscreteCRRPolicy
from tianshou.policy.imitation.gail import GAILPolicy
from tianshou.policy.modelbased.psrl import PSRLPolicy
from tianshou.policy.modelbased.icm import ICMPolicy
from tianshou.policy.multiagent.mapolicy import MultiAgentPolicyManager

__all__ = [
    "Algorithm",
    "MARLRandomPolicy",
    "DeepQLearning",
    "BranchingDuelingQNetwork",
    "C51",
    "RainbowPolicy",
    "QRDQN",
    "IQNPolicy",
    "FQFPolicy",
    "Reinforce",
    "A2C",
    "NPG",
    "DDPG",
    "PPO",
    "TRPO",
    "TD3",
    "SAC",
    "REDQ",
    "DiscreteSAC",
    "ImitationPolicy",
    "BCQPolicy",
    "CQLPolicy",
    "TD3BCPolicy",
    "DiscreteBCQPolicy",
    "DiscreteCQLPolicy",
    "DiscreteCRRPolicy",
    "GAILPolicy",
    "PSRLPolicy",
    "ICMPolicy",
    "MultiAgentPolicyManager",
    "TrainingStats",
]
