"""Policy package."""
# isort:skip_file

from tianshou.policy.base import Algorithm, TrainingStats
from tianshou.policy.modelfree.pg import Reinforce
from tianshou.policy.modelfree.dqn import DeepQLearning
from tianshou.policy.modelfree.ddpg import DDPG

from tianshou.policy.random import MARLRandomPolicy
from tianshou.policy.modelfree.bdqn import BranchingDuelingQNetwork
from tianshou.policy.modelfree.c51 import C51
from tianshou.policy.modelfree.rainbow import RainbowDQN
from tianshou.policy.modelfree.qrdqn import QRDQN
from tianshou.policy.modelfree.iqn import IQN
from tianshou.policy.modelfree.fqf import FQF
from tianshou.policy.modelfree.a2c import A2C
from tianshou.policy.modelfree.npg import NPG
from tianshou.policy.modelfree.ppo import PPO
from tianshou.policy.modelfree.trpo import TRPO
from tianshou.policy.modelfree.td3 import TD3
from tianshou.policy.modelfree.sac import SAC
from tianshou.policy.modelfree.redq import REDQ
from tianshou.policy.modelfree.discrete_sac import DiscreteSAC
from tianshou.policy.imitation.base import ImitationLearning
from tianshou.policy.imitation.bcq import BCQ
from tianshou.policy.imitation.cql import CQL
from tianshou.policy.imitation.td3_bc import TD3BCPolicy
from tianshou.policy.imitation.discrete_bcq import DiscreteBCQ
from tianshou.policy.imitation.discrete_cql import DiscreteCQLPolicy
from tianshou.policy.imitation.discrete_crr import DiscreteCRR
from tianshou.policy.imitation.gail import GAIL
from tianshou.policy.modelbased.psrl import PSRLPolicy
from tianshou.policy.modelbased.icm import ICMOffPolicyWrapper
from tianshou.policy.multiagent.mapolicy import MultiAgentPolicyManager

__all__ = [
    "Algorithm",
    "MARLRandomPolicy",
    "DeepQLearning",
    "BranchingDuelingQNetwork",
    "C51",
    "RainbowDQN",
    "QRDQN",
    "IQN",
    "FQF",
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
    "ImitationLearning",
    "BCQ",
    "CQL",
    "TD3BCPolicy",
    "DiscreteBCQ",
    "DiscreteCQLPolicy",
    "DiscreteCRR",
    "GAIL",
    "PSRLPolicy",
    "ICMOffPolicyWrapper",
    "MultiAgentPolicyManager",
    "TrainingStats",
]
