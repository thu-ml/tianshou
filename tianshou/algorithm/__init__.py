"""Algorithm package."""
# isort:skip_file

from tianshou.algorithm.algorithm_base import Algorithm, TrainingStats
from tianshou.algorithm.modelfree.pg import Reinforce
from tianshou.algorithm.modelfree.dqn import DQN
from tianshou.algorithm.modelfree.ddpg import DDPG

from tianshou.algorithm.random import MARLRandomDiscreteMaskedOffPolicyAlgorithm
from tianshou.algorithm.modelfree.bdqn import BDQN
from tianshou.algorithm.modelfree.c51 import C51
from tianshou.algorithm.modelfree.rainbow import RainbowDQN
from tianshou.algorithm.modelfree.qrdqn import QRDQN
from tianshou.algorithm.modelfree.iqn import IQN
from tianshou.algorithm.modelfree.fqf import FQF
from tianshou.algorithm.modelfree.a2c import A2C
from tianshou.algorithm.modelfree.npg import NPG
from tianshou.algorithm.modelfree.ppo import PPO
from tianshou.algorithm.modelfree.trpo import TRPO
from tianshou.algorithm.modelfree.td3 import TD3
from tianshou.algorithm.modelfree.sac import SAC
from tianshou.algorithm.modelfree.redq import REDQ
from tianshou.algorithm.modelfree.discrete_sac import DiscreteSAC
from tianshou.algorithm.imitation.base import OffPolicyImitationLearning
from tianshou.algorithm.imitation.bcq import BCQ
from tianshou.algorithm.imitation.cql import CQL
from tianshou.algorithm.imitation.td3_bc import TD3BC
from tianshou.algorithm.imitation.discrete_bcq import DiscreteBCQ
from tianshou.algorithm.imitation.discrete_cql import DiscreteCQL
from tianshou.algorithm.imitation.discrete_crr import DiscreteCRR
from tianshou.algorithm.imitation.gail import GAIL
from tianshou.algorithm.modelbased.psrl import PSRL
from tianshou.algorithm.modelbased.icm import ICMOffPolicyWrapper
from tianshou.algorithm.modelbased.icm import ICMOnPolicyWrapper
from tianshou.algorithm.multiagent.mapolicy import MultiAgentOffPolicyAlgorithm
