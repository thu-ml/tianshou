import torch
import numpy as np
from copy import deepcopy
import torch.nn.functional as F

from tianshou.data import Batch
from tianshou.policy import DDPGPolicy


class SACPolicy(DDPGPolicy):
    """docstring for SACPolicy"""
    def __init__(self, actor, actor_optim, critic, critic_optim,
                 tau, gamma, ):
        super().__init__()
        self.actor, self.actor_old = actor, deepcopy(actor)
        self.actor_old.eval()
        self.actor_optim = actor_optim
        self.critic, self.critic_old = critic, deepcopy(critic)
        self.critic_old.eval()
        self.critic_optim = critic_optim

    def __call__(self, batch, state=None):
        pass

    def learn(self, batch, batch_size=None, repeat=1):
        pass
