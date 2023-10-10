from dataclasses import dataclass

import torch

from tianshou.utils.net.common import ActorCritic


@dataclass
class ModuleOpt:
    module: torch.nn.Module
    optim: torch.optim.Optimizer


@dataclass
class ActorCriticModuleOpt:
    actor_critic_module: ActorCritic
    optim: torch.optim.Optimizer

    @property
    def actor(self) -> torch.nn.Module:
        return self.actor_critic_module.actor

    @property
    def critic(self) -> torch.nn.Module:
        return self.actor_critic_module.critic
