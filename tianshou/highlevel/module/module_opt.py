from dataclasses import dataclass

import torch

from tianshou.utils.net.common import ActorCritic


@dataclass
class ModuleOpt:
    """Container for a torch module along with its optimizer."""

    module: torch.nn.Module
    optim: torch.optim.Optimizer


@dataclass
class ActorCriticOpt:
    """Container for an :class:`ActorCritic` instance along with its optimizer."""

    actor_critic_module: ActorCritic
    optim: torch.optim.Optimizer

    @property
    def actor(self) -> torch.nn.Module:
        return self.actor_critic_module.actor

    @property
    def critic(self) -> torch.nn.Module:
        return self.actor_critic_module.critic
