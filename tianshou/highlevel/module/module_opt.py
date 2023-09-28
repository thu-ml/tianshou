from dataclasses import dataclass

import torch

from tianshou.highlevel.env import Environments
from tianshou.highlevel.module.actor import ActorFactory
from tianshou.highlevel.module.core import TDevice
from tianshou.highlevel.module.critic import CriticFactory
from tianshou.highlevel.optim import OptimizerFactory
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
    def actor(self):
        return self.actor_critic_module.actor

    @property
    def critic(self):
        return self.actor_critic_module.critic


class ActorModuleOptFactory:
    def __init__(self, actor_factory: ActorFactory, optim_factory: OptimizerFactory):
        self.actor_factory = actor_factory
        self.optim_factory = optim_factory

    def create_module_opt(self, envs: Environments, device: TDevice, lr: float) -> ModuleOpt:
        actor = self.actor_factory.create_module(envs, device)
        opt = self.optim_factory.create_optimizer(actor, lr)
        return ModuleOpt(actor, opt)


class CriticModuleOptFactory:
    def __init__(
        self,
        critic_factory: CriticFactory,
        optim_factory: OptimizerFactory,
        use_action: bool,
    ):
        self.critic_factory = critic_factory
        self.optim_factory = optim_factory
        self.use_action = use_action

    def create_module_opt(self, envs: Environments, device: TDevice, lr: float) -> ModuleOpt:
        critic = self.critic_factory.create_module(envs, device, self.use_action)
        opt = self.optim_factory.create_optimizer(critic, lr)
        return ModuleOpt(critic, opt)
