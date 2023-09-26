from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import torch
from torch import nn

from tianshou.highlevel.env import Environments, EnvType
from tianshou.highlevel.optim import OptimizerFactory
from tianshou.utils.net import continuous
from tianshou.utils.net.common import ActorCritic, Net

TDevice: TypeAlias = str | int | torch.device


def init_linear_orthogonal(module: torch.nn.Module):
    """Applies orthogonal initialization to linear layers of the given module and sets bias weights to 0.

    :param module: the module whose submodules are to be processed
    """
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)


class ContinuousActorType:
    GAUSSIAN = "gaussian"
    DETERMINISTIC = "deterministic"


class ActorFactory(ABC):
    @abstractmethod
    def create_module(self, envs: Environments, device: TDevice) -> nn.Module:
        pass

    @staticmethod
    def _init_linear(actor: torch.nn.Module):
        """Initializes linear layers of an actor module using default mechanisms.

        :param module: the actor module.
        """
        init_linear_orthogonal(actor)
        if hasattr(actor, "mu"):
            # For continuous action spaces with Gaussian policies
            # do last policy layer scaling, this will make initial actions have (close to)
            # 0 mean and std, and will help boost performances,
            # see https://arxiv.org/abs/2006.05990, Fig.24 for details
            for m in actor.mu.modules():
                if isinstance(m, torch.nn.Linear):
                    m.weight.data.copy_(0.01 * m.weight.data)


class DefaultActorFactory(ActorFactory):
    """An actor factory which, depending on the type of environment, creates a suitable MLP-based policy."""

    DEFAULT_HIDDEN_SIZES = (64, 64)

    def __init__(
        self,
        continuous_actor_type: ContinuousActorType,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
        continuous_unbounded=False,
        continuous_conditioned_sigma=False,
    ):
        self.continuous_actor_type = continuous_actor_type
        self.continuous_unbounded = continuous_unbounded
        self.continuous_conditioned_sigma = continuous_conditioned_sigma
        self.hidden_sizes = hidden_sizes

    def create_module(self, envs: Environments, device: TDevice) -> nn.Module:
        env_type = envs.get_type()
        if env_type == EnvType.CONTINUOUS:
            match self.continuous_actor_type:
                case ContinuousActorType.GAUSSIAN:
                    factory = ContinuousActorFactoryGaussian(
                        self.hidden_sizes,
                        unbounded=self.continuous_unbounded,
                        conditioned_sigma=self.continuous_conditioned_sigma,
                    )
                case ContinuousActorType.DETERMINISTIC:
                    factory = ContinuousActorFactoryDeterministic(self.hidden_sizes)
                case _:
                    raise ValueError(self.continuous_actor_type)
            return factory.create_module(envs, device)
        elif env_type == EnvType.DISCRETE:
            raise NotImplementedError
        else:
            raise ValueError(f"{env_type} not supported")


class ContinuousActorFactory(ActorFactory, ABC):
    """Serves as a type bound for actor factories that are suitable for continuous action spaces."""


class ContinuousActorFactoryDeterministic(ContinuousActorFactory):
    def __init__(self, hidden_sizes: Sequence[int]):
        self.hidden_sizes = hidden_sizes

    def create_module(self, envs: Environments, device: TDevice) -> nn.Module:
        net_a = Net(
            envs.get_observation_shape(),
            hidden_sizes=self.hidden_sizes,
            device=device,
        )
        return continuous.Actor(
            net_a,
            envs.get_action_shape(),
            hidden_sizes=(),
            device=device,
        ).to(device)


class ContinuousActorFactoryGaussian(ContinuousActorFactory):
    def __init__(self, hidden_sizes: Sequence[int], unbounded=True, conditioned_sigma=False):
        self.hidden_sizes = hidden_sizes
        self.unbounded = unbounded
        self.conditioned_sigma = conditioned_sigma

    def create_module(self, envs: Environments, device: TDevice) -> nn.Module:
        net_a = Net(
            envs.get_observation_shape(),
            hidden_sizes=self.hidden_sizes,
            activation=nn.Tanh,
            device=device,
        )
        actor = continuous.ActorProb(
            net_a,
            envs.get_action_shape(),
            unbounded=self.unbounded,
            device=device,
            conditioned_sigma=self.conditioned_sigma,
        ).to(device)

        # init params
        if not self.conditioned_sigma:
            torch.nn.init.constant_(actor.sigma_param, -0.5)
        self._init_linear(actor)

        return actor


class CriticFactory(ABC):
    @abstractmethod
    def create_module(self, envs: Environments, device: TDevice, use_action: bool) -> nn.Module:
        pass


class DefaultCriticFactory(CriticFactory):
    """A critic factory which, depending on the type of environment, creates a suitable MLP-based critic."""

    DEFAULT_HIDDEN_SIZES = (64, 64)

    def __init__(self, hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES):
        self.hidden_sizes = hidden_sizes

    def create_module(self, envs: Environments, device: TDevice, use_action: bool) -> nn.Module:
        env_type = envs.get_type()
        if env_type == EnvType.CONTINUOUS:
            factory = ContinuousNetCriticFactory(self.hidden_sizes)
            return factory.create_module(envs, device, use_action)
        elif env_type == EnvType.DISCRETE:
            raise NotImplementedError
        else:
            raise ValueError(f"{env_type} not supported")


class ContinuousCriticFactory(CriticFactory, ABC):
    pass


class ContinuousNetCriticFactory(ContinuousCriticFactory):
    def __init__(self, hidden_sizes: Sequence[int]):
        self.hidden_sizes = hidden_sizes

    def create_module(self, envs: Environments, device: TDevice, use_action: bool) -> nn.Module:
        action_shape = envs.get_action_shape() if use_action else 0
        net_c = Net(
            envs.get_observation_shape(),
            action_shape=action_shape,
            hidden_sizes=self.hidden_sizes,
            concat=use_action,
            activation=nn.Tanh,
            device=device,
        )
        critic = continuous.Critic(net_c, device=device).to(device)
        init_linear_orthogonal(critic)
        return critic


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
