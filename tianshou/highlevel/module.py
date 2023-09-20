from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from tianshou.highlevel.env import Environments
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb
from tianshou.utils.net.continuous import Critic as ContinuousCritic

TDevice = str | int | torch.device


def init_linear_orthogonal(module: torch.nn.Module):
    """Applies orthogonal initialization to linear layers of the given module and sets bias weights to 0.

    :param module: the module whose submodules are to be processed
    """
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)


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


class ContinuousActorFactory(ActorFactory, ABC):
    pass


class ContinuousActorProbFactory(ContinuousActorFactory):
    def __init__(self, hidden_sizes: Sequence[int], unbounded=True, conditioned_sigma=False):
        self.hidden_sizes = hidden_sizes
        self.unbounded = unbounded
        self.conditioned_sigma = conditioned_sigma

    def create_module(self, envs: Environments, device: TDevice) -> nn.Module:
        net_a = Net(
            envs.get_state_shape(),
            hidden_sizes=self.hidden_sizes,
            activation=nn.Tanh,
            device=device,
        )
        actor = ActorProb(
            net_a,
            envs.get_action_shape(),
            unbounded=True,
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


class ContinuousCriticFactory(CriticFactory, ABC):
    pass


class ContinuousNetCriticFactory(ContinuousCriticFactory):
    def __init__(self, hidden_sizes: Sequence[int], action_shape=0):
        self.action_shape = action_shape
        self.hidden_sizes = hidden_sizes

    def create_module(self, envs: Environments, device: TDevice, use_action: bool) -> nn.Module:
        action_shape = envs.get_action_shape() if use_action else 0
        net_c = Net(
            envs.get_state_shape(),
            action_shape=action_shape,
            hidden_sizes=self.hidden_sizes,
            concat=use_action,
            activation=nn.Tanh,
            device=device,
        )
        critic = ContinuousCritic(net_c, device=device).to(device)
        init_linear_orthogonal(critic)
        return critic
