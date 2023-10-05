from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from torch import nn

from tianshou.highlevel.env import Environments, EnvType
from tianshou.highlevel.module.core import TDevice, init_linear_orthogonal
from tianshou.utils.net import continuous, discrete
from tianshou.utils.net.common import BaseActor, Net
from tianshou.utils.string import ToStringMixin


class ContinuousActorType:
    GAUSSIAN = "gaussian"
    DETERMINISTIC = "deterministic"
    UNSUPPORTED = "unsupported"


class ActorFactory(ToStringMixin, ABC):
    @abstractmethod
    def create_module(self, envs: Environments, device: TDevice) -> BaseActor | nn.Module:
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


class ActorFactoryDefault(ActorFactory):
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

    def create_module(self, envs: Environments, device: TDevice) -> BaseActor:
        env_type = envs.get_type()
        if env_type == EnvType.CONTINUOUS:
            match self.continuous_actor_type:
                case ContinuousActorType.GAUSSIAN:
                    factory = ActorFactoryContinuousGaussianNet(
                        self.hidden_sizes,
                        unbounded=self.continuous_unbounded,
                        conditioned_sigma=self.continuous_conditioned_sigma,
                    )
                case ContinuousActorType.DETERMINISTIC:
                    factory = ActorFactoryContinuousDeterministicNet(self.hidden_sizes)
                case ContinuousActorType.UNSUPPORTED:
                    raise ValueError("Continuous action spaces are not supported by the algorithm")
                case _:
                    raise ValueError(self.continuous_actor_type)
            return factory.create_module(envs, device)
        elif env_type == EnvType.DISCRETE:
            factory = ActorFactoryDiscreteNet(self.DEFAULT_HIDDEN_SIZES)
            return factory.create_module(envs, device)
        else:
            raise ValueError(f"{env_type} not supported")


class ActorFactoryContinuous(ActorFactory, ABC):
    """Serves as a type bound for actor factories that are suitable for continuous action spaces."""


class ActorFactoryContinuousDeterministicNet(ActorFactoryContinuous):
    def __init__(self, hidden_sizes: Sequence[int]):
        self.hidden_sizes = hidden_sizes

    def create_module(self, envs: Environments, device: TDevice) -> BaseActor:
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


class ActorFactoryContinuousGaussianNet(ActorFactoryContinuous):
    def __init__(self, hidden_sizes: Sequence[int], unbounded=True, conditioned_sigma=False):
        self.hidden_sizes = hidden_sizes
        self.unbounded = unbounded
        self.conditioned_sigma = conditioned_sigma

    def create_module(self, envs: Environments, device: TDevice) -> BaseActor:
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


class ActorFactoryDiscreteNet(ActorFactory):
    def __init__(self, hidden_sizes: Sequence[int]):
        self.hidden_sizes = hidden_sizes

    def create_module(self, envs: Environments, device: TDevice) -> BaseActor:
        net_a = Net(
            envs.get_observation_shape(),
            hidden_sizes=self.hidden_sizes,
            device=device,
        )
        return discrete.Actor(
            net_a,
            envs.get_action_shape(),
            hidden_sizes=(),
            device=device,
        ).to(device)
