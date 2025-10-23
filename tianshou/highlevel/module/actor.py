from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import torch
from sensai.util.string import ToStringMixin
from torch import nn

from tianshou.algorithm.modelfree.reinforce import TDistFnDiscrOrCont
from tianshou.highlevel.env import Environments, EnvType
from tianshou.highlevel.module.core import (
    ModuleFactory,
    TDevice,
    init_linear_orthogonal,
)
from tianshou.highlevel.module.intermediate import (
    IntermediateModule,
    IntermediateModuleFactory,
)
from tianshou.highlevel.params.dist_fn import (
    DistributionFunctionFactoryCategorical,
    DistributionFunctionFactoryIndependentGaussians,
)
from tianshou.utils.net import continuous, discrete
from tianshou.utils.net.common import (
    Actor,
    ModuleType,
    ModuleWithVectorOutput,
    Net,
)


class ContinuousActorType(Enum):
    GAUSSIAN = "gaussian"
    DETERMINISTIC = "deterministic"
    UNSUPPORTED = "unsupported"


@dataclass
class ActorFuture:
    """Container, which, in the future, will hold an actor instance."""

    actor: Actor | nn.Module | None = None


class ActorFutureProviderProtocol(Protocol):
    def get_actor_future(self) -> ActorFuture:
        pass


class ActorFactory(ModuleFactory, ToStringMixin, ABC):
    @abstractmethod
    def create_module(self, envs: Environments, device: TDevice) -> Actor | nn.Module:
        pass

    @abstractmethod
    def create_dist_fn(self, envs: Environments) -> TDistFnDiscrOrCont | None:
        """
        :param envs: the environments
        :return: the distribution function, which converts the actor's output into a distribution, or None
            if the actor does not output distribution parameters
        """

    @staticmethod
    def _init_linear(actor: torch.nn.Module) -> None:
        """Initializes linear layers of an actor module using default mechanisms.

        :param actor: the actor module.
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
        hidden_activation: ModuleType = nn.ReLU,
        continuous_unbounded: bool = False,
        continuous_conditioned_sigma: bool = False,
        discrete_softmax: bool = True,
    ):
        self.continuous_actor_type = continuous_actor_type
        self.continuous_unbounded = continuous_unbounded
        self.continuous_conditioned_sigma = continuous_conditioned_sigma
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.discrete_softmax = discrete_softmax

    def _create_factory(self, envs: Environments) -> ActorFactory:
        env_type = envs.get_type()
        factory: (
            ActorFactoryContinuousDeterministicNet
            | ActorFactoryContinuousGaussianNet
            | ActorFactoryDiscreteNet
        )
        if env_type == EnvType.CONTINUOUS:
            match self.continuous_actor_type:
                case ContinuousActorType.GAUSSIAN:
                    factory = ActorFactoryContinuousGaussianNet(
                        self.hidden_sizes,
                        activation=self.hidden_activation,
                        unbounded=self.continuous_unbounded,
                        conditioned_sigma=self.continuous_conditioned_sigma,
                    )
                case ContinuousActorType.DETERMINISTIC:
                    factory = ActorFactoryContinuousDeterministicNet(
                        self.hidden_sizes,
                        activation=self.hidden_activation,
                    )
                case ContinuousActorType.UNSUPPORTED:
                    raise ValueError("Continuous action spaces are not supported by the algorithm")
                case _:
                    raise ValueError(self.continuous_actor_type)
        elif env_type == EnvType.DISCRETE:
            factory = ActorFactoryDiscreteNet(
                self.hidden_sizes,
                activation=self.hidden_activation,
                softmax_output=self.discrete_softmax,
            )
        else:
            raise ValueError(f"{env_type} not supported")
        return factory

    def create_module(self, envs: Environments, device: TDevice) -> Actor | nn.Module:
        factory = self._create_factory(envs)
        return factory.create_module(envs, device)

    def create_dist_fn(self, envs: Environments) -> TDistFnDiscrOrCont | None:
        factory = self._create_factory(envs)
        return factory.create_dist_fn(envs)


class ActorFactoryContinuous(ActorFactory, ABC):
    """Serves as a type bound for actor factories that are suitable for continuous action spaces."""


class ActorFactoryContinuousDeterministicNet(ActorFactoryContinuous):
    def __init__(self, hidden_sizes: Sequence[int], activation: ModuleType = nn.ReLU):
        self.hidden_sizes = hidden_sizes
        self.activation = activation

    def create_module(self, envs: Environments, device: TDevice) -> Actor:
        net_a = Net(
            state_shape=envs.get_observation_shape(),
            hidden_sizes=self.hidden_sizes,
            activation=self.activation,
        )
        return continuous.ContinuousActorDeterministic(
            preprocess_net=net_a,
            action_shape=envs.get_action_shape(),
            hidden_sizes=(),
        ).to(device)

    def create_dist_fn(self, envs: Environments) -> TDistFnDiscrOrCont | None:
        return None


class ActorFactoryContinuousGaussianNet(ActorFactoryContinuous):
    def __init__(
        self,
        hidden_sizes: Sequence[int],
        unbounded: bool = True,
        conditioned_sigma: bool = False,
        activation: ModuleType = nn.ReLU,
    ):
        """For actors with Gaussian policies.

        :param hidden_sizes: the sequence of hidden dimensions to use in the network structure
        :param unbounded: whether to apply tanh activation on final logits
        :param conditioned_sigma: if True, the standard deviation of continuous actions (sigma) is computed from the
            input; if False, sigma is an independent parameter
        """
        self.hidden_sizes = hidden_sizes
        self.unbounded = unbounded
        self.conditioned_sigma = conditioned_sigma
        self.activation = activation

    def create_module(self, envs: Environments, device: TDevice) -> Actor:
        net_a = Net(
            state_shape=envs.get_observation_shape(),
            hidden_sizes=self.hidden_sizes,
            activation=self.activation,
        )
        actor = continuous.ContinuousActorProbabilistic(
            preprocess_net=net_a,
            action_shape=envs.get_action_shape(),
            unbounded=self.unbounded,
            conditioned_sigma=self.conditioned_sigma,
        ).to(device)

        # init params
        if not self.conditioned_sigma:
            torch.nn.init.constant_(actor.sigma_param, -0.5)
        self._init_linear(actor)

        return actor

    def create_dist_fn(self, envs: Environments) -> TDistFnDiscrOrCont | None:
        return DistributionFunctionFactoryIndependentGaussians().create_dist_fn(envs)


class ActorFactoryDiscreteNet(ActorFactory):
    def __init__(
        self,
        hidden_sizes: Sequence[int],
        softmax_output: bool = True,
        activation: ModuleType = nn.ReLU,
    ):
        self.hidden_sizes = hidden_sizes
        self.softmax_output = softmax_output
        self.activation = activation

    def create_module(self, envs: Environments, device: TDevice) -> Actor:
        net_a = Net(
            state_shape=envs.get_observation_shape(),
            hidden_sizes=self.hidden_sizes,
            activation=self.activation,
        )
        return discrete.DiscreteActor(
            preprocess_net=net_a,
            action_shape=envs.get_action_shape(),
            hidden_sizes=(),
            softmax_output=self.softmax_output,
        ).to(device)

    def create_dist_fn(self, envs: Environments) -> TDistFnDiscrOrCont | None:
        return DistributionFunctionFactoryCategorical(
            is_probs_input=self.softmax_output,
        ).create_dist_fn(envs)


class ActorFactoryTransientStorageDecorator(ActorFactory):
    """Wraps an actor factory, storing the most recently created actor instance such that it can be retrieved."""

    def __init__(self, actor_factory: ActorFactory, actor_future: ActorFuture):
        self.actor_factory = actor_factory
        self._actor_future = actor_future

    def __getstate__(self) -> dict:
        d = dict(self.__dict__)
        del d["_actor_future"]
        return d

    def __setstate__(self, state: dict) -> None:
        self.__dict__ = state
        self._actor_future = ActorFuture()

    def _tostring_excludes(self) -> list[str]:
        return [*super()._tostring_excludes(), "_actor_future"]

    def create_module(self, envs: Environments, device: TDevice) -> Actor | nn.Module:
        module = self.actor_factory.create_module(envs, device)
        self._actor_future.actor = module
        return module

    def create_dist_fn(self, envs: Environments) -> TDistFnDiscrOrCont | None:
        return self.actor_factory.create_dist_fn(envs)


class IntermediateModuleFactoryFromActorFactory(IntermediateModuleFactory):
    def __init__(self, actor_factory: ActorFactory):
        self.actor_factory = actor_factory

    def create_intermediate_module(self, envs: Environments, device: TDevice) -> IntermediateModule:
        actor = self.actor_factory.create_module(envs, device)
        assert isinstance(actor, ModuleWithVectorOutput), (
            "Actor factory must produce an actor with known vector output dimension"
        )
        return IntermediateModule(actor, actor.get_output_dim())
