from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar

from sensai.util.string import ToStringMixin

from tianshou.highlevel.env import Environments
from tianshou.highlevel.module.core import TDevice
from tianshou.highlevel.module.intermediate import IntermediateModuleFactory
from tianshou.highlevel.optim import OptimizerFactoryFactory
from tianshou.policy import Algorithm, ICMOffPolicyWrapper
from tianshou.policy.base import OffPolicyAlgorithm, OnPolicyAlgorithm
from tianshou.policy.modelbased.icm import ICMOnPolicyWrapper
from tianshou.utils.net.discrete import IntrinsicCuriosityModule

TAlgorithmOut = TypeVar("TAlgorithmOut", bound=Algorithm)


class AlgorithmWrapperFactory(Generic[TAlgorithmOut], ToStringMixin, ABC):
    @abstractmethod
    def create_wrapped_algorithm(
        self,
        policy: Algorithm,
        envs: Environments,
        optim_factory: OptimizerFactoryFactory,
        device: TDevice,
    ) -> TAlgorithmOut:
        pass


class AlgorithmWrapperFactoryIntrinsicCuriosity(
    AlgorithmWrapperFactory[ICMOffPolicyWrapper | ICMOnPolicyWrapper],
):
    def __init__(
        self,
        *,
        feature_net_factory: IntermediateModuleFactory,
        hidden_sizes: Sequence[int],
        lr: float,
        lr_scale: float,
        reward_scale: float,
        forward_loss_weight: float,
        optim: OptimizerFactoryFactory | None = None,
    ):
        self.feature_net_factory = feature_net_factory
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.lr_scale = lr_scale
        self.reward_scale = reward_scale
        self.forward_loss_weight = forward_loss_weight
        self.optim_factory = optim

    def create_wrapped_algorithm(
        self,
        algorithm: Algorithm,
        envs: Environments,
        optim_factory_default: OptimizerFactoryFactory,
        device: TDevice,
    ) -> ICMOffPolicyWrapper | ICMOnPolicyWrapper:
        feature_net = self.feature_net_factory.create_intermediate_module(envs, device)
        action_dim = envs.get_action_shape()
        if not isinstance(action_dim, int):
            raise ValueError(f"Environment action shape must be an integer, got {action_dim}")
        feature_dim = feature_net.output_dim
        icm_net = IntrinsicCuriosityModule(
            feature_net=feature_net.module,
            feature_dim=feature_dim,
            action_dim=action_dim,
            hidden_sizes=self.hidden_sizes,
        )
        optim_factory = self.optim_factory or optim_factory_default
        icm_optim = optim_factory.create_optimizer_factory(lr=self.lr)
        if isinstance(algorithm, OffPolicyAlgorithm):
            return ICMOffPolicyWrapper(
                wrapped_algorithm=algorithm,
                model=icm_net,
                optim=icm_optim,
                lr_scale=self.lr_scale,
                reward_scale=self.reward_scale,
                forward_loss_weight=self.forward_loss_weight,
            ).to(device)
        elif isinstance(algorithm, OnPolicyAlgorithm):
            return ICMOnPolicyWrapper(
                wrapped_algorithm=algorithm,
                model=icm_net,
                optim=icm_optim,
                lr_scale=self.lr_scale,
                reward_scale=self.reward_scale,
                forward_loss_weight=self.forward_loss_weight,
            ).to(device)
        else:
            raise ValueError(f"{algorithm} is not supported by ICM")
