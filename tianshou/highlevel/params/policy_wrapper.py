from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar

from tianshou.highlevel.env import Environments
from tianshou.highlevel.module.core import TDevice
from tianshou.highlevel.module.intermediate import IntermediateModuleFactory
from tianshou.highlevel.optim import OptimizerFactory
from tianshou.policy import BasePolicy, ICMPolicy
from tianshou.utils.net.discrete import IntrinsicCuriosityModule
from tianshou.utils.string import ToStringMixin

TPolicyOut = TypeVar("TPolicyOut", bound=BasePolicy)


class PolicyWrapperFactory(Generic[TPolicyOut], ToStringMixin, ABC):
    @abstractmethod
    def create_wrapped_policy(
        self,
        policy: BasePolicy,
        envs: Environments,
        optim_factory: OptimizerFactory,
        device: TDevice,
    ) -> TPolicyOut:
        pass


class PolicyWrapperFactoryIntrinsicCuriosity(
    PolicyWrapperFactory[ICMPolicy],
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
    ):
        self.feature_net_factory = feature_net_factory
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.lr_scale = lr_scale
        self.reward_scale = reward_scale
        self.forward_loss_weight = forward_loss_weight

    def create_wrapped_policy(
        self,
        policy: BasePolicy,
        envs: Environments,
        optim_factory: OptimizerFactory,
        device: TDevice,
    ) -> ICMPolicy:
        feature_net = self.feature_net_factory.create_intermediate_module(envs, device)
        action_dim = envs.get_action_shape()
        if not isinstance(action_dim, int):
            raise ValueError(f"Environment action shape must be an integer, got {action_dim}")
        feature_dim = feature_net.output_dim
        icm_net = IntrinsicCuriosityModule(
            feature_net.module,
            feature_dim,
            action_dim,
            hidden_sizes=self.hidden_sizes,
            device=device,
        )
        icm_optim = optim_factory.create_optimizer(icm_net, lr=self.lr)
        return ICMPolicy(
            policy=policy,
            model=icm_net,
            optim=icm_optim,
            action_space=envs.get_action_space(),
            lr_scale=self.lr_scale,
            reward_scale=self.reward_scale,
            forward_loss_weight=self.forward_loss_weight,
        ).to(device)
