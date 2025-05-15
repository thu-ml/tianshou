from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from tianshou.highlevel.env import Environments
from tianshou.highlevel.module.actor import ActorFactory
from tianshou.highlevel.module.core import (
    TDevice,
)
from tianshou.highlevel.module.intermediate import (
    IntermediateModule,
    IntermediateModuleFactory,
)
from tianshou.highlevel.params.dist_fn import DistributionFunctionFactoryCategorical
from tianshou.algorithm.modelfree.pg import TDistFnDiscrOrCont
from tianshou.utils.net.common import NetBase
from tianshou.utils.net.discrete import DiscreteActor, NoisyLinear
from tianshou.utils.torch_utils import torch_device


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """TODO."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ScaledObsInputModule(NetBase):
    def __init__(self, module: NetBase, denom: float = 255.0) -> None:
        super().__init__(module.get_output_dim())
        self.module = module
        self.denom = denom

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: Any | None = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, Any]:
        if info is None:
            info = {}
        return self.module.forward(obs / self.denom, state, info)


def scale_obs(module: NetBase, denom: float = 255.0) -> ScaledObsInputModule:
    """TODO."""
    return ScaledObsInputModule(module, denom=denom)


class DQNet(NetBase[Any]):
    """Reference: Human-level control through deep reinforcement learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        c: int,
        h: int,
        w: int,
        action_shape: Sequence[int] | int,
        features_only: bool = False,
        output_dim_added_layer: int | None = None,
        layer_init: Callable[[nn.Module], nn.Module] = lambda x: x,
    ) -> None:
        # TODO: Add docstring
        if not features_only and output_dim_added_layer is not None:
            raise ValueError(
                "Should not provide explicit output dimension using `output_dim_added_layer` when `features_only` is true.",
            )
        net = nn.Sequential(
            layer_init(nn.Conv2d(c, 32, kernel_size=8, stride=4)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            base_cnn_output_dim = int(np.prod(net(torch.zeros(1, c, h, w)).shape[1:]))
        if not features_only:
            action_dim = int(np.prod(action_shape))
            net = nn.Sequential(
                net,
                layer_init(nn.Linear(base_cnn_output_dim, 512)),
                nn.ReLU(inplace=True),
                layer_init(nn.Linear(512, action_dim)),
            )
            output_dim = action_dim
        elif output_dim_added_layer is not None:
            net = nn.Sequential(
                net,
                layer_init(nn.Linear(base_cnn_output_dim, output_dim_added_layer)),
                nn.ReLU(inplace=True),
            )
            output_dim = output_dim_added_layer
        else:
            output_dim = base_cnn_output_dim
        super().__init__(output_dim)
        self.net = net

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: Any | None = None,
        info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        device = torch_device(self)
        obs = torch.as_tensor(obs, device=device, dtype=torch.float32)
        return self.net(obs), state


class C51Net(DQNet):
    """Reference: A distributional perspective on reinforcement learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        *,
        c: int,
        h: int,
        w: int,
        action_shape: Sequence[int],
        num_atoms: int = 51,
    ) -> None:
        self.action_num = int(np.prod(action_shape))
        super().__init__(c=c, h=h, w=w, action_shape=[self.action_num * num_atoms])
        self.num_atoms = num_atoms

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: Any | None = None,
        info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        r"""Mapping: x -> Z(x, \*)."""
        obs, state = super().forward(obs)
        obs = obs.view(-1, self.num_atoms).softmax(dim=-1)
        obs = obs.view(-1, self.action_num, self.num_atoms)
        return obs, state


class Rainbow(DQNet):
    """Reference: Rainbow: Combining Improvements in Deep Reinforcement Learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        *,
        c: int,
        h: int,
        w: int,
        action_shape: Sequence[int],
        num_atoms: int = 51,
        noisy_std: float = 0.5,
        is_dueling: bool = True,
        is_noisy: bool = True,
    ) -> None:
        super().__init__(c=c, h=h, w=w, action_shape=action_shape, features_only=True)
        self.action_num = int(np.prod(action_shape))
        self.num_atoms = num_atoms

        def linear(x: int, y: int) -> NoisyLinear | nn.Linear:
            if is_noisy:
                return NoisyLinear(x, y, noisy_std)
            return nn.Linear(x, y)

        self.Q = nn.Sequential(
            linear(self.output_dim, 512),
            nn.ReLU(inplace=True),
            linear(512, self.action_num * self.num_atoms),
        )
        self._is_dueling = is_dueling
        if self._is_dueling:
            self.V = nn.Sequential(
                linear(self.output_dim, 512),
                nn.ReLU(inplace=True),
                linear(512, self.num_atoms),
            )
        self.output_dim = self.action_num * self.num_atoms

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: Any | None = None,
        info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        r"""Mapping: x -> Z(x, \*)."""
        obs, state = super().forward(obs)
        q = self.Q(obs)
        q = q.view(-1, self.action_num, self.num_atoms)
        if self._is_dueling:
            v = self.V(obs)
            v = v.view(-1, 1, self.num_atoms)
            logits = q - q.mean(dim=1, keepdim=True) + v
        else:
            logits = q
        probs = logits.softmax(dim=2)
        return probs, state


class QRDQNet(DQNet):
    """Reference: Distributional Reinforcement Learning with Quantile Regression.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        *,
        c: int,
        h: int,
        w: int,
        action_shape: Sequence[int] | int,
        num_quantiles: int = 200,
    ) -> None:
        self.action_num = int(np.prod(action_shape))
        super().__init__(c=c, h=h, w=w, action_shape=[self.action_num * num_quantiles])
        self.num_quantiles = num_quantiles

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: Any | None = None,
        info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        r"""Mapping: x -> Z(x, \*)."""
        obs, state = super().forward(obs)
        obs = obs.view(-1, self.action_num, self.num_quantiles)
        return obs, state


class ActorFactoryAtariDQN(ActorFactory):
    USE_SOFTMAX_OUTPUT = False

    def __init__(
        self,
        scale_obs: bool = True,
        features_only: bool = False,
        output_dim_added_layer: int | None = None,
    ) -> None:
        self.output_dim_added_layer = output_dim_added_layer
        self.scale_obs = scale_obs
        self.features_only = features_only

    def create_module(self, envs: Environments, device: TDevice) -> DiscreteActor:
        c, h, w = envs.get_observation_shape()  # type: ignore  # only right shape is a sequence of length 3
        action_shape = envs.get_action_shape()
        if isinstance(action_shape, np.int64):
            action_shape = int(action_shape)
        net: DQNet | ScaledObsInputModule
        net = DQNet(
            c=c,
            h=h,
            w=w,
            action_shape=action_shape,
            features_only=self.features_only,
            output_dim_added_layer=self.output_dim_added_layer,
            layer_init=layer_init,
        )
        if self.scale_obs:
            net = scale_obs(net)
        return DiscreteActor(
            preprocess_net=net,
            action_shape=envs.get_action_shape(),
            softmax_output=self.USE_SOFTMAX_OUTPUT,
        ).to(device)

    def create_dist_fn(self, envs: Environments) -> TDistFnDiscrOrCont | None:
        return DistributionFunctionFactoryCategorical(
            is_probs_input=self.USE_SOFTMAX_OUTPUT,
        ).create_dist_fn(envs)


class IntermediateModuleFactoryAtariDQN(IntermediateModuleFactory):
    def __init__(self, features_only: bool = False, net_only: bool = False) -> None:
        self.features_only = features_only
        self.net_only = net_only

    def create_intermediate_module(self, envs: Environments, device: TDevice) -> IntermediateModule:
        obs_shape = envs.get_observation_shape()
        if isinstance(obs_shape, int):
            obs_shape = [obs_shape]
        assert len(obs_shape) == 3
        c, h, w = obs_shape
        action_shape = envs.get_action_shape()
        if isinstance(action_shape, np.int64):
            action_shape = int(action_shape)
        dqn = DQNet(
            c=c,
            h=h,
            w=w,
            action_shape=action_shape,
            features_only=self.features_only,
        ).to(device)
        module = dqn.net if self.net_only else dqn
        return IntermediateModule(module, dqn.output_dim)


class IntermediateModuleFactoryAtariDQNFeatures(IntermediateModuleFactoryAtariDQN):
    def __init__(self) -> None:
        super().__init__(features_only=True, net_only=True)
