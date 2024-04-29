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
from tianshou.utils.net.common import NetBase
from tianshou.utils.net.discrete import Actor, NoisyLinear


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ScaledObsInputModule(torch.nn.Module):
    def __init__(self, module: NetBase, denom: float = 255.0) -> None:
        super().__init__()
        self.module = module
        self.denom = denom
        # This is required such that the value can be retrieved by downstream modules (see usages of get_output_dim)
        self.output_dim = module.output_dim

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
    return ScaledObsInputModule(module, denom=denom)


class DQN(NetBase[Any]):
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
        device: str | int | torch.device = "cpu",
        features_only: bool = False,
        output_dim_added_layer: int | None = None,
        layer_init: Callable[[nn.Module], nn.Module] = lambda x: x,
    ) -> None:
        # TODO: Add docstring
        if not features_only and output_dim_added_layer is not None:
            raise ValueError(
                "Should not provide explicit output dimension using `output_dim_added_layer` when `features_only` is true.",
            )
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            layer_init(nn.Conv2d(c, 32, kernel_size=8, stride=4)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            base_cnn_output_dim = int(np.prod(self.net(torch.zeros(1, c, h, w)).shape[1:]))
        if not features_only:
            action_dim = int(np.prod(action_shape))
            self.net = nn.Sequential(
                self.net,
                layer_init(nn.Linear(base_cnn_output_dim, 512)),
                nn.ReLU(inplace=True),
                layer_init(nn.Linear(512, action_dim)),
            )
            self.output_dim = action_dim
        elif output_dim_added_layer is not None:
            self.net = nn.Sequential(
                self.net,
                layer_init(nn.Linear(base_cnn_output_dim, output_dim_added_layer)),
                nn.ReLU(inplace=True),
            )
            self.output_dim = output_dim_added_layer
        else:
            self.output_dim = base_cnn_output_dim

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: Any | None = None,
        info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        return self.net(obs), state


class C51(DQN):
    """Reference: A distributional perspective on reinforcement learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        c: int,
        h: int,
        w: int,
        action_shape: Sequence[int],
        num_atoms: int = 51,
        device: str | int | torch.device = "cpu",
    ) -> None:
        self.action_num = int(np.prod(action_shape))
        super().__init__(c, h, w, [self.action_num * num_atoms], device)
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


class Rainbow(DQN):
    """Reference: Rainbow: Combining Improvements in Deep Reinforcement Learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        c: int,
        h: int,
        w: int,
        action_shape: Sequence[int],
        num_atoms: int = 51,
        noisy_std: float = 0.5,
        device: str | int | torch.device = "cpu",
        is_dueling: bool = True,
        is_noisy: bool = True,
    ) -> None:
        super().__init__(c, h, w, action_shape, device, features_only=True)
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


class QRDQN(DQN):
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
        device: str | int | torch.device = "cpu",
    ) -> None:
        self.action_num = int(np.prod(action_shape))
        super().__init__(c, h, w, [self.action_num * num_quantiles], device)
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
    def __init__(
        self,
        scale_obs: bool = True,
        features_only: bool = False,
        output_dim_added_layer: int | None = None,
    ) -> None:
        self.output_dim_added_layer = output_dim_added_layer
        self.scale_obs = scale_obs
        self.features_only = features_only

    def create_module(self, envs: Environments, device: TDevice) -> Actor:
        c, h, w = envs.get_observation_shape()  # type: ignore  # only right shape is a sequence of length 3
        action_shape = envs.get_action_shape()
        if isinstance(action_shape, np.int64):
            action_shape = int(action_shape)
        net: DQN | ScaledObsInputModule
        net = DQN(
            c=c,
            h=h,
            w=w,
            action_shape=action_shape,
            device=device,
            features_only=self.features_only,
            output_dim_added_layer=self.output_dim_added_layer,
            layer_init=layer_init,
        )
        if self.scale_obs:
            net = scale_obs(net)
        return Actor(net, envs.get_action_shape(), device=device, softmax_output=False).to(device)


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
        dqn = DQN(
            c=c,
            h=h,
            w=w,
            action_shape=action_shape,
            device=device,
            features_only=self.features_only,
        ).to(device)
        module = dqn.net if self.net_only else dqn
        return IntermediateModule(module, dqn.output_dim)


class IntermediateModuleFactoryAtariDQNFeatures(IntermediateModuleFactoryAtariDQN):
    def __init__(self) -> None:
        super().__init__(features_only=True, net_only=True)
