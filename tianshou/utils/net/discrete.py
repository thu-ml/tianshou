import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from typing import Any, Dict, Tuple, Union, Optional, Sequence

from tianshou.data import to_torch


class Actor(nn.Module):
    """Simple actor network with MLP.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        action_shape: Sequence[int],
        hidden_layer_size: int = 128,
        softmax_output: bool = True,
    ) -> None:
        super().__init__()
        self.preprocess = preprocess_net
        self.last = nn.Linear(hidden_layer_size, np.prod(action_shape))
        self.softmax_output = softmax_output

    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        logits, h = self.preprocess(s, state)
        logits = self.last(logits)
        if self.softmax_output:
            logits = F.softmax(logits, dim=-1)
        return logits, h


class Critic(nn.Module):
    """Simple critic network with MLP.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        hidden_layer_size: int = 128,
        last_size: int = 1
    ) -> None:
        super().__init__()
        self.preprocess = preprocess_net
        self.last = nn.Linear(hidden_layer_size, last_size)

    def forward(
        self, s: Union[np.ndarray, torch.Tensor], **kwargs: Any
    ) -> torch.Tensor:
        """Mapping: s -> V(s)."""
        logits, h = self.preprocess(s, state=kwargs.get("state", None))
        logits = self.last(logits)
        return logits


class DQN(nn.Module):
    """Reference: Human-level control through deep reinforcement learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        c: int,
        h: int,
        w: int,
        action_shape: Sequence[int],
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.device = device

        def conv2d_size_out(
            size: int, kernel_size: int = 5, stride: int = 2
        ) -> int:
            return (size - (kernel_size - 1) - 1) // stride + 1

        def conv2d_layers_size_out(
            size: int,
            kernel_size_1: int = 8,
            stride_1: int = 4,
            kernel_size_2: int = 4,
            stride_2: int = 2,
            kernel_size_3: int = 3,
            stride_3: int = 1,
        ) -> int:
            size = conv2d_size_out(size, kernel_size_1, stride_1)
            size = conv2d_size_out(size, kernel_size_2, stride_2)
            size = conv2d_size_out(size, kernel_size_3, stride_3)
            return size

        convw = conv2d_layers_size_out(w)
        convh = conv2d_layers_size_out(h)
        linear_input_size = convw * convh * 64

        self.net = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(linear_input_size, 512),
            nn.Linear(512, np.prod(action_shape)),
        )

    def forward(
        self,
        x: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: x -> Q(x, \*)."""
        if not isinstance(x, torch.Tensor):
            x = to_torch(x, device=self.device, dtype=torch.float32)
        return self.net(x), state
