import torch
import numpy as np
from torch import nn
from typing import Any, Dict, List, Tuple, Union, Callable, Optional, Sequence

from tianshou.data import to_torch


def miniblock(
    inp: int,
    oup: int,
    norm_layer: Optional[Callable[[int], nn.modules.Module]],
) -> List[nn.modules.Module]:
    """Construct a miniblock with given input/output-size and norm layer."""
    ret: List[nn.modules.Module] = [nn.Linear(inp, oup)]
    if norm_layer is not None:
        ret += [norm_layer(oup)]
    ret += [nn.ReLU(inplace=True)]
    return ret


class Net(nn.Module):
    """Simple MLP backbone.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    :param bool concat: whether the input shape is concatenated by state_shape
        and action_shape. If it is True, ``action_shape`` is not the output
        shape, but affects the input shape.
    :param bool dueling: whether to use dueling network to calculate Q values
        (for Dueling DQN), defaults to False.
    :param norm_layer: use which normalization before ReLU, e.g.,
        ``nn.LayerNorm`` and ``nn.BatchNorm1d``, defaults to None.
    """

    def __init__(
        self,
        layer_num: int,
        state_shape: tuple,
        action_shape: Optional[Union[tuple, int]] = 0,
        device: Union[str, int, torch.device] = "cpu",
        softmax: bool = False,
        concat: bool = False,
        hidden_layer_size: int = 128,
        dueling: Optional[Tuple[int, int]] = None,
        norm_layer: Optional[Callable[[int], nn.modules.Module]] = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.dueling = dueling
        self.softmax = softmax
        input_size = np.prod(state_shape)
        if concat:
            input_size += np.prod(action_shape)

        model = miniblock(input_size, hidden_layer_size, norm_layer)

        for i in range(layer_num):
            model += miniblock(
                hidden_layer_size, hidden_layer_size, norm_layer)

        if dueling is None:
            if action_shape and not concat:
                model += [nn.Linear(hidden_layer_size, np.prod(action_shape))]
        else:  # dueling DQN
            q_layer_num, v_layer_num = dueling
            Q, V = [], []

            for i in range(q_layer_num):
                Q += miniblock(
                    hidden_layer_size, hidden_layer_size, norm_layer)
            for i in range(v_layer_num):
                V += miniblock(
                    hidden_layer_size, hidden_layer_size, norm_layer)

            if action_shape and not concat:
                Q += [nn.Linear(hidden_layer_size, np.prod(action_shape))]
                V += [nn.Linear(hidden_layer_size, 1)]

            self.Q = nn.Sequential(*Q)
            self.V = nn.Sequential(*V)
        self.model = nn.Sequential(*model)

    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        """Mapping: s -> flatten -> logits."""
        s = to_torch(s, device=self.device, dtype=torch.float32)
        s = s.reshape(s.size(0), -1)
        logits = self.model(s)
        if self.dueling is not None:  # Dueling DQN
            q, v = self.Q(logits), self.V(logits)
            logits = q - q.mean(dim=1, keepdim=True) + v
        if self.softmax:
            logits = torch.softmax(logits, dim=-1)
        return logits, state


class Recurrent(nn.Module):
    """Simple Recurrent network based on LSTM.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        layer_num: int,
        state_shape: Sequence[int],
        action_shape: Sequence[int],
        device: Union[str, int, torch.device] = "cpu",
        hidden_layer_size: int = 128,
    ) -> None:
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.nn = nn.LSTM(
            input_size=hidden_layer_size,
            hidden_size=hidden_layer_size,
            num_layers=layer_num,
            batch_first=True,
        )
        self.fc1 = nn.Linear(np.prod(state_shape), hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, np.prod(action_shape))

    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        state: Optional[Dict[str, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Mapping: s -> flatten -> logits.

        In the evaluation mode, s should be with shape ``[bsz, dim]``; in the
        training mode, s should be with shape ``[bsz, len, dim]``. See the code
        and comment for more detail.
        """
        s = to_torch(s, device=self.device, dtype=torch.float32)
        # s [bsz, len, dim] (training) or [bsz, dim] (evaluation)
        # In short, the tensor's shape in training phase is longer than which
        # in evaluation phase.
        if len(s.shape) == 2:
            s = s.unsqueeze(-2)
        s = self.fc1(s)
        self.nn.flatten_parameters()
        if state is None:
            s, (h, c) = self.nn(s)
        else:
            # we store the stack data in [bsz, len, ...] format
            # but pytorch rnn needs [len, bsz, ...]
            s, (h, c) = self.nn(s, (state["h"].transpose(0, 1).contiguous(),
                                    state["c"].transpose(0, 1).contiguous()))
        s = self.fc2(s[:, -1])
        # please ensure the first dim is batch size: [bsz, len, ...]
        return s, {"h": h.transpose(0, 1).detach(),
                   "c": c.transpose(0, 1).detach()}
