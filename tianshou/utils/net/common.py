import torch
import numpy as np
from torch import nn
from typing import Any, Dict, List, Type, Tuple, Union, Optional, Sequence

ModuleType = Type[nn.Module]


def miniblock(
    input_size: int,
    output_size: int,
    norm_layer: Optional[ModuleType] = None,
    activation: Optional[ModuleType] = None,
) -> List[nn.Module]:
    """Construct a miniblock with given input/output-size and norm layer."""
    layers: List[nn.Module] = [nn.Linear(input_size, output_size)]
    if norm_layer is not None:
        layers += [norm_layer(output_size)]  # type: ignore
    if activation is not None:
        layers += [activation()]
    return layers


class MLP(nn.Module):
    """Simple MLP backbone.

    Create a MLP of size input_shape * hidden_sizes[0] * hidden_sizes[1] * ...

    :param int input_dim: dimension of the input vector.
    :param int output_dim: dimension of the output vector.
    :param hidden_sizes: shape of MLP passed in as a list, not incluing
        input_shape and output_shape.
    :param norm_layer: use which normalization before activation, e.g.,
        ``nn.LayerNorm`` and ``nn.BatchNorm1d``, defaults to no normalization.
        You can also pass in a list of hidden_layer_number-1 normalization
        modules to use different normalization module in different layers.
        Default to empty list (no normalization).
    :param activation: which activation to use after each layer, can be both
        the same actvition for all layers if passed in nn.Module, or different
        activation for different Modules if passed in a list.
        Default to nn.ReLU.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: Sequence[int] = [],
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]]
        = nn.ReLU,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        if norm_layer:
            if isinstance(norm_layer, list):
                assert len(norm_layer) == len(hidden_sizes), (
                    "length of norm_layer should match the "
                    "length of hidden_sizes.")
                norm_layer_list = norm_layer
            else:
                norm_layer_list = [norm_layer] * len(hidden_sizes)
        if activation:
            if isinstance(activation, list):
                assert len(activation) == len(hidden_sizes), (
                    "length of activation should match the "
                    "length of hidden_sizes.")
                activation_list = activation
            else:
                activation_list = [activation] * len(hidden_sizes)
        hidden_sizes = [input_dim] + list(hidden_sizes)
        model = []
        for i in range(len(hidden_sizes) - 1):
            kwargs = {}
            if norm_layer:
                kwargs["norm_layer"] = norm_layer_list[i]
            if activation:
                kwargs["activation"] = activation_list[i]
            model += miniblock(hidden_sizes[i], hidden_sizes[i + 1], **kwargs)
        if output_dim > 0:
            model += [nn.Linear(hidden_sizes[-1], output_dim)]
        self.output_dim = output_dim or hidden_sizes[-1]
        self.model = nn.Sequential(*model)

    def forward(
        self, features: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        features = torch.as_tensor(
            features, device=self.device, dtype=torch.float32)  # type: ignore
        return self.model(features.flatten(1))


class Net(nn.Module):
    """Wrapper of MLP to support more specific DRL usage.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    :param state_shape: int or a list of int of the shape of state.
    :param action_shape: int or a list of int of the shape of action.
    :param hidden_sizes: shape of MLP passed in as a list.
    :param bool concat: whether the input shape is concatenated by state_shape
        and action_shape. If it is True, ``action_shape`` is not the output
        shape, but affects the input shape.
    :param int num_atoms: in order to expand to the net of distributional RL,
         defaults to 1 (no use).
    :param bool use_dueling: whether to use dueling network to calculate Q
        values (for Dueling DQN). Default to False.
    :param dueling_q_hidden_sizes: the MLP hidden sizes for Q value head.
    :param dueling_v_hidden_sizes: the MLP hidden sizes for V value head.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.MLP` for more
        detailed explanation on the usage of activation, norm_layer, etc.

        You can also refer to :class:`~tianshou.utils.net.continuous.Actor`,
        :class:`~tianshou.utils.net.continuous.Critic`, etc, to see how it's
        suggested be used.
    """

    def __init__(
        self,
        state_shape: Union[int, Sequence[int]],
        action_shape: Optional[Union[int, Sequence[int]]] = 0,
        hidden_sizes: List[int] = [],
        device: Union[str, int, torch.device] = "cpu",
        softmax: bool = False,
        concat: bool = False,
        norm_layer: Optional[ModuleType] = None,
        activation: Optional[ModuleType] = nn.ReLU,
        num_atoms: int = 1,
        use_dueling: bool = False,
        dueling_q_hidden_sizes: List[int] = [],
        dueling_v_hidden_sizes: List[int] = [],
    ) -> None:
        super().__init__()
        self.device = device
        self.use_dueling = use_dueling
        self.softmax = softmax
        self.num_atoms = num_atoms
        input_dim = np.prod(state_shape)
        action_dim = np.prod(action_shape) * num_atoms
        if concat:
            input_dim += action_dim
        output_dim = action_dim if not use_dueling and not concat else 0
        self.model = MLP(input_dim, output_dim, hidden_sizes,
                         norm_layer, activation, device)
        self.output_dim = self.model.output_dim
        if use_dueling:  # dueling DQN
            q_output_dim, v_output_dim = 0, 0
            if not concat:
                q_output_dim, v_output_dim = action_dim, num_atoms
            self.Q = MLP(
                self.output_dim, q_output_dim, dueling_q_hidden_sizes,
                norm_layer, activation, device)
            self.V = MLP(
                self.output_dim, v_output_dim, dueling_v_hidden_sizes,
                norm_layer, activation, device)
            self.output_dim = self.Q.output_dim

    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        """Mapping: s -> flatten (inside MLP)-> logits."""
        logits = self.model(s)
        bsz = logits.shape[0]
        if self.use_dueling:  # Dueling DQN
            q, v = self.Q(logits), self.V(logits)
            if self.num_atoms > 1:
                q = q.view(bsz, -1, self.num_atoms)
                v = v.view(bsz, 1, self.num_atoms)
            logits = q - q.mean(dim=1, keepdim=True) + v
        elif self.num_atoms > 1:
            logits = logits.view(bsz, -1, self.num_atoms)
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
        state_shape: Union[int, Sequence[int]],
        action_shape: Union[int, Sequence[int]],
        device: Union[str, int, torch.device] = "cpu",
        hidden_layer_size: int = 128,
    ) -> None:
        super().__init__()
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
        s = torch.as_tensor(
            s, device=self.device, dtype=torch.float32)  # type: ignore
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
