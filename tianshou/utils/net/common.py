import torch
import numpy as np
from torch import nn
from typing import Any, Dict, List, Tuple, Union, Callable, Optional, Sequence

from tianshou.data import to_torch


def miniblock(
    inp: int,
    oup: int,
    norm_layer: Optional[Callable[[int], nn.modules.Module]] = None,
    activation: Optional[nn.modules.Module] = nn.ReLU,
) -> List[nn.modules.Module]:
    """Construct a miniblock with given input/output-size and norm layer."""
    ret: List[nn.modules.Module] = [nn.Linear(inp, oup)]
    if norm_layer is not None:
        ret += [norm_layer(oup)]
    ret += [activation(inplace=True)]
    return ret


class MLP(nn.Module):
    """Simple MLP backbone.
    Create a MLP of size inp_shape*hidden_layer_size[0]*hidden_layer_size[1]...

    :param hidden_layer_size: shape of MLP passed in as a list, not incluing
        inp_shape.
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
        hidden_layer_size: Union[int, List[int]] = [],
        norm_layer: Optional[Union[nn.modules.Module,
                                   List[nn.modules.Module]]] = [],
        activation: Optional[Union[nn.modules.Module,
                                   List[nn.modules.Module]]] = nn.ReLU,
        inp_shape: Optional[Union[tuple, int]] = None,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        if isinstance(hidden_layer_size, int):
            hidden_layer_size = [hidden_layer_size]
        layer_size = hidden_layer_size.copy()
        if inp_shape is not None:
            layer_size.insert(0, np.prod(inp_shape))
        self.inp_dim = layer_size[0]
        self.out_dim = layer_size[-1]
        self.device = device
        if norm_layer:
            if isinstance(norm_layer, list):
                assert len(norm_layer) == len(layer_size) - 1, (
                    "length of norm_layer should match the "
                    "length of hidden_layer_size.")
            else:
                norm_layer = [norm_layer]*(len(layer_size) - 1)
        if activation:
            if isinstance(activation, list):
                assert len(activation) == len(layer_size) - 1, (
                    "length of activation should match the "
                    "length of hidden_layer_size.")
            else:
                activation = [activation]*(len(layer_size) - 1)
        model = []
        kwargs = {}
        for i in range(len(layer_size) - 1):
            if norm_layer:
                kwargs["norm_layer"] = norm_layer[i]
            if activation:
                kwargs["activation"] = activation[i]
            model += miniblock(
                layer_size[i], layer_size[i+1], **kwargs)
        self.model = nn.Sequential(*model)

    def forward(self, inp):
        inp = to_torch(inp, device=self.device, dtype=torch.float32)
        inp = inp.reshape(inp.size(0), -1)
        return self.model(inp)


class Net(nn.Module):
    """Wrapper of MLP to support more specific DRL usage.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    :param hidden_layer_size: shape of MLP passed in as a list, not incluing
        inp_shape.
    :param bool concat: whether the input shape is concatenated by state_shape
        and action_shape. If it is True, ``action_shape`` is not the output
        shape, but affects the input shape.
    :param bool dueling: whether to use dueling network to calculate Q values
        (for Dueling DQN). If you want to use dueling option, you should pass
        a tuple of two dict stating self-defined arguments as stated in
        class:`~tianshou.utils.net.common.MLP`. Note that input shape will be
        automatically generated based on param hidden_layer_size thus cannot
        be stated in dict or included in hidden_layer_size. Defaults to False.
    :param int num_atoms: in order to expand to the net of distributional RL,
         defaults to 1.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.MLP` for more
        detailed explanation on the usage of activation, norm_layer, etc.

        You can also refer to :class:`~tianshou.utils.net.continuous.Actor`,
        :class:`~tianshou.utils.net.continuous.Critic`, etc, to see how it's
        suggested be used.
    """

    def __init__(
        self,
        hidden_layer_size: List[int],
        state_shape: tuple,
        action_shape: Optional[Union[tuple, int]] = 0,
        device: Union[str, int, torch.device] = "cpu",
        softmax: bool = False,
        concat: bool = False,
        dueling: Optional[Tuple[dict, dict]] = None,
        norm_layer: Optional[Callable[[int], nn.modules.Module]] = None,
        activation: Optional[Callable[[int], nn.modules.Module]] = nn.ReLU,
        num_atoms: int = 1,
    ) -> None:
        assert not isinstance(hidden_layer_size, int), (
            "Class Net no longer support layer_num as its argument from "
            "now on and no longer use 128 as default layer_size. To use "
            "'Net' to build your customized MLP, you can now pass in a list"
            "of int as the shape of MLP to replace original layer_num. "
            "eg. 'hidden_layer_size=128, layer_num=1' is replaced by "
            "'hidden_layer_size=[128, 128]' now.")
        super().__init__()
        self.device = device
        self.dueling = dueling
        self.softmax = softmax
        self.num_atoms = num_atoms
        self.action_num = np.prod(action_shape)
        input_size = np.prod(state_shape)
        if concat:
            input_size += self.action_num
        self.model = MLP(hidden_layer_size, norm_layer, activation,
                         inp_shape=input_size, device=device)
        self.out_dim = self.model.out_dim
        if dueling is None:
            if action_shape and not concat:
                # net serves as an actor
                self.model = nn.Sequential(
                    self.model,
                    nn.Linear(self.model.out_dim, num_atoms * self.action_num))
        else:  # dueling DQN
            q_layer_kwargs, v_layer_kwargs = dueling
            assert not isinstance(q_layer_kwargs, int), (
                "You should pass in a tuple of 2 dictionary to use dueling "
                "option now because Net begins to support any form of MLP. "
                "Check docs of :class:`~tianshou.utils.net.common.Net` "
                "for detailed explanation.")
            assert ("inp_shape" not in q_layer_kwargs or
                    "inp_shape" not in v_layer_kwargs), (
                "inp_shape should not be stated in dueling's dictionary.")
            self.Q = MLP(inp_shape=self.model.out_dim,
                         device=device, **q_layer_kwargs)
            self.V = MLP(inp_shape=self.model.out_dim,
                         device=device, **v_layer_kwargs)
            assert self.V.out_dim == self.Q.out_dim, (
                                        "Q/V output shape mismatch.")
            self.out_dim = self.V.out_dim
            if action_shape and not concat:
                # net serves as an actor
                self.Q = nn.Sequential(
                    self.Q,
                    nn.Linear(self.Q.out_dim, num_atoms * self.action_num))
                self.V = nn.Sequential(
                    self.V,
                    nn.Linear(self.V.out_dim, num_atoms))

    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        """Mapping: s -> flatten (inside MLP)-> logits."""
        logits = self.model(s)
        if self.dueling is not None:  # Dueling DQN
            q, v = self.Q(logits), self.V(logits)
            if self.num_atoms > 1:
                v = v.view(-1, 1, self.num_atoms)
                q = q.view(-1, self.action_num, self.num_atoms)
            logits = q - q.mean(dim=1, keepdim=True) + v
        elif self.num_atoms > 1:
            logits = logits.view(-1, self.action_num, self.num_atoms)
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
