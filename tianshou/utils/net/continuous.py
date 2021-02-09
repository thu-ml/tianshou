import torch
import numpy as np
from torch import nn
from typing import Any, Dict, Tuple, Union, Optional, Sequence

from tianshou.utils.net.common import MLP


SIGMA_MIN = -20
SIGMA_MAX = 2


class Actor(nn.Module):
    """Simple actor network. Will create an actor operated in continuous \
    action space with structure of preprocess_net ---> action_shape.

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param float max_action: the scale for the final action logits. Default to
        1.
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        action_shape: Sequence[int],
        hidden_sizes: Sequence[int] = (),
        max_action: float = 1.0,
        device: Union[str, int, torch.device] = "cpu",
        preprocess_net_output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = np.prod(action_shape)
        input_dim = getattr(preprocess_net, "output_dim",
                            preprocess_net_output_dim)
        self.last = MLP(input_dim, self.output_dim,
                        hidden_sizes, device=self.device)
        self._max = max_action

    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        """Mapping: s -> logits -> action."""
        logits, h = self.preprocess(s, state)
        logits = self._max * torch.tanh(self.last(logits))
        return logits, h


class Critic(nn.Module):
    """Simple critic network. Will create an actor operated in continuous \
    action space with structure of preprocess_net ---> 1(q value).

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        hidden_sizes: Sequence[int] = (),
        device: Union[str, int, torch.device] = "cpu",
        preprocess_net_output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = 1
        input_dim = getattr(preprocess_net, "output_dim",
                            preprocess_net_output_dim)
        self.last = MLP(input_dim, 1, hidden_sizes, device=self.device)

    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        a: Optional[Union[np.ndarray, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> torch.Tensor:
        """Mapping: (s, a) -> logits -> Q(s, a)."""
        s = torch.as_tensor(
            s, device=self.device, dtype=torch.float32  # type: ignore
        ).flatten(1)
        if a is not None:
            a = torch.as_tensor(
                a, device=self.device, dtype=torch.float32  # type: ignore
            ).flatten(1)
            s = torch.cat([s, a], dim=1)
        logits, h = self.preprocess(s)
        logits = self.last(logits)
        return logits


class ActorProb(nn.Module):
    """Simple actor network (output with a Gauss distribution).

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param float max_action: the scale for the final action logits. Default to
        1.
    :param bool unbounded: whether to apply tanh activation on final logits.
        Default to False.
    :param bool conditioned_sigma: True when sigma is calculated from the
        input, False when sigma is an independent parameter. Default to False.
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        action_shape: Sequence[int],
        hidden_sizes: Sequence[int] = (),
        max_action: float = 1.0,
        device: Union[str, int, torch.device] = "cpu",
        unbounded: bool = False,
        conditioned_sigma: bool = False,
        preprocess_net_output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.preprocess = preprocess_net
        self.device = device
        self.output_dim = np.prod(action_shape)
        input_dim = getattr(preprocess_net, "output_dim",
                            preprocess_net_output_dim)
        self.mu = MLP(input_dim, self.output_dim,
                      hidden_sizes, device=self.device)
        self._c_sigma = conditioned_sigma
        if conditioned_sigma:
            self.sigma = MLP(input_dim, self.output_dim,
                             hidden_sizes, device=self.device)
        else:
            self.sigma_param = nn.Parameter(torch.zeros(self.output_dim, 1))
        self._max = max_action
        self._unbounded = unbounded

    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Any]:
        """Mapping: s -> logits -> (mu, sigma)."""
        logits, h = self.preprocess(s, state)
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self._max * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(
                self.sigma(logits), min=SIGMA_MIN, max=SIGMA_MAX
            ).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        return (mu, sigma), state


class RecurrentActorProb(nn.Module):
    """Recurrent version of ActorProb.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        layer_num: int,
        state_shape: Sequence[int],
        action_shape: Sequence[int],
        hidden_layer_size: int = 128,
        max_action: float = 1.0,
        device: Union[str, int, torch.device] = "cpu",
        unbounded: bool = False,
        conditioned_sigma: bool = False,
    ) -> None:
        super().__init__()
        self.device = device
        self.nn = nn.LSTM(
            input_size=np.prod(state_shape),
            hidden_size=hidden_layer_size,
            num_layers=layer_num,
            batch_first=True,
        )
        output_dim = np.prod(action_shape)
        self.mu = nn.Linear(hidden_layer_size, output_dim)
        self._c_sigma = conditioned_sigma
        if conditioned_sigma:
            self.sigma = nn.Linear(hidden_layer_size, output_dim)
        else:
            self.sigma_param = nn.Parameter(torch.zeros(output_dim, 1))
        self._max = max_action
        self._unbounded = unbounded

    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        state: Optional[Dict[str, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """Almost the same as :class:`~tianshou.utils.net.common.Recurrent`."""
        s = torch.as_tensor(
            s, device=self.device, dtype=torch.float32)  # type: ignore
        # s [bsz, len, dim] (training) or [bsz, dim] (evaluation)
        # In short, the tensor's shape in training phase is longer than which
        # in evaluation phase.
        if len(s.shape) == 2:
            s = s.unsqueeze(-2)
        self.nn.flatten_parameters()
        if state is None:
            s, (h, c) = self.nn(s)
        else:
            # we store the stack data in [bsz, len, ...] format
            # but pytorch rnn needs [len, bsz, ...]
            s, (h, c) = self.nn(s, (state["h"].transpose(0, 1).contiguous(),
                                    state["c"].transpose(0, 1).contiguous()))
        logits = s[:, -1]
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self._max * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(
                self.sigma(logits), min=SIGMA_MIN, max=SIGMA_MAX
            ).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        # please ensure the first dim is batch size: [bsz, len, ...]
        return (mu, sigma), {"h": h.transpose(0, 1).detach(),
                             "c": c.transpose(0, 1).detach()}


class RecurrentCritic(nn.Module):
    """Recurrent version of Critic.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        layer_num: int,
        state_shape: Sequence[int],
        action_shape: Sequence[int] = [0],
        device: Union[str, int, torch.device] = "cpu",
        hidden_layer_size: int = 128,
    ) -> None:
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.nn = nn.LSTM(
            input_size=np.prod(state_shape),
            hidden_size=hidden_layer_size,
            num_layers=layer_num,
            batch_first=True,
        )
        self.fc2 = nn.Linear(hidden_layer_size + np.prod(action_shape), 1)

    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        a: Optional[Union[np.ndarray, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> torch.Tensor:
        """Almost the same as :class:`~tianshou.utils.net.common.Recurrent`."""
        s = torch.as_tensor(
            s, device=self.device, dtype=torch.float32)  # type: ignore
        # s [bsz, len, dim] (training) or [bsz, dim] (evaluation)
        # In short, the tensor's shape in training phase is longer than which
        # in evaluation phase.
        assert len(s.shape) == 3
        self.nn.flatten_parameters()
        s, (h, c) = self.nn(s)
        s = s[:, -1]
        if a is not None:
            a = torch.as_tensor(
                a, device=self.device, dtype=torch.float32)  # type: ignore
            s = torch.cat([s, a], dim=1)
        s = self.fc2(s)
        return s
