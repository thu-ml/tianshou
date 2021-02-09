import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from typing import Any, Dict, Tuple, Union, Optional, Sequence

from tianshou.utils.net.common import MLP


class Actor(nn.Module):
    """Simple actor network.

    Will create an actor operated in discrete action space with structure of
    preprocess_net ---> action_shape.

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param bool softmax_output: whether to apply a softmax layer over the last
        layer's output.
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
        softmax_output: bool = True,
        preprocess_net_output_dim: Optional[int] = None,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = np.prod(action_shape)
        input_dim = getattr(preprocess_net, "output_dim",
                            preprocess_net_output_dim)
        self.last = MLP(input_dim, self.output_dim,
                        hidden_sizes, device=self.device)
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
    """Simple critic network. Will create an actor operated in discrete \
    action space with structure of preprocess_net ---> 1(q value).

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param int last_size: the output dimension of Critic network. Default to 1.
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
        last_size: int = 1,
        preprocess_net_output_dim: Optional[int] = None,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = last_size
        input_dim = getattr(preprocess_net, "output_dim",
                            preprocess_net_output_dim)
        self.last = MLP(input_dim, last_size,
                        hidden_sizes, device=self.device)

    def forward(
        self, s: Union[np.ndarray, torch.Tensor], **kwargs: Any
    ) -> torch.Tensor:
        """Mapping: s -> V(s)."""
        logits, _ = self.preprocess(s, state=kwargs.get("state", None))
        return self.last(logits)
