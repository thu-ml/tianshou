from copy import deepcopy
from dataclasses import dataclass
from typing import Self

import torch


def polyak_parameter_update(tgt: torch.nn.Module, src: torch.nn.Module, tau: float) -> None:
    """Softly updates the parameters of a target network `tgt` with the parameters of a source network `src`
    using Polyak averaging: `tau * src + (1 - tau) * tgt`.

    :param tgt: the target network that receives the parameter update
    :param src: the source network whose parameters are used for the update
    :param tau: the fraction with which to use the source network's parameters, the inverse `1-tau` being
        the fraction with which to retain the target network's parameters.
    """
    for tgt_param, src_param in zip(tgt.parameters(), src.parameters(), strict=True):
        tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)


class EvalModeModuleWrapper(torch.nn.Module):
    """
    A wrapper around a torch.nn.Module that forces the module to eval mode.

    The wrapped module supports only the forward method, attribute access is not supported.
    **NOTE**: It is *not* recommended to support attribute/method access beyond this via `__getattr__`,
    because torch.nn.Module already heavily relies on `__getattr__` to provides its own attribute access.
    Overriding it naively will cause problems!
    But it's also not necessary for our use cases; forward is enough.
    """

    def __init__(self, m: torch.nn.Module):
        super().__init__()
        m.eval()
        self.module = m

    def forward(self, *args, **kwargs):  # type: ignore
        self.module.eval()
        return self.module(*args, **kwargs)

    def train(self, mode: bool = True) -> Self:
        super().train(mode=mode)
        self.module.eval()  # force eval mode
        return self


@dataclass
class LaggedNetworkPair:
    target: torch.nn.Module
    source: torch.nn.Module


class LaggedNetworkCollection:
    def __init__(self) -> None:
        self._lagged_network_pairs: list[LaggedNetworkPair] = []

    def add_lagged_network(self, source: torch.nn.Module) -> EvalModeModuleWrapper:
        """
        Adds a lagged network to the collection, returning the target network, which
        is forced to eval mode. The target network is a copy of the source network,
        which, however, supports only the forward method (hence the type torch.nn.Module);
        attribute access is not supported.

        :param source: the source network whose parameters are to be copied to the target network
        :return: the target network, which supports only the forward method and is forced to eval mode
        """
        target = deepcopy(source)
        self._lagged_network_pairs.append(LaggedNetworkPair(target, source))
        return EvalModeModuleWrapper(target)

    def polyak_parameter_update(self, tau: float) -> None:
        """Softly updates the parameters of each target network `tgt` with the parameters of a source network `src`
        using Polyak averaging: `tau * src + (1 - tau) * tgt`.

        :param tau: the fraction with which to use the source network's parameters, the inverse `1-tau` being
            the fraction with which to retain the target network's parameters.
        """
        for pair in self._lagged_network_pairs:
            polyak_parameter_update(pair.target, pair.source, tau)

    def full_parameter_update(self) -> None:
        """Fully updates the target networks with the source networks' parameters (exact copy)."""
        for pair in self._lagged_network_pairs:
            for tgt_param, src_param in zip(
                pair.target.parameters(), pair.source.parameters(), strict=True
            ):
                tgt_param.data.copy_(src_param.data)
