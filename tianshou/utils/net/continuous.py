import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from sensai.util.pickle import setstate
from torch import nn

from tianshou.utils.net.common import (
    MLP,
    BaseActor,
    Net,
    TActionShape,
    TLinearLayer,
    get_output_dim,
)

SIGMA_MIN = -20
SIGMA_MAX = 2


class Actor(BaseActor):
    """Simple actor network that directly outputs actions for continuous action space.
    Used primarily in DDPG and its variants. For probabilistic policies, see :class:`~ActorProb`.

    It will create an actor operated in continuous action space with structure of preprocess_net ---> action_shape.

    :param preprocess_net: a self-defined preprocess_net, see usage.
        Typically, an instance of :class:`~tianshou.utils.net.common.Net`.
    :param action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net.
    :param max_action: the scale for the final action.
    :param preprocess_net_output_dim: the output dimension of
        `preprocess_net`. Only used when `preprocess_net` does not have the attribute `output_dim`.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        preprocess_net: nn.Module | Net,
        action_shape: TActionShape,
        hidden_sizes: Sequence[int] = (),
        max_action: float = 1.0,
        device: str | int | torch.device = "cpu",
        preprocess_net_output_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = int(np.prod(action_shape))
        input_dim = get_output_dim(preprocess_net, preprocess_net_output_dim)
        self.last = MLP(
            input_dim,
            self.output_dim,
            hidden_sizes,
            device=self.device,
        )
        self.max_action = max_action

    def get_preprocess_net(self) -> nn.Module:
        return self.preprocess

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: Any = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, Any]:
        """Mapping: s_B -> action_values_BA, hidden_state_BH | None.

        Returns a tensor representing the actions directly, i.e, of shape
        `(n_actions, )`, and a hidden state (which may be None).
        The hidden state is only not None if a recurrent net is used as part of the
        learning algorithm (support for RNNs is currently experimental).
        """
        action_BA, hidden_BH = self.preprocess(obs, state)
        action_BA = self.max_action * torch.tanh(self.last(action_BA))
        return action_BA, hidden_BH


class CriticBase(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        act: np.ndarray | torch.Tensor | None = None,
        info: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Mapping: (s_B, a_B) -> Q(s, a)_B."""


class Critic(CriticBase):
    """Simple critic network.

    It will create an actor operated in continuous action space with structure of preprocess_net ---> 1(q value).

    :param preprocess_net: a self-defined preprocess_net, see usage.
        Typically, an instance of :class:`~tianshou.utils.net.common.Net`.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net.
    :param preprocess_net_output_dim: the output dimension of
        `preprocess_net`. Only used when `preprocess_net` does not have the attribute `output_dim`.
    :param linear_layer: use this module as linear layer.
    :param flatten_input: whether to flatten input data for the last layer.
    :param apply_preprocess_net_to_obs_only: whether to apply `preprocess_net` to the observations only (before
        concatenating with the action) - and without the observations being modified in any way beforehand.
        This allows the actor's preprocessing network to be reused for the critic.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        preprocess_net: nn.Module | Net,
        hidden_sizes: Sequence[int] = (),
        device: str | int | torch.device = "cpu",
        preprocess_net_output_dim: int | None = None,
        linear_layer: TLinearLayer = nn.Linear,
        flatten_input: bool = True,
        apply_preprocess_net_to_obs_only: bool = False,
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = 1
        self.apply_preprocess_net_to_obs_only = apply_preprocess_net_to_obs_only
        input_dim = get_output_dim(preprocess_net, preprocess_net_output_dim)
        self.last = MLP(
            input_dim,
            1,
            hidden_sizes,
            device=self.device,
            linear_layer=linear_layer,
            flatten_input=flatten_input,
        )

    def __setstate__(self, state: dict) -> None:
        setstate(
            Critic,
            self,
            state,
            new_default_properties={"apply_preprocess_net_to_obs_only": False},
        )

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        act: np.ndarray | torch.Tensor | None = None,
        info: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Mapping: (s_B, a_B) -> Q(s, a)_B."""
        obs = torch.as_tensor(
            obs,
            device=self.device,
            dtype=torch.float32,
        )
        if self.apply_preprocess_net_to_obs_only:
            obs, _ = self.preprocess(obs)
        obs = obs.flatten(1)
        if act is not None:
            act = torch.as_tensor(
                act,
                device=self.device,
                dtype=torch.float32,
            ).flatten(1)
            obs = torch.cat([obs, act], dim=1)
        if not self.apply_preprocess_net_to_obs_only:
            obs, _ = self.preprocess(obs)
        return self.last(obs)


class ActorProb(BaseActor):
    """Simple actor network that outputs `mu` and `sigma` to be used as input for a `dist_fn` (typically, a Gaussian).

    Used primarily in SAC, PPO and variants thereof. For deterministic policies, see :class:`~Actor`.

    :param preprocess_net: a self-defined preprocess_net, see usage.
        Typically, an instance of :class:`~tianshou.utils.net.common.Net`.
    :param action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net.
    :param max_action: the scale for the final action logits.
    :param unbounded: whether to apply tanh activation on final logits.
    :param conditioned_sigma: True when sigma is calculated from the
        input, False when sigma is an independent parameter.
    :param preprocess_net_output_dim: the output dimension of
        `preprocess_net`. Only used when `preprocess_net` does not have the attribute `output_dim`.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    # TODO: force kwargs, adjust downstream code
    def __init__(
        self,
        preprocess_net: nn.Module | Net,
        action_shape: TActionShape,
        hidden_sizes: Sequence[int] = (),
        max_action: float = 1.0,
        device: str | int | torch.device = "cpu",
        unbounded: bool = False,
        conditioned_sigma: bool = False,
        preprocess_net_output_dim: int | None = None,
    ) -> None:
        super().__init__()
        if unbounded and not np.isclose(max_action, 1.0):
            warnings.warn("Note that max_action input will be discarded when unbounded is True.")
            max_action = 1.0
        self.preprocess = preprocess_net
        self.device = device
        self.output_dim = int(np.prod(action_shape))
        input_dim = get_output_dim(preprocess_net, preprocess_net_output_dim)
        self.mu = MLP(input_dim, self.output_dim, hidden_sizes, device=self.device)
        self._c_sigma = conditioned_sigma
        if conditioned_sigma:
            self.sigma = MLP(
                input_dim,
                self.output_dim,
                hidden_sizes,
                device=self.device,
            )
        else:
            self.sigma_param = nn.Parameter(torch.zeros(self.output_dim, 1))
        self.max_action = max_action
        self._unbounded = unbounded

    def get_preprocess_net(self) -> nn.Module:
        return self.preprocess

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: Any = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], Any]:
        """Mapping: obs -> logits -> (mu, sigma)."""
        if info is None:
            info = {}
        logits, hidden = self.preprocess(obs, state)
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self.max_action * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(self.sigma(logits), min=SIGMA_MIN, max=SIGMA_MAX).exp()
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
        device: str | int | torch.device = "cpu",
        unbounded: bool = False,
        conditioned_sigma: bool = False,
    ) -> None:
        super().__init__()
        if unbounded and not np.isclose(max_action, 1.0):
            warnings.warn("Note that max_action input will be discarded when unbounded is True.")
            max_action = 1.0
        self.device = device
        self.nn = nn.LSTM(
            input_size=int(np.prod(state_shape)),
            hidden_size=hidden_layer_size,
            num_layers=layer_num,
            batch_first=True,
        )
        output_dim = int(np.prod(action_shape))
        self.mu = nn.Linear(hidden_layer_size, output_dim)
        self._c_sigma = conditioned_sigma
        if conditioned_sigma:
            self.sigma = nn.Linear(hidden_layer_size, output_dim)
        else:
            self.sigma_param = nn.Parameter(torch.zeros(output_dim, 1))
        self.max_action = max_action
        self._unbounded = unbounded

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: dict[str, torch.Tensor] | None = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]:
        """Almost the same as :class:`~tianshou.utils.net.common.Recurrent`."""
        if info is None:
            info = {}
        obs = torch.as_tensor(
            obs,
            device=self.device,
            dtype=torch.float32,
        )
        # obs [bsz, len, dim] (training) or [bsz, dim] (evaluation)
        # In short, the tensor's shape in training phase is longer than which
        # in evaluation phase.
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(-2)
        self.nn.flatten_parameters()
        if state is None:
            obs, (hidden, cell) = self.nn(obs)
        else:
            # we store the stack data in [bsz, len, ...] format
            # but pytorch rnn needs [len, bsz, ...]
            obs, (hidden, cell) = self.nn(
                obs,
                (
                    state["hidden"].transpose(0, 1).contiguous(),
                    state["cell"].transpose(0, 1).contiguous(),
                ),
            )
        logits = obs[:, -1]
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self.max_action * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(self.sigma(logits), min=SIGMA_MIN, max=SIGMA_MAX).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        # please ensure the first dim is batch size: [bsz, len, ...]
        return (mu, sigma), {
            "hidden": hidden.transpose(0, 1).detach(),
            "cell": cell.transpose(0, 1).detach(),
        }


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
        device: str | int | torch.device = "cpu",
        hidden_layer_size: int = 128,
    ) -> None:
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.nn = nn.LSTM(
            input_size=int(np.prod(state_shape)),
            hidden_size=hidden_layer_size,
            num_layers=layer_num,
            batch_first=True,
        )
        self.fc2 = nn.Linear(hidden_layer_size + int(np.prod(action_shape)), 1)

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        act: np.ndarray | torch.Tensor | None = None,
        info: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Almost the same as :class:`~tianshou.utils.net.common.Recurrent`."""
        if info is None:
            info = {}
        obs = torch.as_tensor(
            obs,
            device=self.device,
            dtype=torch.float32,
        )
        # obs [bsz, len, dim] (training) or [bsz, dim] (evaluation)
        # In short, the tensor's shape in training phase is longer than which
        # in evaluation phase.
        assert len(obs.shape) == 3
        self.nn.flatten_parameters()
        obs, (hidden, cell) = self.nn(obs)
        obs = obs[:, -1]
        if act is not None:
            act = torch.as_tensor(
                act,
                device=self.device,
                dtype=torch.float32,
            )
            obs = torch.cat([obs, act], dim=1)
        return self.fc2(obs)


class Perturbation(nn.Module):
    """Implementation of perturbation network in BCQ algorithm.

    Given a state and action, it can generate perturbed action.

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param max_action: the maximum value of each dimension of action.
    :param device: which device to create this model on.
    :param phi: max perturbation parameter for BCQ.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        You can refer to `examples/offline/offline_bcq.py` to see how to use it.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        max_action: float,
        device: str | int | torch.device = "cpu",
        phi: float = 0.05,
    ):
        # preprocess_net: input_dim=state_dim+action_dim, output_dim=action_dim
        super().__init__()
        self.preprocess_net = preprocess_net
        self.device = device
        self.max_action = max_action
        self.phi = phi

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # preprocess_net
        logits = self.preprocess_net(torch.cat([state, action], -1))[0]
        noise = self.phi * self.max_action * torch.tanh(logits)
        # clip to [-max_action, max_action]
        return (noise + action).clamp(-self.max_action, self.max_action)


class VAE(nn.Module):
    """Implementation of VAE.

    It models the distribution of action. Given a state, it can generate actions similar to those in batch.
    It is used in BCQ algorithm.

    :param encoder: the encoder in VAE. Its input_dim must be
        state_dim + action_dim, and output_dim must be hidden_dim.
    :param decoder: the decoder in VAE. Its input_dim must be
        state_dim + latent_dim, and output_dim must be action_dim.
    :param hidden_dim: the size of the last linear-layer in encoder.
    :param latent_dim: the size of latent layer.
    :param max_action: the maximum value of each dimension of action.
    :param device: which device to create this model on.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        You can refer to `examples/offline/offline_bcq.py` to see how to use it.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        hidden_dim: int,
        latent_dim: int,
        max_action: float,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.encoder = encoder

        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)

        self.decoder = decoder

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [state, action] -> z , [state, z] -> action
        latent_z = self.encoder(torch.cat([state, action], -1))
        # shape of z: (state.shape[:-1], hidden_dim)

        mean = self.mean(latent_z)
        # Clamped for numerical stability
        log_std = self.log_std(latent_z).clamp(-4, 15)
        std = torch.exp(log_std)
        # shape of mean, std: (state.shape[:-1], latent_dim)

        latent_z = mean + std * torch.randn_like(std)  # (state.shape[:-1], latent_dim)

        reconstruction = self.decode(state, latent_z)  # (state.shape[:-1], action_dim)
        return reconstruction, mean, std

    def decode(
        self,
        state: torch.Tensor,
        latent_z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # decode(state) -> action
        if latent_z is None:
            # state.shape[0] may be batch_size
            # latent vector clipped to [-0.5, 0.5]
            latent_z = (
                torch.randn(state.shape[:-1] + (self.latent_dim,)).to(self.device).clamp(-0.5, 0.5)
            )

        # decode z with state!
        return self.max_action * torch.tanh(self.decoder(torch.cat([state, latent_z], -1)))
