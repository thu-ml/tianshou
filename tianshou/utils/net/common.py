from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Generic, TypeAlias, TypeVar, cast, no_type_check

import numpy as np
import torch
from gymnasium import spaces
from torch import nn

from tianshou.data.batch import Batch
from tianshou.data.types import RecurrentStateBatch, TObs
from tianshou.utils.space_info import ActionSpaceInfo
from tianshou.utils.torch_utils import torch_device

ModuleType = type[nn.Module]
ArgsType = tuple[Any, ...] | dict[Any, Any] | Sequence[tuple[Any, ...]] | Sequence[dict[Any, Any]]
TActionShape: TypeAlias = Sequence[int] | int | np.int64
TLinearLayer: TypeAlias = Callable[[int, int], nn.Module]
T = TypeVar("T")


def miniblock(
    input_size: int,
    output_size: int = 0,
    norm_layer: ModuleType | None = None,
    norm_args: tuple[Any, ...] | dict[Any, Any] | None = None,
    activation: ModuleType | None = None,
    act_args: tuple[Any, ...] | dict[Any, Any] | None = None,
    linear_layer: TLinearLayer = nn.Linear,
) -> list[nn.Module]:
    """Construct a miniblock with given input/output-size, norm layer and activation."""
    layers: list[nn.Module] = [linear_layer(input_size, output_size)]
    if norm_layer is not None:
        if isinstance(norm_args, tuple):
            layers += [norm_layer(output_size, *norm_args)]
        elif isinstance(norm_args, dict):
            layers += [norm_layer(output_size, **norm_args)]
        else:
            layers += [norm_layer(output_size)]
    if activation is not None:
        if isinstance(act_args, tuple):
            layers += [activation(*act_args)]
        elif isinstance(act_args, dict):
            layers += [activation(**act_args)]
        else:
            layers += [activation()]
    return layers


class ModuleWithVectorOutput(nn.Module):
    """
    A module that outputs a vector of a known size.

    Use `from_module` to adapt a module to this interface.
    """

    def __init__(self, output_dim: int) -> None:
        """:param output_dim: the dimension of the output vector."""
        super().__init__()
        self.output_dim = output_dim

    @staticmethod
    def from_module(module: nn.Module, output_dim: int) -> "ModuleWithVectorOutput":
        """
        :param module: the module to adapt.
        :param output_dim: dimension of the output vector produced by the module.
        """
        return ModuleWithVectorOutputAdapter(module, output_dim)

    def get_output_dim(self) -> int:
        """:return: the dimension of the output vector."""
        return self.output_dim


class ModuleWithVectorOutputAdapter(ModuleWithVectorOutput):
    """Adapts a module with vector output to provide the :class:`ModuleWithVectorOutput` interface."""

    def __init__(self, module: nn.Module, output_dim: int) -> None:
        """
        :param module: the module to adapt.
        :param output_dim: the dimension of the output vector produced by the module.
        """
        super().__init__(output_dim)
        self.module = module

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)


class MLP(ModuleWithVectorOutput):
    """Simple MLP backbone."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int = 0,
        hidden_sizes: Sequence[int] = (),
        norm_layer: ModuleType | Sequence[ModuleType] | None = None,
        norm_args: ArgsType | None = None,
        activation: ModuleType | Sequence[ModuleType] | None = nn.ReLU,
        act_args: ArgsType | None = None,
        linear_layer: TLinearLayer = nn.Linear,
        flatten_input: bool = True,
    ) -> None:
        """
        :param input_dim: dimension of the input vector.
        :param output_dim: dimension of the output vector. If set to 0, there
            is no explicit final linear layer and the output dimension is the last hidden layer's dimension.
        :param hidden_sizes: shape of MLP passed in as a list, not including
            input_dim and output_dim.
        :param norm_layer: use which normalization before activation, e.g.,
            ``nn.LayerNorm`` and ``nn.BatchNorm1d``. Default to no normalization.
            You can also pass a list of normalization modules with the same length
            of hidden_sizes, to use different normalization module in different
            layers. Default to no normalization.
        :param activation: which activation to use after each layer, can be both
            the same activation for all layers if passed in nn.Module, or different
            activation for different Modules if passed in a list. Default to
            nn.ReLU.
        :param linear_layer: use this module as linear layer. Default to nn.Linear.
        :param flatten_input: whether to flatten input data. Default to True.
        """
        if norm_layer:
            if isinstance(norm_layer, list):
                assert len(norm_layer) == len(hidden_sizes)
                norm_layer_list = norm_layer
                if isinstance(norm_args, list):
                    assert len(norm_args) == len(hidden_sizes)
                    norm_args_list = norm_args
                else:
                    norm_args_list = [norm_args for _ in range(len(hidden_sizes))]
            else:
                norm_layer_list = [norm_layer for _ in range(len(hidden_sizes))]
                norm_args_list = [norm_args for _ in range(len(hidden_sizes))]
        else:
            norm_layer_list = [None] * len(hidden_sizes)
            norm_args_list = [None] * len(hidden_sizes)
        if activation:
            if isinstance(activation, list):
                assert len(activation) == len(hidden_sizes)
                activation_list = activation
                if isinstance(act_args, list):
                    assert len(act_args) == len(hidden_sizes)
                    act_args_list = act_args
                else:
                    act_args_list = [act_args for _ in range(len(hidden_sizes))]
            else:
                activation_list = [activation for _ in range(len(hidden_sizes))]
                act_args_list = [act_args for _ in range(len(hidden_sizes))]
        else:
            activation_list = [None] * len(hidden_sizes)
            act_args_list = [None] * len(hidden_sizes)
        hidden_sizes = [input_dim, *list(hidden_sizes)]
        model = []
        for in_dim, out_dim, norm, norm_args, activ, act_args in zip(
            hidden_sizes[:-1],
            hidden_sizes[1:],
            norm_layer_list,
            norm_args_list,
            activation_list,
            act_args_list,
            strict=True,
        ):
            model += miniblock(in_dim, out_dim, norm, norm_args, activ, act_args, linear_layer)
        if output_dim > 0:
            model += [linear_layer(hidden_sizes[-1], output_dim)]
        super().__init__(output_dim or hidden_sizes[-1])
        self.model = nn.Sequential(*model)
        self.flatten_input = flatten_input

    @no_type_check
    def forward(self, obs: np.ndarray | torch.Tensor) -> torch.Tensor:
        device = torch_device(self)
        obs = torch.as_tensor(obs, device=device, dtype=torch.float32)
        if self.flatten_input:
            obs = obs.flatten(1)
        return self.model(obs)


TRecurrentState = TypeVar("TRecurrentState", bound=Any)


class ActionReprNet(Generic[TRecurrentState], nn.Module, ABC):
    """Abstract base class for neural networks used to compute action-related
    representations from environment observations, which defines the
    signature of the forward method.

    An action-related representation can be a number of things, including:
      * a distribution over actions in a discrete action space in the form of a vector of
        unnormalized log probabilities (called "logits" in PyTorch jargon)
      * the Q-values of all actions in a discrete action space
      * the parameters of a distribution (e.g., mean and std. dev. for a Gaussian distribution)
        over actions in a continuous action space
    """

    @abstractmethod
    def forward(
        self,
        obs: TObs,
        state: TRecurrentState | None = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor | Sequence[torch.Tensor], TRecurrentState | None]:
        """
        The main method for tianshou to compute action representations (such as actions, inputs of distributions, Q-values, etc)
        from env observations.
        Implementations will always make use of the preprocess_net as the first processing step.

        :param obs: the observations from the environment as retrieved from `ObsBatchProtocol.obs`.
            If the environment is a dict env, this will be an instance of Batch, otherwise it will be an array (or tensor if your
            env returns tensors).
        :param state: the hidden state of the RNN, if applicable
        :param info: the info object from the environment step
        :return: a tuple (action_repr, hidden_state), where action_repr is either an actual action for the environment or
            a representation from which it can be retrieved/sampled (e.g., mean and std for a Gaussian distribution),
            and hidden_state is the new hidden state of the RNN, if applicable.
        """


class ActionReprNetWithVectorOutput(Generic[T], ActionReprNet[T], ModuleWithVectorOutput):
    """A neural network for the computation of action-related representations which outputs
    a vector of a known size.
    """

    def __init__(self, output_dim: int) -> None:
        super().__init__(output_dim)


class Actor(Generic[T], ActionReprNetWithVectorOutput[T], ABC):
    @abstractmethod
    def get_preprocess_net(self) -> ModuleWithVectorOutput:
        """Returns the network component that is used for pre-processing, i.e.
        the component which produces a latent representation, which then is transformed
        into the final output.
        This is, therefore, the first part of the network which processes the input.
        For example, a CNN is often used in Atari examples.

        We need this method to be able to share latent representation computations with
        other networks (e.g. critics) within an algorithm.

        Actors that do not have a pre-processing stage can return nn.Identity()
        (see :class:`RandomActor` for an example).
        """


class Net(ActionReprNetWithVectorOutput[Any]):
    """A multi-layer perceptron which outputs an action-related representation.

    :param state_shape: int or a sequence of int of the shape of state.
    :param action_shape: int or a sequence of int of the shape of action.
    :param hidden_sizes: shape of MLP passed in as a list.
    :param norm_layer: use which normalization before activation, e.g.,
        ``nn.LayerNorm`` and ``nn.BatchNorm1d``. Default to no normalization.
        You can also pass a list of normalization modules with the same length
        of hidden_sizes, to use different normalization module in different
        layers. Default to no normalization.
    :param activation: which activation to use after each layer, can be both
        the same activation for all layers if passed in nn.Module, or different
        activation for different Modules if passed in a list. Default to
        nn.ReLU.
    :param softmax: whether to apply a softmax layer over the last layer's
        output.
    :param concat: whether the input shape is concatenated by state_shape
        and action_shape. If it is True, ``action_shape`` is not the output
        shape, but affects the input shape only.
    :param num_atoms: in order to expand to the net of distributional RL.
        Default to 1 (not use).
    :param dueling_param: whether to use dueling network to calculate Q
        values (for Dueling DQN). If you want to use dueling option, you should
        pass a tuple of two dict (first for Q and second for V) stating
        self-defined arguments as stated in
        class:`~tianshou.utils.net.common.MLP`. Default to None.
    :param linear_layer: use this module constructor, which takes the input
        and output dimension as input, as linear layer. Default to nn.Linear.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.MLP` for more
        detailed explanation on the usage of activation, norm_layer, etc.

        You can also refer to :class:`~tianshou.utils.net.continuous.Actor`,
        :class:`~tianshou.utils.net.continuous.Critic`, etc, to see how it's
        suggested be used.
    """

    def __init__(
        self,
        *,
        state_shape: int | Sequence[int],
        action_shape: TActionShape = 0,
        hidden_sizes: Sequence[int] = (),
        norm_layer: ModuleType | Sequence[ModuleType] | None = None,
        norm_args: ArgsType | None = None,
        activation: ModuleType | Sequence[ModuleType] | None = nn.ReLU,
        act_args: ArgsType | None = None,
        softmax: bool = False,
        concat: bool = False,
        num_atoms: int = 1,
        dueling_param: tuple[dict[str, Any], dict[str, Any]] | None = None,
        linear_layer: TLinearLayer = nn.Linear,
    ) -> None:
        input_dim = int(np.prod(state_shape))
        action_dim = int(np.prod(action_shape)) * num_atoms
        if concat:
            input_dim += action_dim
        use_dueling = dueling_param is not None
        model = MLP(
            input_dim=input_dim,
            output_dim=action_dim if not use_dueling and not concat else 0,
            hidden_sizes=hidden_sizes,
            norm_layer=norm_layer,
            norm_args=norm_args,
            activation=activation,
            act_args=act_args,
            linear_layer=linear_layer,
        )
        Q: MLP | None = None
        V: MLP | None = None
        if use_dueling:  # dueling DQN
            assert dueling_param is not None
            kwargs_update = {
                "input_dim": model.output_dim,
            }
            # Important: don't change the original dict (e.g., don't use .update())
            q_kwargs = {**dueling_param[0], **kwargs_update}
            v_kwargs = {**dueling_param[1], **kwargs_update}

            q_kwargs["output_dim"] = 0 if concat else action_dim
            v_kwargs["output_dim"] = 0 if concat else num_atoms
            Q, V = MLP(**q_kwargs), MLP(**v_kwargs)
            output_dim = Q.output_dim
        else:
            output_dim = model.output_dim

        super().__init__(output_dim)
        self.use_dueling = use_dueling
        self.softmax = softmax
        self.num_atoms = num_atoms
        self.model = model
        self.Q = Q
        self.V = V

    def forward(
        self,
        obs: TObs,
        state: T | None = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, T | Any]:
        """Mapping: obs -> flatten (inside MLP)-> logits.

        :param obs:
        :param state: unused and returned as is
        :param info: unused
        """
        logits = self.model(obs)
        batch_size = logits.shape[0]
        if self.use_dueling:  # Dueling DQN
            assert self.Q is not None
            assert self.V is not None
            q, v = self.Q(logits), self.V(logits)
            if self.num_atoms > 1:
                q = q.view(batch_size, -1, self.num_atoms)
                v = v.view(batch_size, -1, self.num_atoms)
            logits = q - q.mean(dim=1, keepdim=True) + v
        elif self.num_atoms > 1:
            logits = logits.view(batch_size, -1, self.num_atoms)
        if self.softmax:
            logits = torch.softmax(logits, dim=-1)
        return logits, state


class Recurrent(ActionReprNetWithVectorOutput[RecurrentStateBatch]):
    """Simple Recurrent network based on LSTM.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        *,
        layer_num: int,
        state_shape: int | Sequence[int],
        action_shape: TActionShape,
        hidden_layer_size: int = 128,
    ) -> None:
        output_dim = int(np.prod(action_shape))
        super().__init__(output_dim)
        self.nn = nn.LSTM(
            input_size=hidden_layer_size,
            hidden_size=hidden_layer_size,
            num_layers=layer_num,
            batch_first=True,
        )
        self.fc1 = nn.Linear(int(np.prod(state_shape)), hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, output_dim)

    def get_preprocess_net(self) -> ModuleWithVectorOutput:
        return ModuleWithVectorOutput.from_module(nn.Identity(), self.output_dim)

    def forward(
        self,
        obs: TObs,
        state: RecurrentStateBatch | None = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, RecurrentStateBatch]:
        """Mapping: obs -> flatten -> logits.

        In the evaluation mode, `obs` should be with shape ``[bsz, dim]``; in the
        training mode, `obs` should be with shape ``[bsz, len, dim]``. See the code
        and comment for more detail.

        :param obs:
        :param state: either None or a dict with keys 'hidden' and 'cell'
        :param info: unused
        :return: predicted action, next state as dict with keys 'hidden' and 'cell'
        """
        # Note: the original type of state is Batch but it might also be a dict
        # If it is a Batch, .issubset(state) will not work. However,
        # issubset(state.keys()) always works
        if state is not None and not {"hidden", "cell"}.issubset(state.keys()):
            raise ValueError(
                f"Expected to find keys 'hidden' and 'cell' but instead found {state.keys()}",
            )

        device = torch_device(self)
        obs = torch.as_tensor(obs, device=device, dtype=torch.float32)
        # obs [bsz, len, dim] (training) or [bsz, dim] (evaluation)
        # In short, the tensor's shape in training phase is longer than which
        # in evaluation phase.
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(-2)
        obs = self.fc1(obs)
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
        obs = self.fc2(obs[:, -1])
        # please ensure the first dim is batch size: [bsz, len, ...]
        rnn_state_batch = cast(
            RecurrentStateBatch,
            Batch(
                {
                    "hidden": hidden.transpose(0, 1).detach(),
                    "cell": cell.transpose(0, 1).detach(),
                },
            ),
        )
        return obs, rnn_state_batch


class ActorCritic(nn.Module):
    """An actor-critic network for parsing parameters.

    Using ``actor_critic.parameters()`` instead of set.union or list+list to avoid
    issue #449.

    :param nn.Module actor: the actor network.
    :param nn.Module critic: the critic network.
    """

    def __init__(self, actor: nn.Module, critic: nn.Module) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic


class DataParallelNet(nn.Module):
    """DataParallel wrapper for training agent with multi-GPU.

    This class does only the conversion of input data type, from numpy array to torch's
    Tensor. If the input is a nested dictionary, the user should create a similar class
    to do the same thing.

    :param net: the network to be distributed in different GPUs.
    """

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = nn.DataParallel(net)

    def forward(
        self,
        obs: TObs,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        obs = obs.cuda()
        return self.net(obs, *args, **kwargs)


# The same functionality as DataParallelNet
# The duplication is worth it because the ActionReprNet abstraction is so important
class ActionReprNetDataParallelWrapper(ActionReprNet):
    def __init__(self, net: ActionReprNet) -> None:
        super().__init__()
        self.net = nn.DataParallel(net)

    def forward(
        self,
        obs: TObs,
        state: TRecurrentState | None = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, TRecurrentState | None]:
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        obs = obs.cuda()
        return self.net(obs, state=state, info=info)


class EnsembleLinear(nn.Module):
    """Linear Layer of Ensemble network.

    :param ensemble_size: Number of subnets in the ensemble.
    :param in_feature: dimension of the input vector.
    :param out_feature: dimension of the output vector.
    :param bias: whether to include an additive bias, default to be True.
    """

    def __init__(
        self,
        ensemble_size: int,
        in_feature: int,
        out_feature: int,
        bias: bool = True,
    ) -> None:
        super().__init__()

        # To be consistent with PyTorch default initializer
        k = np.sqrt(1.0 / in_feature)
        weight_data = torch.rand((ensemble_size, in_feature, out_feature)) * 2 * k - k
        self.weight = nn.Parameter(weight_data, requires_grad=True)

        self.bias_weights: nn.Parameter | None = None
        if bias:
            bias_data = torch.rand((ensemble_size, 1, out_feature)) * 2 * k - k
            self.bias_weights = nn.Parameter(bias_data, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.matmul(x, self.weight)
        if self.bias_weights is not None:
            x = x + self.bias_weights
        return x


class BranchingNet(ActionReprNet):
    """Branching dual Q network.

    Network for the BranchingDQNPolicy, it uses a common network module, a value module
    and action "branches" one for each dimension. It allows for a linear scaling
    of Q-value the output w.r.t. the number of dimensions in the action space.

    This network architecture efficiently handles environments with multiple independent
    action dimensions by using a branching structure. Instead of representing all action
    combinations (which grows exponentially), it represents each action dimension separately
    (linear scaling).
    For example, if there are 3 actions with 3 possible values each, then we would normally
    need to consider 3^4 = 81 unique actions, whereas with this architecture, we can instead
    use 3 branches with 4 actions per dimension, resulting in 3 * 4 = 12 values to be considered.

    Common use cases include multi-joint robotic control tasks, where each joint can be controlled
    independently.

    For more information, please refer to: arXiv:1711.08946.
    """

    def __init__(
        self,
        *,
        state_shape: int | Sequence[int],
        num_branches: int = 0,
        action_per_branch: int = 2,
        common_hidden_sizes: list[int] | None = None,
        value_hidden_sizes: list[int] | None = None,
        action_hidden_sizes: list[int] | None = None,
        norm_layer: ModuleType | None = None,
        norm_args: ArgsType | None = None,
        activation: ModuleType | None = nn.ReLU,
        act_args: ArgsType | None = None,
    ) -> None:
        """
        :param state_shape: int or a sequence of int of the shape of state.
        :param num_branches: number of action dimensions in the environment.
            Each branch represents one independent action dimension.
            For example, in a robot with 7 joints, you would set this to 7.
        :param action_per_branch: Number of possible discrete values for each action dimension.
             For example, if each joint can have 3 positions (left, center, right),
             you would set this to 3.
        :param common_hidden_sizes: shape of the common MLP network passed in as a list.
        :param value_hidden_sizes: shape of the value MLP network passed in as a list.
        :param action_hidden_sizes: shape of the action MLP network passed in as a list.
        :param norm_layer: use which normalization before activation, e.g.,
            ``nn.LayerNorm`` and ``nn.BatchNorm1d``. Default to no normalization.
            You can also pass a list of normalization modules with the same length
            of hidden_sizes, to use different normalization module in different
            layers. Default to no normalization.
        :param activation: which activation to use after each layer, can be both
            the same activation for all layers if passed in nn.Module, or different
            activation for different Modules if passed in a list. Default to
            nn.ReLU.
        """
        super().__init__()
        common_hidden_sizes = common_hidden_sizes or []
        value_hidden_sizes = value_hidden_sizes or []
        action_hidden_sizes = action_hidden_sizes or []

        self.num_branches = num_branches
        self.action_per_branch = action_per_branch
        # common network
        common_input_dim = int(np.prod(state_shape))
        common_output_dim = 0
        self.common = MLP(
            input_dim=common_input_dim,
            output_dim=common_output_dim,
            hidden_sizes=common_hidden_sizes,
            norm_layer=norm_layer,
            norm_args=norm_args,
            activation=activation,
            act_args=act_args,
        )
        # value network
        value_input_dim = common_hidden_sizes[-1]
        value_output_dim = 1
        self.value = MLP(
            input_dim=value_input_dim,
            output_dim=value_output_dim,
            hidden_sizes=value_hidden_sizes,
            norm_layer=norm_layer,
            norm_args=norm_args,
            activation=activation,
            act_args=act_args,
        )
        # action branching network
        action_input_dim = common_hidden_sizes[-1]
        action_output_dim = action_per_branch
        self.branches = nn.ModuleList(
            [
                MLP(
                    input_dim=action_input_dim,
                    output_dim=action_output_dim,
                    hidden_sizes=action_hidden_sizes,
                    norm_layer=norm_layer,
                    norm_args=norm_args,
                    activation=activation,
                    act_args=act_args,
                )
                for _ in range(self.num_branches)
            ],
        )

    def forward(
        self,
        obs: TObs,
        state: T | None = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, T | None]:
        """Mapping: obs -> model -> logits."""
        common_out = self.common(obs)
        value_out = self.value(common_out)
        value_out = torch.unsqueeze(value_out, 1)
        action_out = []
        for b in self.branches:
            action_out.append(b(common_out))
        action_scores = torch.stack(action_out, 1)
        action_scores = action_scores - torch.mean(action_scores, 2, keepdim=True)
        logits = value_out + action_scores
        return logits, state


def get_dict_state_decorator(
    state_shape: dict[str, int | Sequence[int]],
    keys: Sequence[str],
) -> tuple[Callable, int]:
    """A helper function to make Net or equivalent classes (e.g. Actor, Critic) applicable to dict state.

    The first return item, ``decorator_fn``, will alter the implementation of forward
    function of the given class by preprocessing the observation. The preprocessing is
    basically flatten the observation and concatenate them based on the ``keys`` order.
    The batch dimension is preserved if presented. The result observation shape will
    be equal to ``new_state_shape``, the second return item.

    :param state_shape: A dictionary indicating each state's shape
    :param keys: A list of state's keys. The flatten observation will be according to
        this list order.
    :returns: a 2-items tuple ``decorator_fn`` and ``new_state_shape``
    """
    original_shape = state_shape
    flat_state_shapes = []
    for k in keys:
        flat_state_shapes.append(int(np.prod(state_shape[k])))
    new_state_shape = sum(flat_state_shapes)

    def preprocess_obs(obs: Batch | dict | torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(obs, dict) or (isinstance(obs, Batch) and keys[0] in obs):
            if original_shape[keys[0]] == obs[keys[0]].shape:
                # No batch dim
                new_obs = torch.Tensor([obs[k] for k in keys]).flatten()
                # new_obs = torch.Tensor([obs[k] for k in keys]).reshape(1, -1)
            else:
                bsz = obs[keys[0]].shape[0]
                new_obs = torch.cat([torch.Tensor(obs[k].reshape(bsz, -1)) for k in keys], dim=1)
        else:
            new_obs = torch.Tensor(obs)
        return new_obs

    @no_type_check
    def decorator_fn(net_class):
        class new_net_class(net_class):
            def forward(self, obs: TObs, *args, **kwargs) -> Any:
                return super().forward(preprocess_obs(obs), *args, **kwargs)

        return new_net_class

    return decorator_fn, new_state_shape


class AbstractContinuousActorProbabilistic(Actor, ABC):
    """Type bound for probabilistic actors which output distribution parameters for continuous action spaces."""


class AbstractDiscreteActor(Actor, ABC):
    """
    Type bound for discrete actors.

    For on-policy algos like Reinforce, this typically directly outputs unnormalized log
    probabilities, which can be interpreted as "logits" in conjunction with a
    `torch.distributions.Categorical` instance.

    In Tianshou, discrete actors are also used for computing action distributions within
    Q-learning type algorithms (e.g., DQN). In this case, the observations are mapped
    to a vector of Q-values (one for each action). In other words, the component is actually
    a critic, not an actor in the traditional sense.
    Note that when sampling actions, the Q-values can be interpreted as inputs for
    a `torch.distributions.Categorical` instance, similar to the on-policy case mentioned
    above.
    """


class RandomActor(AbstractContinuousActorProbabilistic, AbstractDiscreteActor):
    """An actor that returns random actions.

    For continuous action spaces, forward returns a batch of random actions sampled from the action space.
    For discrete action spaces, forward returns a batch of n-dimensional arrays corresponding to the
    uniform distribution over the n possible actions (same interface as in :class:`~.net.discrete.Actor`).
    """

    def __init__(self, action_space: spaces.Box | spaces.Discrete) -> None:
        if isinstance(action_space, spaces.Discrete):
            output_dim = action_space.n
        else:
            output_dim = np.prod(action_space.shape)
        super().__init__(int(output_dim))
        self._action_space = action_space
        self._space_info = ActionSpaceInfo.from_space(action_space)

    @property
    def action_space(self) -> spaces.Box | spaces.Discrete:
        return self._action_space

    @property
    def space_info(self) -> ActionSpaceInfo:
        return self._space_info

    def get_preprocess_net(self) -> ModuleWithVectorOutput:
        return ModuleWithVectorOutput.from_module(nn.Identity(), self.output_dim)

    def get_output_dim(self) -> int:
        return self.space_info.action_dim

    @property
    def is_discrete(self) -> bool:
        return isinstance(self.action_space, spaces.Discrete)

    def forward(
        self,
        obs: TObs,
        state: T | None = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, T | None]:
        batch_size = len(obs)
        if isinstance(self.action_space, spaces.Box):
            action = np.stack([self.action_space.sample() for _ in range(batch_size)])
        else:
            # Discrete Actors currently return an n-dimensional array of probabilities for each action
            action = 1 / self.action_space.n * np.ones((batch_size, self.action_space.n))
        return torch.Tensor(action), state

    def compute_action_batch(self, obs: TObs) -> torch.Tensor:
        if self.is_discrete:
            # Different from forward which returns discrete probabilities, see comment there
            assert isinstance(self.action_space, spaces.Discrete)  # for mypy
            return torch.Tensor(np.random.randint(low=0, high=self.action_space.n, size=len(obs)))
        else:
            return self.forward(obs)[0]
