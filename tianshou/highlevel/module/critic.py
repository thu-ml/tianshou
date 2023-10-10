from abc import ABC, abstractmethod
from collections.abc import Sequence

from torch import nn

from tianshou.highlevel.env import Environments, EnvType
from tianshou.highlevel.module.core import TDevice, init_linear_orthogonal
from tianshou.highlevel.module.module_opt import ModuleOpt
from tianshou.highlevel.optim import OptimizerFactory
from tianshou.utils.net import continuous, discrete
from tianshou.utils.net.common import Net
from tianshou.utils.string import ToStringMixin


class CriticFactory(ToStringMixin, ABC):
    @abstractmethod
    def create_module(self, envs: Environments, device: TDevice, use_action: bool) -> nn.Module:
        pass

    def create_module_opt(
        self,
        envs: Environments,
        device: TDevice,
        use_action: bool,
        optim_factory: OptimizerFactory,
        lr: float,
    ) -> ModuleOpt:
        module = self.create_module(envs, device, use_action)
        opt = optim_factory.create_optimizer(module, lr)
        return ModuleOpt(module, opt)


class CriticFactoryDefault(CriticFactory):
    """A critic factory which, depending on the type of environment, creates a suitable MLP-based critic."""

    DEFAULT_HIDDEN_SIZES = (64, 64)

    def __init__(self, hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES):
        self.hidden_sizes = hidden_sizes

    def create_module(self, envs: Environments, device: TDevice, use_action: bool) -> nn.Module:
        env_type = envs.get_type()
        if env_type == EnvType.CONTINUOUS:
            return CriticFactoryContinuousNet(self.hidden_sizes).create_module(
                envs,
                device,
                use_action,
            )
        elif env_type == EnvType.DISCRETE:
            return CriticFactoryDiscreteNet(self.hidden_sizes).create_module(
                envs,
                device,
                use_action,
            )
        else:
            raise ValueError(f"{env_type} not supported")


class CriticFactoryContinuousNet(CriticFactory):
    def __init__(self, hidden_sizes: Sequence[int]):
        self.hidden_sizes = hidden_sizes

    def create_module(self, envs: Environments, device: TDevice, use_action: bool) -> nn.Module:
        action_shape = envs.get_action_shape() if use_action else 0
        net_c = Net(
            envs.get_observation_shape(),
            action_shape=action_shape,
            hidden_sizes=self.hidden_sizes,
            concat=use_action,
            activation=nn.Tanh,
            device=device,
        )
        critic = continuous.Critic(net_c, device=device).to(device)
        init_linear_orthogonal(critic)
        return critic


class CriticFactoryDiscreteNet(CriticFactory):
    def __init__(self, hidden_sizes: Sequence[int]):
        self.hidden_sizes = hidden_sizes

    def create_module(self, envs: Environments, device: TDevice, use_action: bool) -> nn.Module:
        action_shape = envs.get_action_shape() if use_action else 0
        net_c = Net(
            envs.get_observation_shape(),
            action_shape=action_shape,
            hidden_sizes=self.hidden_sizes,
            concat=use_action,
            activation=nn.Tanh,
            device=device,
        )
        critic = discrete.Critic(net_c, device=device).to(device)
        init_linear_orthogonal(critic)
        return critic
