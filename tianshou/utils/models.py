import numpy as np
import torch
from torch import nn
from torch.distributions import Independent, Normal
from typing import Optional, Sequence

from tianshou.env import VectorEnvNormObs
from tianshou.policy import BasePolicy
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils.types import TDevice, TOptimClass, TShape


def resume_from_checkpoint(
    path: str,
    policy: BasePolicy,
    train_envs: Optional[VectorEnvNormObs] = None,
    test_envs: Optional[VectorEnvNormObs] = None,
    device: TDevice = None,
):
    ckpt = torch.load(path, map_location=device)
    policy.load_state_dict(ckpt["model"])
    print("Loaded agent from: ", path)

    obs_rms = ckpt.get("obs_rms")
    if obs_rms is not None:
        print(f"Loaded observation running mean from {path}")
        if train_envs:
            train_envs.set_obs_rms(ckpt["obs_rms"])
        if test_envs:
            test_envs.set_obs_rms(ckpt["obs_rms"])


def get_actor_critic(
    state_shape: TShape,
    hidden_sizes: Sequence[int],
    action_shape: TShape,
    device: TDevice = "cpu",
):
    net_a = Net(
        state_shape, hidden_sizes=hidden_sizes, activation=nn.Tanh, device=device
    )
    actor = ActorProb(net_a, action_shape, unbounded=True, device=device)
    net_c = Net(
        state_shape, hidden_sizes=hidden_sizes, activation=nn.Tanh, device=device
    )
    critic = Critic(net_c)
    return actor, critic


def init_actor_critic(actor: nn.Module, critic: nn.Module):
    """Initializes layers of actor and critic and returns an actor_critic object.

    **Note**: this modifies the actor and critic in place.
    """

    actor_critic = ActorCritic(actor, critic)
    torch.nn.init.constant_(actor.sigma_param, -0.5)
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
    if hasattr(actor, "mu"):
        # For continuous action spaces with Gaussian policies
        # do last policy layer scaling, this will make initial actions have (close to)
        # 0 mean and std, and will help boost performances,
        # see https://arxiv.org/abs/2006.05990, Fig.24 for details
        for m in actor.mu.modules():
            # TODO: seems like biases are initialized twice for the actor
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.zeros_(m.bias)
                m.weight.data.copy_(0.01 * m.weight.data)
    return actor_critic


def init_and_get_optim(
    actor: nn.Module,
    critic: nn.Module,
    lr: float,
    optim_class: TOptimClass = torch.optim.Adam,
):
    """Initializes layers of actor and critic and returns an optimizer.

    :param actor:
    :param critic:
    :param lr:
    :param optim_class: optimizer class or callable, should accept `lr` as kwarg
    :return: the optimizer instance
    """
    actor_critic = init_actor_critic(actor, critic)
    optim = optim_class(actor_critic.parameters(), lr=lr)
    return optim


def fixed_std_normal(*logits):
    return Independent(Normal(*logits), 1)
