from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch

from tianshou.exploration import BaseNoise
from tianshou.highlevel.params.alpha import AutoAlphaFactory
from tianshou.highlevel.params.env_param import FloatEnvParamFactory
from tianshou.highlevel.params.lr_scheduler import LRSchedulerFactory
from tianshou.highlevel.params.noise import NoiseFactory


class ParamTransformer(ABC):
    @abstractmethod
    def transform(self, kwargs: dict[str, Any]) -> None:
        pass

    @staticmethod
    def get(d: dict[str, Any], key: str, drop: bool = False) -> Any:
        value = d[key]
        if drop:
            del d[key]
        return value


@dataclass
class Params:
    def create_kwargs(self, *transformers: ParamTransformer) -> dict[str, Any]:
        d = asdict(self)
        for transformer in transformers:
            transformer.transform(d)
        return d


@dataclass
class PGParams(Params):
    """Config of general policy-gradient algorithms."""

    discount_factor: float = 0.99
    reward_normalization: bool = False
    deterministic_eval: bool = False
    action_scaling: bool = True
    action_bound_method: Literal["clip", "tanh"] | None = "clip"


@dataclass
class A2CParams(PGParams):
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float | None = None
    gae_lambda: float = 0.95
    max_batchsize: int = 256


@dataclass
class PPOParams(A2CParams):
    """PPO specific config."""

    eps_clip: float = 0.2
    dual_clip: float | None = None
    value_clip: bool = False
    advantage_normalization: bool = True
    recompute_advantage: bool = False
    lr: float = 1e-3
    lr_scheduler_factory: LRSchedulerFactory | None = None


@dataclass
class ActorAndDualCriticsParams(Params):
    actor_lr: float = 1e-3
    critic1_lr: float = 1e-3
    critic2_lr: float = 1e-3
    actor_lr_scheduler_factory: LRSchedulerFactory | None = None
    critic1_lr_scheduler_factory: LRSchedulerFactory | None = None
    critic2_lr_scheduler_factory: LRSchedulerFactory | None = None


@dataclass
class SACParams(ActorAndDualCriticsParams):
    tau: float = 0.005
    gamma: float = 0.99
    alpha: float | tuple[float, torch.Tensor, torch.optim.Optimizer] | AutoAlphaFactory = 0.2
    estimation_step: int = 1
    exploration_noise: BaseNoise | Literal["default"] | None = None
    deterministic_eval: bool = True
    action_scaling: bool = True
    action_bound_method: Literal["clip"] | None = "clip"


@dataclass
class TD3Params(ActorAndDualCriticsParams):
    tau: float = 0.005
    gamma: float = 0.99
    exploration_noise: BaseNoise | Literal["default"] | NoiseFactory | None = "default"
    policy_noise: float | FloatEnvParamFactory = 0.2
    noise_clip: float | FloatEnvParamFactory = 0.5
    update_actor_freq: int = 2
    estimation_step: int = 1
    action_scaling: bool = True
    action_bound_method: Literal["clip"] | None = "clip"
