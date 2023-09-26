from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import torch

from tianshou.exploration import BaseNoise
from tianshou.highlevel.env import Environments
from tianshou.highlevel.module import ModuleOpt, TDevice
from tianshou.highlevel.optim import OptimizerFactory
from tianshou.highlevel.params.alpha import AutoAlphaFactory
from tianshou.highlevel.params.env_param import FloatEnvParamFactory
from tianshou.highlevel.params.lr_scheduler import LRSchedulerFactory
from tianshou.highlevel.params.noise import NoiseFactory
from tianshou.utils import MultipleLRSchedulers


@dataclass(kw_only=True)
class ParamTransformerData:
    """Holds data that can be used by `ParamTransformer` instances to perform their transformation.

    The representation contains the superset of all data items that are required by different types of agent factories.
    An agent factory is expected to set only the attributes that are relevant to its parameters.
    """

    envs: Environments
    device: TDevice
    optim_factory: OptimizerFactory
    optim: torch.optim.Optimizer | None = None
    """the single optimizer for the case where there is just one"""
    actor: ModuleOpt | None = None
    critic1: ModuleOpt | None = None
    critic2: ModuleOpt | None = None


class ParamTransformer(ABC):
    @abstractmethod
    def transform(self, params: dict[str, Any], data: ParamTransformerData) -> None:
        pass

    @staticmethod
    def get(d: dict[str, Any], key: str, drop: bool = False) -> Any:
        value = d[key]
        if drop:
            del d[key]
        return value


class ParamTransformerDrop(ParamTransformer):
    def __init__(self, *keys: str):
        self.keys = keys

    def transform(self, kwargs: dict[str, Any], data: ParamTransformerData) -> None:
        for k in self.keys:
            del kwargs[k]


class ParamTransformerLRScheduler(ParamTransformer):
    """Transforms a key containing a learning rate scheduler factory (removed) into a key containing
    a learning rate scheduler (added) for the data member `optim`.
    """

    def __init__(self, key_scheduler_factory: str, key_scheduler: str):
        self.key_scheduler_factory = key_scheduler_factory
        self.key_scheduler = key_scheduler

    def transform(self, params: dict[str, Any], data: ParamTransformerData) -> None:
        assert data.optim is not None
        factory: LRSchedulerFactory | None = self.get(params, self.key_scheduler_factory, drop=True)
        params[self.key_scheduler] = (
            factory.create_scheduler(data.optim) if factory is not None else None
        )


class ParamTransformerMultiLRScheduler(ParamTransformer):
    """Transforms several scheduler factories into a single scheduler, which may be a MultipleLRSchedulers instance
    if more than one factory is indeed given.
    """

    def __init__(self, optim_key_list: list[tuple[torch.optim.Optimizer, str]], key_scheduler: str):
        """:param optim_key_list: a list of tuples (optimizer, key of learning rate factory)
        :param key_scheduler: the key under which to store the resulting learning rate scheduler
        """
        self.optim_key_list = optim_key_list
        self.key_scheduler = key_scheduler

    def transform(self, params: dict[str, Any], data: ParamTransformerData) -> None:
        lr_schedulers = []
        for optim, lr_scheduler_factory_key in self.optim_key_list:
            lr_scheduler_factory: LRSchedulerFactory | None = self.get(
                params,
                lr_scheduler_factory_key,
                drop=True,
            )
            if lr_scheduler_factory is not None:
                lr_schedulers.append(lr_scheduler_factory.create_scheduler(optim))
        match len(lr_schedulers):
            case 0:
                lr_scheduler = None
            case 1:
                lr_scheduler = lr_schedulers[0]
            case _:
                lr_scheduler = MultipleLRSchedulers(*lr_schedulers)
        params[self.key_scheduler] = lr_scheduler


class ParamTransformerActorDualCriticsLRScheduler(ParamTransformer):
    def __init__(
        self,
        key_scheduler_factory_actor: str,
        key_scheduler_factory_critic1: str,
        key_scheduler_factory_critic2: str,
        key_scheduler: str,
    ):
        self.key_factory_actor = key_scheduler_factory_actor
        self.key_factory_critic1 = key_scheduler_factory_critic1
        self.key_factory_critic2 = key_scheduler_factory_critic2
        self.key_scheduler = key_scheduler

    def transform(self, params: dict[str, Any], data: ParamTransformerData) -> None:
        transformer = ParamTransformerMultiLRScheduler(
            [
                (data.actor.optim, self.key_factory_actor),
                (data.critic1.optim, self.key_factory_critic1),
                (data.critic2.optim, self.key_factory_critic2),
            ],
            self.key_scheduler,
        )
        transformer.transform(params, data)


class ParamTransformerAutoAlpha(ParamTransformer):
    def __init__(self, key: str):
        self.key = key

    def transform(self, kwargs: dict[str, Any], data: ParamTransformerData) -> None:
        alpha = self.get(kwargs, self.key)
        if isinstance(alpha, AutoAlphaFactory):
            kwargs[self.key] = alpha.create_auto_alpha(data.envs, data.optim_factory, data.device)


class ParamTransformerNoiseFactory(ParamTransformer):
    def __init__(self, key: str):
        self.key = key

    def transform(self, params: dict[str, Any], data: ParamTransformerData) -> None:
        value = params[self.key]
        if isinstance(value, NoiseFactory):
            params[self.key] = value.create_noise(data.envs)


class ParamTransformerFloatEnvParamFactory(ParamTransformer):
    def __init__(self, key: str):
        self.key = key

    def transform(self, kwargs: dict[str, Any], data: ParamTransformerData) -> None:
        value = kwargs[self.key]
        if isinstance(value, FloatEnvParamFactory):
            kwargs[self.key] = value.create_param(data.envs)


class ITransformableParams(ABC):
    @abstractmethod
    def _add_transformer(self, transformer: ParamTransformer):
        pass


@dataclass
class Params(ITransformableParams):
    _param_transformers: list[ParamTransformer] = field(
        init=False,
        default_factory=list,
        repr=False,
    )

    def _add_transformer(self, transformer: ParamTransformer):
        self._param_transformers.append(transformer)

    def create_kwargs(self, data: ParamTransformerData) -> dict[str, Any]:
        params = asdict(self)
        for transformer in self._param_transformers:
            transformer.transform(params, data)
        del params["_param_transformers"]
        return params


@dataclass
class ParamsMixinLearningRateWithScheduler(ITransformableParams, ABC):
    lr: float = 1e-3
    lr_scheduler_factory: LRSchedulerFactory | None = None

    def __post_init__(self):
        self._add_transformer(ParamTransformerDrop("lr"))
        self._add_transformer(ParamTransformerLRScheduler("lr_scheduler_factory", "lr_scheduler"))


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
class PPOParams(A2CParams, ParamsMixinLearningRateWithScheduler):
    """PPO specific config."""

    eps_clip: float = 0.2
    dual_clip: float | None = None
    value_clip: bool = False
    advantage_normalization: bool = True
    recompute_advantage: bool = False


@dataclass
class ParamsMixinActorAndDualCritics(ITransformableParams, ABC):
    actor_lr: float = 1e-3
    critic1_lr: float = 1e-3
    critic2_lr: float = 1e-3
    actor_lr_scheduler_factory: LRSchedulerFactory | None = None
    critic1_lr_scheduler_factory: LRSchedulerFactory | None = None
    critic2_lr_scheduler_factory: LRSchedulerFactory | None = None

    def __post_init__(self):
        self._add_transformer(ParamTransformerDrop("actor_lr", "critic1_lr", "critic2_lr"))
        self._add_transformer(
            ParamTransformerActorDualCriticsLRScheduler(
                "actor_lr_scheduler_factory",
                "critic1_lr_scheduler_factory",
                "critic2_lr_scheduler_factory",
                "lr_scheduler",
            ),
        )


@dataclass
class SACParams(Params, ParamsMixinActorAndDualCritics):
    tau: float = 0.005
    gamma: float = 0.99
    alpha: float | tuple[float, torch.Tensor, torch.optim.Optimizer] | AutoAlphaFactory = 0.2
    estimation_step: int = 1
    exploration_noise: BaseNoise | Literal["default"] | NoiseFactory | None = None
    deterministic_eval: bool = True
    action_scaling: bool = True
    action_bound_method: Literal["clip"] | None = "clip"

    def __post_init__(self):
        ParamsMixinActorAndDualCritics.__post_init__(self)
        self._add_transformer(ParamTransformerAutoAlpha("alpha"))
        self._add_transformer(ParamTransformerNoiseFactory("exploration_noise"))


@dataclass
class TD3Params(Params, ParamsMixinActorAndDualCritics):
    tau: float = 0.005
    gamma: float = 0.99
    exploration_noise: BaseNoise | Literal["default"] | NoiseFactory | None = "default"
    policy_noise: float | FloatEnvParamFactory = 0.2
    noise_clip: float | FloatEnvParamFactory = 0.5
    update_actor_freq: int = 2
    estimation_step: int = 1
    action_scaling: bool = True
    action_bound_method: Literal["clip"] | None = "clip"

    def __post_init__(self):
        ParamsMixinActorAndDualCritics.__post_init__(self)
        self._add_transformer(ParamTransformerNoiseFactory("exploration_noise"))
        self._add_transformer(ParamTransformerFloatEnvParamFactory("policy_noise"))
        self._add_transformer(ParamTransformerFloatEnvParamFactory("noise_clip"))
