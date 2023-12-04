from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol

import torch
from torch.optim.lr_scheduler import LRScheduler

from tianshou.exploration import BaseNoise
from tianshou.highlevel.env import Environments
from tianshou.highlevel.module.core import TDevice
from tianshou.highlevel.module.module_opt import ModuleOpt
from tianshou.highlevel.optim import OptimizerFactory
from tianshou.highlevel.params.alpha import AutoAlphaFactory
from tianshou.highlevel.params.dist_fn import (
    DistributionFunctionFactory,
    DistributionFunctionFactoryDefault,
)
from tianshou.highlevel.params.env_param import EnvValueFactory, FloatEnvValueFactory
from tianshou.highlevel.params.lr_scheduler import LRSchedulerFactory
from tianshou.highlevel.params.noise import NoiseFactory
from tianshou.policy.modelfree.pg import TDistributionFunction
from tianshou.utils import MultipleLRSchedulers
from tianshou.utils.string import ToStringMixin


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
    """Base class for parameter transformations from high to low-level API.

    Transforms one or more parameters from the representation used by the high-level API
    to the representation required by the (low-level) policy implementation.
    It operates directly on a dictionary of keyword arguments, which is initially
    generated from the parameter dataclass (subclass of `Params`).
    """

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


class ParamTransformerChangeValue(ParamTransformer):
    def __init__(self, key: str):
        self.key = key

    def transform(self, params: dict[str, Any], data: ParamTransformerData) -> None:
        params[self.key] = self.change_value(params[self.key], data)

    @abstractmethod
    def change_value(self, value: Any, data: ParamTransformerData) -> Any:
        pass


class ParamTransformerLRScheduler(ParamTransformer):
    """Transformer for learning rate scheduler params.

    Transforms a key containing a learning rate scheduler factory (removed) into a key containing
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
    def __init__(self, optim_key_list: list[tuple[torch.optim.Optimizer, str]], key_scheduler: str):
        """Transforms several scheduler factories into a single scheduler.

         The result may be a `MultipleLRSchedulers` instance if more than one factory is indeed given.

        :param optim_key_list: a list of tuples (optimizer, key of learning rate factory)
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
        lr_scheduler: LRScheduler | MultipleLRSchedulers | None
        match len(lr_schedulers):
            case 0:
                lr_scheduler = None
            case 1:
                lr_scheduler = lr_schedulers[0]
            case _:
                lr_scheduler = MultipleLRSchedulers(*lr_schedulers)
        params[self.key_scheduler] = lr_scheduler


class ParamTransformerActorAndCriticLRScheduler(ParamTransformer):
    def __init__(
        self,
        key_scheduler_factory_actor: str,
        key_scheduler_factory_critic: str,
        key_scheduler: str,
    ):
        self.key_factory_actor = key_scheduler_factory_actor
        self.key_factory_critic = key_scheduler_factory_critic
        self.key_scheduler = key_scheduler

    def transform(self, params: dict[str, Any], data: ParamTransformerData) -> None:
        assert data.actor is not None and data.critic1 is not None
        transformer = ParamTransformerMultiLRScheduler(
            [
                (data.actor.optim, self.key_factory_actor),
                (data.critic1.optim, self.key_factory_critic),
            ],
            self.key_scheduler,
        )
        transformer.transform(params, data)


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
        assert data.actor is not None and data.critic1 is not None and data.critic2 is not None
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


class ParamTransformerNoiseFactory(ParamTransformerChangeValue):
    def change_value(self, value: Any, data: ParamTransformerData) -> Any:
        if isinstance(value, NoiseFactory):
            value = value.create_noise(data.envs)
        return value


class ParamTransformerFloatEnvParamFactory(ParamTransformerChangeValue):
    def change_value(self, value: Any, data: ParamTransformerData) -> Any:
        if isinstance(value, EnvValueFactory):
            value = value.create_value(data.envs)
        return value


class ParamTransformerDistributionFunction(ParamTransformerChangeValue):
    def change_value(self, value: Any, data: ParamTransformerData) -> Any:
        if value == "default":
            value = DistributionFunctionFactoryDefault().create_dist_fn(data.envs)
        elif isinstance(value, DistributionFunctionFactory):
            value = value.create_dist_fn(data.envs)
        return value


class ParamTransformerActionScaling(ParamTransformerChangeValue):
    def change_value(self, value: Any, data: ParamTransformerData) -> Any:
        if value == "default":
            return data.envs.get_type().is_continuous()
        else:
            return value


class GetParamTransformersProtocol(Protocol):
    def _get_param_transformers(self) -> list[ParamTransformer]:
        pass


@dataclass
class Params(GetParamTransformersProtocol, ToStringMixin):
    def create_kwargs(self, data: ParamTransformerData) -> dict[str, Any]:
        params = asdict(self)
        for transformer in self._get_param_transformers():
            transformer.transform(params, data)
        return params

    def _get_param_transformers(self) -> list[ParamTransformer]:
        return []


@dataclass
class ParamsMixinLearningRateWithScheduler(GetParamTransformersProtocol):
    lr: float = 1e-3
    """the learning rate to use in the gradient-based optimizer"""
    lr_scheduler_factory: LRSchedulerFactory | None = None
    """factory for the creation of a learning rate scheduler"""

    def _get_param_transformers(self) -> list[ParamTransformer]:
        return [
            ParamTransformerDrop("lr"),
            ParamTransformerLRScheduler("lr_scheduler_factory", "lr_scheduler"),
        ]


@dataclass
class ParamsMixinActorAndCritic(GetParamTransformersProtocol):
    actor_lr: float = 1e-3
    """the learning rate to use for the actor network"""
    critic_lr: float = 1e-3
    """the learning rate to use for the critic network"""
    actor_lr_scheduler_factory: LRSchedulerFactory | None = None
    """factory for the creation of a learning rate scheduler to use for the actor network (if any)"""
    critic_lr_scheduler_factory: LRSchedulerFactory | None = None
    """factory for the creation of a learning rate scheduler to use for the critic network (if any)"""

    def _get_param_transformers(self) -> list[ParamTransformer]:
        return [
            ParamTransformerDrop("actor_lr", "critic_lr"),
            ParamTransformerActorAndCriticLRScheduler(
                "actor_lr_scheduler_factory",
                "critic_lr_scheduler_factory",
                "lr_scheduler",
            ),
        ]


@dataclass
class ParamsMixinActionScaling(GetParamTransformersProtocol):
    action_scaling: bool | Literal["default"] = "default"
    """whether to apply action scaling; when set to "default", it will be enabled for continuous action spaces"""
    action_bound_method: Literal["clip", "tanh"] | None = "clip"
    """
    method to bound action to range [-1, 1]. Only used if the action_space is continuous.
    """

    def _get_param_transformers(self) -> list[ParamTransformer]:
        return []


@dataclass
class ParamsMixinExplorationNoise(GetParamTransformersProtocol):
    exploration_noise: BaseNoise | Literal["default"] | NoiseFactory | None = None
    """
    If not None, add noise to actions for exploration.
    This is useful when solving "hard exploration" problems.
    It can either be a distribution, a factory for the creation of a distribution or "default".
    When set to "default", use Gaussian noise with standard deviation 0.1.
    """

    def _get_param_transformers(self) -> list[ParamTransformer]:
        return [ParamTransformerNoiseFactory("exploration_noise")]


@dataclass
class PGParams(Params, ParamsMixinActionScaling, ParamsMixinLearningRateWithScheduler):
    discount_factor: float = 0.99
    """
    discount factor (gamma) for future rewards; must be in [0, 1]
    """
    reward_normalization: bool = False
    """
    if True, will normalize the returns by subtracting the running mean and dividing by the running
    standard deviation.
    """
    deterministic_eval: bool = False
    """
    whether to use deterministic action (the dist's mode) instead of stochastic one during evaluation.
    Does not affect training.
    """
    dist_fn: TDistributionFunction | DistributionFunctionFactory | Literal["default"] = "default"
    """
    This can either be a function which maps the model output to a torch distribution or a
    factory for the creation of such a function.
    When set to "default", a factory which creates Gaussian distributions from mean and standard
    deviation will be used for the continuous case and which creates categorical distributions
    for the discrete case (see :class:`DistributionFunctionFactoryDefault`)
    """

    def _get_param_transformers(self) -> list[ParamTransformer]:
        transformers = super()._get_param_transformers()
        transformers.extend(ParamsMixinActionScaling._get_param_transformers(self))
        transformers.extend(ParamsMixinLearningRateWithScheduler._get_param_transformers(self))
        transformers.append(ParamTransformerActionScaling("action_scaling"))
        transformers.append(ParamTransformerDistributionFunction("dist_fn"))
        return transformers


@dataclass
class ParamsMixinGeneralAdvantageEstimation(GetParamTransformersProtocol):
    gae_lambda: float = 0.95
    """
    determines the blend between Monte Carlo and one-step temporal difference (TD) estimates of the advantage
    function in general advantage estimation (GAE).
    A value of 0 gives a fully TD-based estimate; lambda=1 gives a fully Monte Carlo estimate.
    """
    max_batchsize: int = 256
    """the maximum size of the batch when computing general advantage estimation (GAE)"""

    def _get_param_transformers(self) -> list[ParamTransformer]:
        return []


@dataclass
class A2CParams(PGParams, ParamsMixinGeneralAdvantageEstimation):
    vf_coef: float = 0.5
    """weight (coefficient) of the value loss in the loss function"""
    ent_coef: float = 0.01
    """weight (coefficient) of the entropy loss in the loss function"""
    max_grad_norm: float | None = None
    """maximum norm for clipping gradients in backpropagation"""

    def _get_param_transformers(self) -> list[ParamTransformer]:
        transformers = super()._get_param_transformers()
        transformers.extend(ParamsMixinGeneralAdvantageEstimation._get_param_transformers(self))
        return transformers


@dataclass
class PPOParams(A2CParams):
    eps_clip: float = 0.2
    """
    determines the range of allowed change in the policy during a policy update:
    The ratio between the probabilities indicated by the new and old policy is
    constrained to stay in the interval [1 - eps_clip, 1 + eps_clip].
    Small values thus force the new policy to stay close to the old policy.
    Typical values range between 0.1 and 0.3.
    The optimal epsilon depends on the environment; more stochastic environments may need larger epsilons.
    """
    dual_clip: float | None = None
    """
    determines the lower bound clipping for the probability ratio
    (corresponds to parameter c in arXiv:1912.09729, Equation 5).
    If set to None, dual clipping is not used and the bounds described in parameter eps_clip apply.
    If set to a float value c, the lower bound is changed from 1 - eps_clip to c,
    where c < 1 - eps_clip.
    Setting c > 0 reduces policy oscillation and further stabilizes training.
    Typical values are between 0 and 0.5. Smaller values provide more stability.
    Setting c = 0 yields PPO with only the upper bound.
    """
    value_clip: bool = False
    """
    whether to apply clipping of the predicted value function during policy learning.
    Value clipping discourages large changes in value predictions between updates.
    Inaccurate value predictions can lead to bad policy updates, which can cause training instability.
    Clipping values prevents sporadic large errors from skewing policy updates too much.
    """
    advantage_normalization: bool = True
    """whether to apply per mini-batch advantage normalization."""
    recompute_advantage: bool = False
    """
    whether to recompute advantage every update repeat as described in
    https://arxiv.org/pdf/2006.05990.pdf, Sec. 3.5.
    The original PPO implementation splits the data in each policy iteration
    step into individual transitions and then randomly assigns them to minibatches.
    This makes it impossible to compute advantages as the temporal structure is broken.
    Therefore, the advantages are computed once at the beginning of each policy iteration step and
    then used in minibatch policy and value function optimization.
    This results in higher diversity of data in each minibatch at the cost of
    using slightly stale advantage estimations.
    Enabling this option will, as a remedy to this problem, recompute the advantages at the beginning
    of each pass over the data instead of just once per iteration.
    """


@dataclass
class NPGParams(PGParams, ParamsMixinGeneralAdvantageEstimation):
    optim_critic_iters: int = 5
    """number of times to optimize critic network per update."""
    actor_step_size: float = 0.5
    """step size for actor update in natural gradient direction"""
    advantage_normalization: bool = True
    """whether to do per mini-batch advantage normalization."""

    def _get_param_transformers(self) -> list[ParamTransformer]:
        transformers = super()._get_param_transformers()
        transformers.extend(ParamsMixinGeneralAdvantageEstimation._get_param_transformers(self))
        return transformers


@dataclass
class TRPOParams(NPGParams):
    max_kl: float = 0.01
    """
    maximum KL divergence, used to constrain each actor network update.
    """
    backtrack_coeff: float = 0.8
    """
    coefficient with which to reduce the step size when constraints are not met.
    """
    max_backtracks: int = 10
    """maximum number of times to backtrack in line search when the constraints are not met."""


@dataclass
class ParamsMixinActorAndDualCritics(GetParamTransformersProtocol):
    actor_lr: float = 1e-3
    """the learning rate to use for the actor network"""
    critic1_lr: float = 1e-3
    """the learning rate to use for the first critic network"""
    critic2_lr: float = 1e-3
    """the learning rate to use for the second critic network"""
    actor_lr_scheduler_factory: LRSchedulerFactory | None = None
    """factory for the creation of a learning rate scheduler to use for the actor network (if any)"""
    critic1_lr_scheduler_factory: LRSchedulerFactory | None = None
    """factory for the creation of a learning rate scheduler to use for the first critic network (if any)"""
    critic2_lr_scheduler_factory: LRSchedulerFactory | None = None
    """factory for the creation of a learning rate scheduler to use for the second critic network (if any)"""

    def _get_param_transformers(self) -> list[ParamTransformer]:
        return [
            ParamTransformerDrop("actor_lr", "critic1_lr", "critic2_lr"),
            ParamTransformerActorDualCriticsLRScheduler(
                "actor_lr_scheduler_factory",
                "critic1_lr_scheduler_factory",
                "critic2_lr_scheduler_factory",
                "lr_scheduler",
            ),
        ]


@dataclass
class _SACParams(Params, ParamsMixinActorAndDualCritics):
    tau: float = 0.005
    """controls the contribution of the entropy term in the overall optimization objective,
     i.e. the desired amount of randomness in the optimal policy.
     Higher values mean greater target entropy and therefore more randomness in the policy.
     Lower values mean lower target entropy and therefore a more deterministic policy.
     """
    gamma: float = 0.99
    """discount factor (gamma) for future rewards; must be in [0, 1]"""
    alpha: float | AutoAlphaFactory = 0.2
    """
    controls the relative importance (coefficient) of the entropy term in the loss function.
    This can be a constant or a factory for the creation of a representation that allows the
    parameter to be automatically tuned;
    use :class:`tianshou.highlevel.params.alpha.AutoAlphaFactoryDefault` for the standard
    auto-adjusted alpha.
    """
    estimation_step: int = 1
    """the number of steps to look ahead"""

    def _get_param_transformers(self) -> list[ParamTransformer]:
        transformers = super()._get_param_transformers()
        transformers.extend(ParamsMixinActorAndDualCritics._get_param_transformers(self))
        transformers.append(ParamTransformerAutoAlpha("alpha"))
        return transformers


@dataclass
class SACParams(_SACParams, ParamsMixinExplorationNoise, ParamsMixinActionScaling):
    deterministic_eval: bool = True
    """
    whether to use deterministic action (mean of Gaussian policy) in evaluation mode instead of stochastic
    action sampled by the policy. Does not affect training."""

    def _get_param_transformers(self) -> list[ParamTransformer]:
        transformers = super()._get_param_transformers()
        transformers.extend(ParamsMixinExplorationNoise._get_param_transformers(self))
        transformers.extend(ParamsMixinActionScaling._get_param_transformers(self))
        return transformers


@dataclass
class DiscreteSACParams(_SACParams):
    pass


@dataclass
class DQNParams(Params, ParamsMixinLearningRateWithScheduler):
    discount_factor: float = 0.99
    """
    discount factor (gamma) for future rewards; must be in [0, 1]
    """
    estimation_step: int = 1
    """the number of steps to look ahead"""
    target_update_freq: int = 0
    """the target network update frequency (0 if no target network is to be used)"""
    reward_normalization: bool = False
    """whether to normalize the returns to Normal(0, 1)"""
    is_double: bool = True
    """whether to use double Q learning"""
    clip_loss_grad: bool = False
    """whether to clip the gradient of the loss in accordance with nature14236; this amounts to using the Huber
    loss instead of the MSE loss."""

    def _get_param_transformers(self) -> list[ParamTransformer]:
        transformers = super()._get_param_transformers()
        transformers.extend(ParamsMixinLearningRateWithScheduler._get_param_transformers(self))
        return transformers


@dataclass
class IQNParams(DQNParams):
    sample_size: int = 32
    """the number of samples for policy evaluation"""
    online_sample_size: int = 8
    """the number of samples for online model in training"""
    target_sample_size: int = 8
    """the number of samples for target model in training."""
    num_quantiles: int = 200
    """the number of quantile midpoints in the inverse cumulative distribution function of the value"""
    hidden_sizes: Sequence[int] = ()
    """hidden dimensions to use in the IQN network"""
    num_cosines: int = 64
    """number of cosines to use in the IQN network"""

    def _get_param_transformers(self) -> list[ParamTransformer]:
        transformers = super()._get_param_transformers()
        transformers.append(ParamTransformerDrop("hidden_sizes", "num_cosines"))
        return transformers


@dataclass
class DDPGParams(
    Params,
    ParamsMixinActorAndCritic,
    ParamsMixinExplorationNoise,
    ParamsMixinActionScaling,
):
    tau: float = 0.005
    """
    controls the soft update of the target network.
    It determines how slowly the target networks track the main networks.
    Smaller tau means slower tracking and more stable learning.
    """
    gamma: float = 0.99
    """discount factor (gamma) for future rewards; must be in [0, 1]"""
    estimation_step: int = 1
    """the number of steps to look ahead."""

    def _get_param_transformers(self) -> list[ParamTransformer]:
        transformers = super()._get_param_transformers()
        transformers.extend(ParamsMixinActorAndCritic._get_param_transformers(self))
        transformers.extend(ParamsMixinExplorationNoise._get_param_transformers(self))
        transformers.extend(ParamsMixinActionScaling._get_param_transformers(self))
        return transformers


@dataclass
class REDQParams(DDPGParams):
    ensemble_size: int = 10
    """the number of sub-networks in the critic ensemble"""
    subset_size: int = 2
    """the number of networks in the subset"""
    alpha: float | AutoAlphaFactory = 0.2
    """
    controls the relative importance (coefficient) of the entropy term in the loss function.
    This can be a constant or a factory for the creation of a representation that allows the
    parameter to be automatically tuned;
    use :class:`tianshou.highlevel.params.alpha.AutoAlphaFactoryDefault` for the standard
    auto-adjusted alpha.
    """
    estimation_step: int = 1
    """the number of steps to look ahead"""
    actor_delay: int = 20
    """the number of critic updates before an actor update"""
    deterministic_eval: bool = True
    """
    whether to use deterministic action (the dist's mode) instead of stochastic one during evaluation.
    Does not affect training.
    """
    target_mode: Literal["mean", "min"] = "min"

    def _get_param_transformers(self) -> list[ParamTransformer]:
        transformers = super()._get_param_transformers()
        transformers.append(ParamTransformerAutoAlpha("alpha"))
        return transformers


@dataclass
class TD3Params(
    Params,
    ParamsMixinActorAndDualCritics,
    ParamsMixinExplorationNoise,
    ParamsMixinActionScaling,
):
    tau: float = 0.005
    """
    controls the soft update of the target network.
    It determines how slowly the target networks track the main networks.
    Smaller tau means slower tracking and more stable learning.
    """
    gamma: float = 0.99
    """discount factor (gamma) for future rewards; must be in [0, 1]"""
    policy_noise: float | FloatEnvValueFactory = 0.2
    """the scale of the the noise used in updating policy network"""
    noise_clip: float | FloatEnvValueFactory = 0.5
    """determines the clipping range of the noise used in updating the policy network as [-noise_clip, noise_clip]"""
    update_actor_freq: int = 2
    """the update frequency of actor network"""
    estimation_step: int = 1
    """the number of steps to look ahead."""

    def _get_param_transformers(self) -> list[ParamTransformer]:
        transformers = super()._get_param_transformers()
        transformers.extend(ParamsMixinActorAndDualCritics._get_param_transformers(self))
        transformers.extend(ParamsMixinExplorationNoise._get_param_transformers(self))
        transformers.extend(ParamsMixinActionScaling._get_param_transformers(self))
        transformers.append(ParamTransformerFloatEnvParamFactory("policy_noise"))
        transformers.append(ParamTransformerFloatEnvParamFactory("noise_clip"))
        return transformers
