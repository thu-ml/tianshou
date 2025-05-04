from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol

from sensai.util.string import ToStringMixin

from tianshou.exploration import BaseNoise
from tianshou.highlevel.env import Environments
from tianshou.highlevel.module.core import TDevice
from tianshou.highlevel.optim import OptimizerFactoryFactory
from tianshou.highlevel.params.alpha import AutoAlphaFactory
from tianshou.highlevel.params.env_param import EnvValueFactory, FloatEnvValueFactory
from tianshou.highlevel.params.lr_scheduler import LRSchedulerFactoryFactory
from tianshou.highlevel.params.noise import NoiseFactory


@dataclass(kw_only=True)
class ParamTransformerData:
    """Holds data that can be used by `ParamTransformer` instances to perform their transformation.

    The representation contains the superset of all data items that are required by different types of agent factories.
    An agent factory is expected to set only the attributes that are relevant to its parameters.
    """

    envs: Environments
    device: TDevice
    optim_factory_default: OptimizerFactoryFactory


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
    def get(
        d: dict[str, Any],
        key: str,
        drop: bool = False,
        default_factory: Callable[[], Any] | None = None,
    ) -> Any:
        try:
            value = d[key]
        except KeyError as e:
            raise Exception(f"Key not found: '{key}'; available keys: {list(d.keys())}") from e
        if value is None and default_factory is not None:
            value = default_factory()
        if drop:
            del d[key]
        return value


class ParamTransformerDrop(ParamTransformer):
    def __init__(self, *keys: str):
        self.keys = keys

    def transform(self, kwargs: dict[str, Any], data: ParamTransformerData) -> None:
        for k in self.keys:
            del kwargs[k]


class ParamTransformerRename(ParamTransformer):
    def __init__(self, renamed_params: dict[str, str]):
        self.renamed_params = renamed_params

    def transform(self, kwargs: dict[str, Any], data: ParamTransformerData) -> None:
        for old_name, new_name in self.renamed_params.items():
            v = kwargs[old_name]
            del kwargs[old_name]
            kwargs[new_name] = v


class ParamTransformerChangeValue(ParamTransformer):
    def __init__(self, key: str):
        self.key = key

    def transform(self, params: dict[str, Any], data: ParamTransformerData) -> None:
        params[self.key] = self.change_value(params[self.key], data)

    @abstractmethod
    def change_value(self, value: Any, data: ParamTransformerData) -> Any:
        pass


class ParamTransformerOptimFactory(ParamTransformer):
    """Transformer for learning rate scheduler params.

    Transforms a key containing a learning rate scheduler factory (removed) into a key containing
    a learning rate scheduler (added) for the data member `optim`.
    """

    def __init__(
        self,
        key_optim_factory_factory: str,
        key_lr: str,
        key_lr_scheduler_factory_factory: str,
        key_optim_output: str,
    ):
        self.key_optim_factory_factory = key_optim_factory_factory
        self.key_lr = key_lr
        self.key_scheduler_factory = key_lr_scheduler_factory_factory
        self.key_optim_output = key_optim_output

    def transform(self, params: dict[str, Any], data: ParamTransformerData) -> None:
        optim_factory_factory: OptimizerFactoryFactory = self.get(
            params,
            self.key_optim_factory_factory,
            drop=True,
            default_factory=lambda: data.optim_factory_default,
        )
        lr_scheduler_factory_factory: LRSchedulerFactoryFactory | None = self.get(
            params, self.key_scheduler_factory, drop=True
        )
        lr: float = self.get(params, self.key_lr, drop=True)
        optim_factory = optim_factory_factory.create_optimizer_factory(lr)
        if lr_scheduler_factory_factory is not None:
            optim_factory.with_lr_scheduler_factory(
                lr_scheduler_factory_factory.create_lr_scheduler_factory()
            )
        params[self.key_optim_output] = optim_factory


class ParamTransformerAutoAlpha(ParamTransformer):
    def __init__(self, key: str):
        self.key = key

    def transform(self, kwargs: dict[str, Any], data: ParamTransformerData) -> None:
        alpha = self.get(kwargs, self.key)
        if isinstance(alpha, AutoAlphaFactory):
            kwargs[self.key] = alpha.create_auto_alpha(data.envs, data.device)


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


class ParamTransformerActionScaling(ParamTransformerChangeValue):
    def change_value(self, value: Any, data: ParamTransformerData) -> Any:
        if value == "default":
            return data.envs.get_type().is_continuous()
        else:
            return value


class GetParamTransformersProtocol(Protocol):
    def _get_param_transformers(self) -> list[ParamTransformer]:
        pass


@dataclass(kw_only=True)
class Params(GetParamTransformersProtocol, ToStringMixin):
    def create_kwargs(self, data: ParamTransformerData) -> dict[str, Any]:
        params = asdict(self)
        for transformer in self._get_param_transformers():
            transformer.transform(params, data)
        return params

    def _get_param_transformers(self) -> list[ParamTransformer]:
        return []


@dataclass(kw_only=True)
class ParamsMixinSingleModel(GetParamTransformersProtocol):
    optim: OptimizerFactoryFactory | None = None
    """the factory for the creation of the model's optimizer; if None, use default"""
    lr: float = 1e-3
    """the learning rate to use in the gradient-based optimizer"""
    lr_scheduler: LRSchedulerFactoryFactory | None = None
    """factory for the creation of a learning rate scheduler"""

    def _get_param_transformers(self) -> list[ParamTransformer]:
        return [
            ParamTransformerOptimFactory("optim", "lr", "lr_scheduler", "optim"),
        ]


@dataclass(kw_only=True)
class ParamsMixinActorAndCritic(GetParamTransformersProtocol):
    actor_optim: OptimizerFactoryFactory | None = None
    """the factory for the creation of the actor's optimizer; if None, use default"""
    critic_optim: OptimizerFactoryFactory | None = None
    """the factory for the creation of the critic's optimizer; if None, use default"""
    actor_lr: float = 1e-3
    """the learning rate to use for the actor network"""
    critic_lr: float = 1e-3
    """the learning rate to use for the critic network"""
    actor_lr_scheduler: LRSchedulerFactoryFactory | None = None
    """factory for the creation of a learning rate scheduler to use for the actor network (if any)"""
    critic_lr_scheduler: LRSchedulerFactoryFactory | None = None
    """factory for the creation of a learning rate scheduler to use for the critic network (if any)"""

    def _get_param_transformers(self) -> list[ParamTransformer]:
        return [
            ParamTransformerOptimFactory(
                "actor_optim", "actor_lr", "actor_lr_scheduler", "policy_optim"
            ),
            ParamTransformerOptimFactory(
                "critic_optim", "critic_lr", "critic_lr_scheduler", "critic_optim"
            ),
        ]


@dataclass(kw_only=True)
class ParamsMixinActionScaling(GetParamTransformersProtocol):
    action_scaling: bool | Literal["default"] = "default"
    """
    flag indicating whether, for continuous action spaces, actions
    should be scaled from the standard neural network output range [-1, 1] to the
    environment's action space range [action_space.low, action_space.high].
    This applies to continuous action spaces only (gym.spaces.Box) and has no effect
    for discrete spaces.
    When enabled, policy outputs are expected to be in the normalized range [-1, 1]
    (after bounding), and are then linearly transformed to the actual required range.
    This improves neural network training stability, allows the same algorithm to work
    across environments with different action ranges, and standardizes exploration
    strategies.
    Should be disabled if the actor model already produces outputs in the correct range.
    """
    action_bound_method: Literal["clip", "tanh"] | None = "clip"
    """
    the method used for bounding actions in continuous action spaces
    to the range [-1, 1] before scaling them to the environment's action space (provided
    that `action_scaling` is enabled).
    This applies to continuous action spaces only (`gym.spaces.Box`) and should be set to None
    for discrete spaces.
    When set to "clip", actions exceeding the [-1, 1] range are simply clipped to this
    range. When set to "tanh", a hyperbolic tangent function is applied, which smoothly
    constrains outputs to [-1, 1] while preserving gradients.
    The choice of bounding method affects both training dynamics and exploration behavior.
    Clipping provides hard boundaries but may create plateau regions in the gradient
    landscape, while tanh provides smoother transitions but can compress sensitivity
    near the boundaries.
    Should be set to None if the actor model inherently produces bounded outputs.
    Typically used together with `action_scaling=True`.
    """

    def _get_param_transformers(self) -> list[ParamTransformer]:
        return [ParamTransformerActionScaling("action_scaling")]


@dataclass(kw_only=True)
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


@dataclass(kw_only=True)
class ParamsMixinEstimationStep:
    estimation_step: int = 1
    """
    the number of future steps (> 0) to consider when computing temporal difference (TD) targets.
    Controls the balance between TD learning and Monte Carlo methods:
    Higher values reduce bias (by relying less on potentially inaccurate value estimates)
    but increase variance (by incorporating more environmental stochasticity and reducing
    the averaging effect).
    A value of 1 corresponds to standard TD learning with immediate bootstrapping, while very
    large values approach Monte Carlo-like estimation that uses complete episode returns.
    """


@dataclass(kw_only=True)
class ParamsMixinGamma:
    gamma: float = 0.99
    """
    the discount factor in [0, 1] for future rewards.
    This determines how much future rewards are valued compared to immediate ones.
    Lower values (closer to 0) make the agent focus on immediate rewards, creating "myopic"
    behavior. Higher values (closer to 1) make the agent value long-term rewards more,
    potentially improving performance in tasks where delayed rewards are important but
    increasing training variance by incorporating more environmental stochasticity.
    Typically set between 0.9 and 0.99 for most reinforcement learning tasks
    """


@dataclass(kw_only=True)
class ParamsMixinTau:
    tau: float = 0.005
    """
    the soft update coefficient for target networks, controlling the rate at which
    target networks track the learned networks.
    When the parameters of the target network are updated with the current (source) network's
    parameters, a weighted average is used: target = tau * source + (1 - tau) * target.
    Smaller values (closer to 0) create more stable but slower learning as target networks
    change more gradually. Higher values (closer to 1) allow faster learning but may reduce
    stability.
    Typically set to a small value (0.001 to 0.01) for most reinforcement learning tasks.
    """


@dataclass(kw_only=True)
class ParamsMixinDeterministicEval:
    deterministic_eval: bool = False
    """
    flag indicating whether the policy should use deterministic
    actions (using the mode of the action distribution) instead of stochastic ones
    (using random sampling) during evaluation.
    When enabled, the policy will always select the most probable action according to
    the learned distribution during evaluation phases, while still using stochastic
    sampling during training. This creates a clear distinction between exploration
    (training) and exploitation (evaluation) behaviors.
    Deterministic actions are generally preferred for final deployment and reproducible
    evaluation as they provide consistent behavior, reduce variance in performance
    metrics, and are more interpretable for human observers.
    Note that this parameter only affects behavior when the policy is not within a
    training step. When collecting rollouts for training, actions remain stochastic
    regardless of this setting to maintain proper exploration behaviour.
    """


@dataclass(kw_only=True)
class ReinforceParams(
    Params,
    ParamsMixinGamma,
    ParamsMixinActionScaling,
    ParamsMixinSingleModel,
    ParamsMixinDeterministicEval,
):
    reward_normalization: bool = False
    """
    if True, will normalize the returns by subtracting the running mean and dividing by the running
    standard deviation.
    """

    def _get_param_transformers(self) -> list[ParamTransformer]:
        transformers = super()._get_param_transformers()
        transformers.extend(ParamsMixinActionScaling._get_param_transformers(self))
        transformers.extend(ParamsMixinSingleModel._get_param_transformers(self))
        return transformers


@dataclass(kw_only=True)
class ParamsMixinGeneralAdvantageEstimation(GetParamTransformersProtocol):
    gae_lambda: float = 0.95
    """
    the lambda parameter in [0, 1] for generalized advantage estimation (GAE).
    Controls the bias-variance tradeoff in advantage estimates, acting as a
    weighting factor for combining different n-step advantage estimators. Higher values
    (closer to 1) reduce bias but increase variance by giving more weight to longer
    trajectories, while lower values (closer to 0) reduce variance but increase bias
    by relying more on the immediate TD error and value function estimates. At λ=0,
    GAE becomes equivalent to the one-step TD error (high bias, low variance); at λ=1,
    it becomes equivalent to Monte Carlo advantage estimation (low bias, high variance).
    Intermediate values create a weighted average of n-step returns, with exponentially
    decaying weights for longer-horizon returns. Typically set between 0.9 and 0.99 for
    most policy gradient methods.
    """
    max_batchsize: int = 256
    """the maximum number of samples to process at once when computing
    generalized advantage estimation (GAE) and value function predictions.
    Controls memory usage by breaking large batches into smaller chunks processed sequentially.
    Higher values may increase speed but require more GPU/CPU memory; lower values
    reduce memory requirements but may increase computation time. Should be adjusted
    based on available hardware resources and total batch size of your training data."""

    def _get_param_transformers(self) -> list[ParamTransformer]:
        return []


@dataclass(kw_only=True)
class A2CParams(ReinforceParams, ParamsMixinGeneralAdvantageEstimation):
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


@dataclass(kw_only=True)
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


@dataclass(kw_only=True)
class NPGParams(ReinforceParams, ParamsMixinGeneralAdvantageEstimation):
    optim_critic_iters: int = 5
    """
    the number of optimization steps performed on the critic network for each policy (actor) update.
    Controls the learning rate balance between critic and actor.
    Higher values prioritize critic accuracy by training the value function more
    extensively before each policy update, which can improve stability but slow down
    training. Lower values maintain a more even learning pace between policy and value
    function but may lead to less reliable advantage estimates.
    Typically set between 1 and 10, depending on the complexity of the value function.
    """
    actor_step_size: float = 0.5
    """
    the scalar multiplier for policy updates in the natural gradient direction.
    Controls how far the policy parameters move in the calculated direction
    during each update. Higher values allow for faster learning but may cause instability
    or policy deterioration; lower values provide more stable but slower learning. Unlike
    regular policy gradients, natural gradients already account for the local geometry of
    the parameter space, making this step size more robust to different parameterizations.
    Typically set between 0.1 and 1.0 for most reinforcement learning tasks.
    """
    advantage_normalization: bool = True
    """whether to do per mini-batch advantage normalization."""

    def _get_param_transformers(self) -> list[ParamTransformer]:
        transformers = super()._get_param_transformers()
        transformers.extend(ParamsMixinGeneralAdvantageEstimation._get_param_transformers(self))
        return transformers


@dataclass(kw_only=True)
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


@dataclass(kw_only=True)
class ParamsMixinActorAndDualCritics(GetParamTransformersProtocol):
    actor_optim: OptimizerFactoryFactory | None = None
    """the factory for the creation of the actor's optimizer; if None, use default"""
    critic1_optim: OptimizerFactoryFactory | None = None
    """the factory for the creation of the first critic's optimizer; if None, use default"""
    critic2_optim: OptimizerFactoryFactory | None = None
    """the factory for the creation of the second critic's optimizer; if None, use default"""
    actor_lr: float = 1e-3
    """the learning rate to use for the actor network"""
    critic1_lr: float = 1e-3
    """the learning rate to use for the first critic network"""
    critic2_lr: float = 1e-3
    """the learning rate to use for the second critic network"""
    actor_lr_scheduler: LRSchedulerFactoryFactory | None = None
    """factory for the creation of a learning rate scheduler to use for the actor network (if any)"""
    critic1_lr_scheduler: LRSchedulerFactoryFactory | None = None
    """factory for the creation of a learning rate scheduler to use for the first critic network (if any)"""
    critic2_lr_scheduler: LRSchedulerFactoryFactory | None = None
    """factory for the creation of a learning rate scheduler to use for the second critic network (if any)"""

    def _get_param_transformers(self) -> list[ParamTransformer]:
        return [
            ParamTransformerOptimFactory(
                "actor_optim", "actor_lr", "actor_lr_scheduler", "policy_optim"
            ),
            ParamTransformerOptimFactory(
                "critic1_optim", "critic1_lr", "critic1_lr_scheduler", "critic_optim"
            ),
            ParamTransformerOptimFactory(
                "critic2_optim", "critic2_lr", "critic2_lr_scheduler", "critic2_optim"
            ),
        ]


@dataclass(kw_only=True)
class _SACParams(
    Params,
    ParamsMixinGamma,
    ParamsMixinActorAndDualCritics,
    ParamsMixinEstimationStep,
    ParamsMixinTau,
    ParamsMixinDeterministicEval,
):
    alpha: float | AutoAlphaFactory = 0.2
    """
    controls the relative importance (coefficient) of the entropy term in the loss function.
    This can be a constant or a factory for the creation of a representation that allows the
    parameter to be automatically tuned;
    use :class:`tianshou.highlevel.params.alpha.AutoAlphaFactoryDefault` for the standard
    auto-adjusted alpha.
    """

    def _get_param_transformers(self) -> list[ParamTransformer]:
        transformers = super()._get_param_transformers()
        transformers.extend(ParamsMixinActorAndDualCritics._get_param_transformers(self))
        transformers.append(ParamTransformerAutoAlpha("alpha"))
        return transformers


@dataclass(kw_only=True)
class SACParams(_SACParams, ParamsMixinExplorationNoise, ParamsMixinActionScaling):
    def _get_param_transformers(self) -> list[ParamTransformer]:
        transformers = super()._get_param_transformers()
        transformers.extend(ParamsMixinExplorationNoise._get_param_transformers(self))
        transformers.extend(ParamsMixinActionScaling._get_param_transformers(self))
        return transformers


@dataclass(kw_only=True)
class DiscreteSACParams(_SACParams):
    pass


@dataclass(kw_only=True)
class QLearningOffPolicyParams(
    Params, ParamsMixinGamma, ParamsMixinSingleModel, ParamsMixinEstimationStep
):
    target_update_freq: int = 0
    """the target network update frequency (0 if no target network is to be used)"""
    reward_normalization: bool = False
    """whether to normalize the returns to Normal(0, 1)"""
    eps_training: float = 0.0
    """
    the epsilon value for epsilon-greedy exploration during training.
    When collecting data for training, this is the probability of choosing a random action
    instead of the action chosen by the policy.
    A value of 0.0 means no exploration (fully greedy) and a value of 1.0 means full
    exploration (fully random).
    """
    eps_inference: float = 0.0
    """
    the epsilon value for epsilon-greedy exploration during inference,
    i.e. non-training cases (such as evaluation during test steps).
    The epsilon value is the probability of choosing a random action instead of the action
    chosen by the policy.
    A value of 0.0 means no exploration (fully greedy) and a value of 1.0 means full
    exploration (fully random).
    """

    def _get_param_transformers(self) -> list[ParamTransformer]:
        transformers = super()._get_param_transformers()
        transformers.extend(ParamsMixinSingleModel._get_param_transformers(self))
        return transformers


@dataclass(kw_only=True)
class DQNParams(QLearningOffPolicyParams):
    is_double: bool = True
    """whether to use double Q learning"""
    clip_loss_grad: bool = False
    """whether to clip the gradient of the loss in accordance with nature14236; this amounts to using the Huber
    loss instead of the MSE loss."""

    def _get_param_transformers(self) -> list[ParamTransformer]:
        return super()._get_param_transformers()


@dataclass(kw_only=True)
class IQNParams(QLearningOffPolicyParams):
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


@dataclass(kw_only=True)
class DDPGParams(
    Params,
    ParamsMixinGamma,
    ParamsMixinActorAndCritic,
    ParamsMixinExplorationNoise,
    ParamsMixinActionScaling,
    ParamsMixinEstimationStep,
    ParamsMixinTau,
):
    def _get_param_transformers(self) -> list[ParamTransformer]:
        transformers = super()._get_param_transformers()
        transformers.extend(ParamsMixinActorAndCritic._get_param_transformers(self))
        transformers.extend(ParamsMixinExplorationNoise._get_param_transformers(self))
        transformers.extend(ParamsMixinActionScaling._get_param_transformers(self))
        return transformers


@dataclass(kw_only=True)
class REDQParams(DDPGParams, ParamsMixinDeterministicEval):
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
    actor_delay: int = 20
    """the number of critic updates before an actor update"""
    target_mode: Literal["mean", "min"] = "min"

    def _get_param_transformers(self) -> list[ParamTransformer]:
        transformers = super()._get_param_transformers()
        transformers.append(ParamTransformerAutoAlpha("alpha"))
        return transformers


@dataclass(kw_only=True)
class TD3Params(
    Params,
    ParamsMixinGamma,
    ParamsMixinActorAndDualCritics,
    ParamsMixinExplorationNoise,
    ParamsMixinActionScaling,
    ParamsMixinEstimationStep,
    ParamsMixinTau,
):
    policy_noise: float | FloatEnvValueFactory = 0.2
    """the scale of the the noise used in updating policy network"""
    noise_clip: float | FloatEnvValueFactory = 0.5
    """determines the clipping range of the noise used in updating the policy network as [-noise_clip, noise_clip]"""
    update_actor_freq: int = 2
    """the update frequency of actor network"""

    def _get_param_transformers(self) -> list[ParamTransformer]:
        transformers = super()._get_param_transformers()
        transformers.extend(ParamsMixinActorAndDualCritics._get_param_transformers(self))
        transformers.extend(ParamsMixinExplorationNoise._get_param_transformers(self))
        transformers.extend(ParamsMixinActionScaling._get_param_transformers(self))
        transformers.append(ParamTransformerFloatEnvParamFactory("policy_noise"))
        transformers.append(ParamTransformerFloatEnvParamFactory("noise_clip"))
        return transformers
