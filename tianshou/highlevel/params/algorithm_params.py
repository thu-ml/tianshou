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

    def _get_param_transformers(self) -> list[ParamTransformer]:
        return [ParamTransformerActionScaling("action_scaling")]


@dataclass(kw_only=True)
class ParamsMixinActionScalingAndBounding(ParamsMixinActionScaling):
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
class ParamsMixinNStepReturnHorizon:
    n_step_return_horizon: int = 1
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


class OnPolicyAlgorithmParams(
    Params,
    ParamsMixinGamma,
    ParamsMixinActionScalingAndBounding,
    ParamsMixinSingleModel,
    ParamsMixinDeterministicEval,
):
    def _get_param_transformers(self) -> list[ParamTransformer]:
        transformers = super()._get_param_transformers()
        transformers.extend(ParamsMixinActionScalingAndBounding._get_param_transformers(self))
        transformers.extend(ParamsMixinSingleModel._get_param_transformers(self))
        return transformers


@dataclass(kw_only=True)
class ReinforceParams(OnPolicyAlgorithmParams):
    return_standardization: bool = False
    """
    whether to standardize episode returns by subtracting the running mean and
    dividing by the running standard deviation.
    Note that this is known to be detrimental to performance in many cases!
    """


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
class ActorCriticOnPolicyParams(OnPolicyAlgorithmParams):
    return_scaling: bool = False
    """
    flag indicating whether to enable scaling of estimated returns by
    dividing them by their running standard deviation without centering the mean.
    This reduces the magnitude variation of advantages across different episodes while
    preserving their signs and relative ordering.
    The use of running statistics (rather than batch-specific scaling) means that early
    training experiences may be scaled differently than later ones as the statistics evolve.
    When enabled, this improves training stability in environments with highly variable
    reward scales and makes the algorithm less sensitive to learning rate settings.
    However, it may reduce the algorithm's ability to distinguish between episodes with
    different absolute return magnitudes.
    Best used in environments where the relative ordering of actions is more important
    than the absolute scale of returns.
    """


@dataclass(kw_only=True)
class A2CParams(ActorCriticOnPolicyParams, ParamsMixinGeneralAdvantageEstimation):
    vf_coef: float = 0.5
    """
    coefficient that weights the value loss relative to the actor loss in the overall
    loss function.
    Higher values prioritize accurate value function estimation over policy improvement.
    Controls the trade-off between policy optimization and value function fitting.
    Typically set between 0.5 and 1.0 for most actor-critic implementations.
    """
    ent_coef: float = 0.01
    """
    coefficient that weights the entropy bonus relative to the actor loss.
    Controls the exploration-exploitation trade-off by encouraging policy entropy.
    Higher values promote more exploration by encouraging a more uniform action distribution.
    Lower values focus more on exploitation of the current policy's knowledge.
    Typically set between 0.01 and 0.05 for most actor-critic implementations.
    """
    max_grad_norm: float | None = None
    """
    the maximum L2 norm threshold for gradient clipping.
    When not None, gradients will be rescaled using to ensure their L2 norm does not
    exceed this value. This prevents exploding gradients and stabilizes training by
    limiting the magnitude of parameter updates.
    Set to None to disable gradient clipping.
    """

    def _get_param_transformers(self) -> list[ParamTransformer]:
        transformers = super()._get_param_transformers()
        transformers.extend(ParamsMixinGeneralAdvantageEstimation._get_param_transformers(self))
        return transformers


@dataclass(kw_only=True)
class PPOParams(A2CParams):
    eps_clip: float = 0.2
    """
    determines the range of allowed change in the policy during a policy update:
    The ratio of action probabilities indicated by the new and old policy is
    constrained to stay in the interval [1 - eps_clip, 1 + eps_clip].
    Small values thus force the new policy to stay close to the old policy.
    Typical values range between 0.1 and 0.3, the value of 0.2 is recommended
    in the original PPO paper.
    The optimal value depends on the environment; more stochastic environments may
    need larger values.
    """
    dual_clip: float | None = None
    """
    a clipping parameter (denoted as c in the literature) that prevents
    excessive pessimism in policy updates for negative-advantage actions.
    Excessive pessimism occurs when the policy update too strongly reduces the probability
    of selecting actions that led to negative advantages, potentially eliminating useful
    actions based on limited negative experiences.
    When enabled (c > 1), the objective for negative advantages becomes:
    max(min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A), c*A), where min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)
    is the original single-clipping objective determined by `eps_clip`.
    This creates a floor on negative policy gradients, maintaining some probability
    of exploring actions despite initial negative outcomes.
    Larger values (e.g., 2.0 to 5.0) maintain more exploration, while values closer
    to 1.0 provide less protection against pessimistic updates.
    Set to None to disable dual clipping.
    """
    value_clip: bool = False
    """
    flag indicating whether to enable clipping for value function updates.
    When enabled, restricts how much the value function estimate can change from its
    previous prediction, using the same clipping range as the policy updates (eps_clip).
    This stabilizes training by preventing large fluctuations in value estimates,
    particularly useful in environments with high reward variance.
    The clipped value loss uses a pessimistic approach, taking the maximum of the
    original and clipped value errors:
    max((returns - value)², (returns - v_clipped)²)
    Setting to True often improves training stability but may slow convergence.
    Implementation follows the approach mentioned in arXiv:1811.02553v3 Sec. 4.1.
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
class NPGParams(ActorCriticOnPolicyParams, ParamsMixinGeneralAdvantageEstimation):
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
    trust_region_size: float = 0.5
    """
    the parameter delta - a scalar multiplier for policy updates in the natural gradient direction.
    The mathematical meaning is the trust region size, which is the maximum KL divergence
    allowed between the old and new policy distributions.
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
class ParamsMixinAlpha(GetParamTransformersProtocol):
    alpha: float | AutoAlphaFactory = 0.2
    """
    the entropy regularization coefficient, which balances exploration and exploitation.
    This coefficient controls how much the agent values randomness in its policy versus
    pursuing higher rewards.
    Higher values (e.g., 0.5-1.0) strongly encourage exploration by rewarding the agent
    for maintaining diverse action choices, even if this means selecting some lower-value actions.
    Lower values (e.g., 0.01-0.1) prioritize exploitation, allowing the policy to become
    more focused on the highest-value actions.
    A value of 0 would completely remove entropy regularization, potentially leading to
    premature convergence to suboptimal deterministic policies.
    Can be provided as a fixed float (0.2 is a reasonable default) or via a factory
    to support automatic tuning during training.
    """

    def _get_param_transformers(self) -> list[ParamTransformer]:
        return [ParamTransformerAutoAlpha("alpha")]


@dataclass(kw_only=True)
class _SACParams(
    Params,
    ParamsMixinGamma,
    ParamsMixinActorAndDualCritics,
    ParamsMixinNStepReturnHorizon,
    ParamsMixinTau,
    ParamsMixinDeterministicEval,
    ParamsMixinAlpha,
):
    def _get_param_transformers(self) -> list[ParamTransformer]:
        transformers = super()._get_param_transformers()
        transformers.extend(ParamsMixinActorAndDualCritics._get_param_transformers(self))
        transformers.extend(ParamsMixinAlpha._get_param_transformers(self))
        return transformers


@dataclass(kw_only=True)
class SACParams(_SACParams, ParamsMixinExplorationNoise, ParamsMixinActionScaling):
    def _get_param_transformers(self) -> list[ParamTransformer]:
        transformers = super()._get_param_transformers()
        transformers.extend(ParamsMixinExplorationNoise._get_param_transformers(self))
        transformers.extend(ParamsMixinActionScalingAndBounding._get_param_transformers(self))
        return transformers


@dataclass(kw_only=True)
class DiscreteSACParams(_SACParams):
    pass


@dataclass(kw_only=True)
class QLearningOffPolicyParams(
    Params, ParamsMixinGamma, ParamsMixinSingleModel, ParamsMixinNStepReturnHorizon
):
    target_update_freq: int = 0
    """
    the number of training iterations between each complete update of the target network.
    Controls how frequently the target Q-network parameters are updated with the current
    Q-network values.
    A value of 0 disables the target network entirely, using only a single network for both
    action selection and bootstrap targets.
    Higher values provide more stable learning targets but slow down the propagation of new
    value estimates. Lower positive values allow faster learning but may lead to instability
    due to rapidly changing targets.
    Typically set between 100-10000 for DQN variants, with exact values depending on
    environment complexity.
    """
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
    """
    flag indicating whether to use the Double DQN algorithm for target value computation.
    If True, the algorithm uses the online network to select actions and the target network to
    evaluate their Q-values. This approach helps reduce the overestimation bias in Q-learning
    by decoupling action selection from action evaluation.
    If False, the algorithm follows the vanilla DQN method that directly takes the maximum Q-value
    from the target network.
    Note: Double Q-learning will only be effective when a target network is used (target_update_freq > 0).
    """
    huber_loss_delta: float | None = None
    """
    controls whether to use the Huber loss instead of the MSE loss for the TD error and the threshold for
    the Huber loss.
    If None, the MSE loss is used.
    If not None, uses the Huber loss as described in the Nature DQN paper (nature14236) with the given delta,
    which limits the influence of outliers.
    Unlike the MSE loss where the gradients grow linearly with the error magnitude, the Huber
    loss causes the gradients to plateau at a constant value for large errors, providing more stable training.
    NOTE: The magnitude of delta should depend on the scale of the returns obtained in the environment.
    """

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
    ParamsMixinActionScalingAndBounding,
    ParamsMixinNStepReturnHorizon,
    ParamsMixinTau,
):
    def _get_param_transformers(self) -> list[ParamTransformer]:
        transformers = super()._get_param_transformers()
        transformers.extend(ParamsMixinActorAndCritic._get_param_transformers(self))
        transformers.extend(ParamsMixinExplorationNoise._get_param_transformers(self))
        transformers.extend(ParamsMixinActionScalingAndBounding._get_param_transformers(self))
        return transformers


@dataclass(kw_only=True)
class REDQParams(DDPGParams, ParamsMixinDeterministicEval, ParamsMixinAlpha):
    ensemble_size: int = 10
    """
    the total number of critic networks in the ensemble.
    This parameter implements the randomized ensemble approach described in REDQ.
    The algorithm maintains `ensemble_size` different critic networks that all share the same architecture.
    During target value computation, a random subset of these networks (determined by `subset_size`) is used.
    Larger values increase the diversity of the ensemble but require more memory and computation.
    The original paper recommends a value of 10 for most tasks, balancing performance and computational efficiency.
    """
    subset_size: int = 2
    """
    the number of critic networks randomly selected from the ensemble for computing target Q-values.
    During each update, the algorithm samples `subset_size` networks from the ensemble of
    `ensemble_size` networks without replacement.
    The target Q-value is then calculated as either the minimum or mean (based on target_mode)
    of the predictions from this subset.
    Smaller values increase randomization and sample efficiency but may introduce more variance.
    Larger values provide more stable estimates but reduce the benefits of randomization.
    The REDQ paper recommends a value of 2 for optimal sample efficiency.
    Must satisfy 0 < subset_size <= ensemble_size.
    """
    actor_delay: int = 20
    """
    the number of critic updates performed before each actor update.
    The actor network is only updated once for every actor_delay critic updates, implementing
    a delayed policy update strategy similar to TD3.
    Larger values stabilize training by allowing critics to become more accurate before policy updates.
    Smaller values allow the policy to adapt more quickly but may lead to less stable learning.
    The REDQ paper recommends a value of 20 for most tasks.
    """
    target_mode: Literal["mean", "min"] = "min"
    """
    the method used to aggregate Q-values from the subset of critic networks.
    Can be either "min" or "mean".
    If "min", uses the minimum Q-value across the selected subset of critics for each state-action pair.
    If "mean", uses the average Q-value across the selected subset of critics.
    Using "min" helps prevent overestimation bias but may lead to more conservative value estimates.
    Using "mean" provides more optimistic value estimates but may suffer from overestimation bias.
    Default is "min" following the conservative value estimation approach common in recent Q-learning
    algorithms.
    """

    def _get_param_transformers(self) -> list[ParamTransformer]:
        transformers = super()._get_param_transformers()
        transformers.extend(ParamsMixinAlpha._get_param_transformers(self))
        return transformers


@dataclass(kw_only=True)
class TD3Params(
    Params,
    ParamsMixinGamma,
    ParamsMixinActorAndDualCritics,
    ParamsMixinExplorationNoise,
    ParamsMixinActionScalingAndBounding,
    ParamsMixinNStepReturnHorizon,
    ParamsMixinTau,
):
    policy_noise: float | FloatEnvValueFactory = 0.2
    """
    scaling factor for the Gaussian noise added to target policy actions.
    This parameter implements target policy smoothing, a regularization technique described in the TD3 paper.
    The noise is sampled from a normal distribution and multiplied by this value before being added to actions.
    Higher values increase exploration in the target policy, helping to address function approximation error.
    The added noise is optionally clipped to a range determined by the noise_clip parameter.
    Typically set between 0.1 and 0.5 relative to the action scale of the environment.
    """
    noise_clip: float | FloatEnvValueFactory = 0.5
    """
    defines the maximum absolute value of the noise added to target policy actions, i.e. noise values
    are clipped to the range [-noise_clip, noise_clip] (after generating and scaling the noise
    via `policy_noise`).
    This parameter implements bounded target policy smoothing as described in the TD3 paper.
    It prevents extreme noise values from causing unrealistic target values during training.
    Setting it 0.0 (or a negative value) disables clipping entirely.
    It is typically set to about twice the `policy_noise` value (e.g. 0.5 when `policy_noise` is 0.2).
    """
    update_actor_freq: int = 2
    """
    the frequency of actor network updates relative to critic network updates
    (the actor network is only updated once for every `update_actor_freq` critic updates).
    This implements the "delayed" policy updates from the TD3 algorithm, where the actor is
    updated less frequently than the critics.
    Higher values (e.g., 2-5) help stabilize training by allowing the critic to become more
    accurate before updating the policy.
    The default value of 2 follows the original TD3 paper's recommendation of updating the
    policy at half the rate of the Q-functions.
    """

    def _get_param_transformers(self) -> list[ParamTransformer]:
        transformers = super()._get_param_transformers()
        transformers.extend(ParamsMixinActorAndDualCritics._get_param_transformers(self))
        transformers.extend(ParamsMixinExplorationNoise._get_param_transformers(self))
        transformers.extend(ParamsMixinActionScalingAndBounding._get_param_transformers(self))
        transformers.append(ParamTransformerFloatEnvParamFactory("policy_noise"))
        transformers.append(ParamTransformerFloatEnvParamFactory("noise_clip"))
        return transformers
