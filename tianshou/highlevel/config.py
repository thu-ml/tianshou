import logging
import multiprocessing
from dataclasses import dataclass

from sensai.util.string import ToStringMixin

log = logging.getLogger(__name__)


@dataclass(kw_only=True)
class TrainingConfig(ToStringMixin):
    """Training configuration."""

    max_epochs: int = 100
    """
    the (maximum) number of epochs to run training for. An **epoch** is the outermost iteration level and each
    epoch consists of a number of training steps and one test step, where each training step

      * [for the online case] collects environment steps/transitions (**collection step**),
        adding them to the (replay) buffer (see :attr:`collection_step_num_env_steps` and :attr:`collection_step_num_episodes`)
      * performs an **update step** via the RL algorithm being used, which can involve
        one or more actual gradient updates, depending on the algorithm

    and the test step collects :attr:`num_episodes_per_test` test episodes in order to evaluate
    agent performance.

    Training may be stopped early if the stop criterion is met (see :attr:`stop_fn`).

    For online training, the number of training steps in each epoch is indirectly determined by
    :attr:`epoch_num_steps`: As many training steps will be performed as are required in
    order to reach :attr:`epoch_num_steps` total steps in the training environments.
    Specifically, if the number of transitions collected per step is `c` (see
    :attr:`collection_step_num_env_steps`) and :attr:`epoch_num_steps` is set to `s`, then the number
    of training steps per epoch is `ceil(s / c)`.
    Therefore, if `max_epochs = e`, the total number of environment steps taken during training
    can be computed as `e * ceil(s / c) * c`.

    For offline training, the number of training steps per epoch is equal to :attr:`epoch_num_steps`.
    """

    epoch_num_steps: int = 30000
    """
    For an online algorithm, this is the total number of environment steps to be collected per epoch, and,
    for an offline algorithm, it is the total number of training steps to take per epoch.
    See :attr:`max_epochs` for an explanation of epoch semantics.
    """

    num_train_envs: int = -1
    """the number of training environments to use. If set to -1, use number of CPUs/threads."""

    num_test_envs: int = 1
    """the number of test environments to use"""

    test_step_num_episodes: int = 1
    """the total number of episodes to collect in each test step (across all test environments).
    """

    buffer_size: int = 4096
    """the total size of the sample/replay buffer, in which environment steps (transitions) are
    stored"""

    collection_step_num_env_steps: int | None = 2048
    """
    the number of environment steps/transitions to collect in each collection step before the
    network update within each training step.

    This is mutually exclusive with :attr:`collection_step_num_episodes`, and one of the two must be set.

    Note that the exact number can be reached only if this is a multiple of the number of
    training environments being used, as each training environment will produce the same
    (non-zero) number of transitions.
    Specifically, if this is set to `n` and `m` training environments are used, then the total
    number of transitions collected per collection step is `ceil(n / m) * m =: c`.

    See :attr:`max_epochs` for information on the total number of environment steps being
    collected during training.
    """

    collection_step_num_episodes: int | None = None
    """
    the number of episodes to collect in each collection step before the network update within
    each training step. If this is set, the number of environment steps collected in each
    collection step is the sum of the lengths of the episodes collected.

    This is mutually exclusive with :attr:`collection_step_num_env_steps`, and one of the two must be set.
    """

    start_timesteps: int = 0
    """
    the number of environment steps to collect before the actual training loop begins
    """

    start_timesteps_random: bool = False
    """
    whether to use a random policy (instead of the initial or restored policy to be trained)
    when collecting the initial :attr:`start_timesteps` environment steps before training
    """

    replay_buffer_ignore_obs_next: bool = False
    """whether to ignore the `obs_next` field in the collected samples when storing them in the
    buffer and instead use the one-in-the-future of `obs` as the next observation.
    This can be useful for very large observations, like for Atari, in order to save RAM.

    However, setting this to True **may introduce an error** at the last steps of episodes! Should
    only be used in exceptional cases and only when you know what you are doing.
    Currently only used in Atari examples and may be removed in the future!
    """

    replay_buffer_save_only_last_obs: bool = False
    """if True, for the case where the environment outputs stacked frames (e.g. because it
    is using a `FrameStack` wrapper), save only the most recent frame so as not to duplicate
    observations in buffer memory. Specifically, if the environment outputs observations `obs` with
    shape (N, ...), only obs[-1] of shape (...) will be stored.
    Frame stacking with a fixed number of frames can then be recreated at the buffer level by setting
    :attr:`replay_buffer_stack_num`.

    Note: Currently only used in Atari examples and may be removed in the future!
    """

    replay_buffer_stack_num: int = 1
    """
    the number of consecutive environment observations to stack and use as the observation input
    to the agent for each time step. Setting this to a value greater than 1 can help agents learn
    temporal aspects (e.g. velocities of moving objects for which only positions are observed).

    Note: it is recommended to do this stacking on the environment level by using something like
    gymnasium's `FrameStack` instead. Setting this to larger than one in conjunction
    with :attr:`replay_buffer_save_only_last_obs` means that
    stacking will be recreated at the buffer level, which is more memory-efficient.

    Currently only used in Atari examples and may be removed in the future!
    """

    def __post_init__(self) -> None:
        if self.num_train_envs == -1:
            self.num_train_envs = multiprocessing.cpu_count()

        if self.test_step_num_episodes == 0 and self.num_test_envs != 0:
            log.warning(
                f"Number of test episodes is set to 0, "
                f"but number of test environments is ({self.num_test_envs}). "
                f"This can cause unnecessary memory usage.",
            )

        if (
            self.test_step_num_episodes != 0
            and self.test_step_num_episodes % self.num_test_envs != 0
        ):
            log.warning(
                f"Number of test episodes ({self.test_step_num_episodes} "
                f"is not divisible by the number of test environments ({self.num_test_envs}). "
                f"This can cause unnecessary memory usage, it is recommended to adjust this.",
            )

        assert (
            sum(
                [
                    self.collection_step_num_env_steps is not None,
                    self.collection_step_num_episodes is not None,
                ]
            )
            == 1
        ), (
            "Only one of `collection_step_num_env_steps` and `collection_step_num_episodes` can be set.",
        )


@dataclass(kw_only=True)
class OnlineTrainingConfig(TrainingConfig):
    collection_step_num_env_steps: int | None = 2048
    """
    the number of environment steps/transitions to collect in each collection step before the
    network update within each training step.

    This is mutually exclusive with :attr:`collection_step_num_episodes`, and one of the two must be set.

    Note that the exact number can be reached only if this is a multiple of the number of
    training environments being used, as each training environment will produce the same
    (non-zero) number of transitions.
    Specifically, if this is set to `n` and `m` training environments are used, then the total
    number of transitions collected per collection step is `ceil(n / m) * m =: c`.

    See :attr:`max_epochs` for information on the total number of environment steps being
    collected during training.
    """

    collection_step_num_episodes: int | None = None
    """
    the number of episodes to collect in each collection step before the network update within
    each training step. If this is set, the number of environment steps collected in each
    collection step is the sum of the lengths of the episodes collected.

    This is mutually exclusive with :attr:`collection_step_num_env_steps`, and one of the two must be set.
    """

    test_in_train: bool = False
    """
    Whether to apply a test step within a training step depending on the early stopping criterion
    (see :meth:`~tianshou.highlevel.Experiment.with_epoch_stop_callback`) being satisfied based
    on the data collected within the training step.
    Specifically, after each collect step, we check whether the early stopping criterion
    would be satisfied by data we collected (provided that at least one episode was indeed completed, such
    that we can evaluate returns, etc.). If the criterion is satisfied, we perform a full test step
    (collecting :attr:`test_step_num_episodes` episodes in order to evaluate performance), and if the early
    stopping criterion is also satisfied based on the test data, we stop training early.
    """


@dataclass(kw_only=True)
class OnPolicyTrainingConfig(OnlineTrainingConfig):
    batch_size: int | None = 64
    """
    Use mini-batches of this size for gradient updates (causing the gradient to be less accurate,
    a form of regularization).
    Set ``batch_size=None`` for the full buffer that was collected within the training step to be
    used for the gradient update (no mini-batching).
    """

    update_step_num_repetitions: int = 1
    """
    controls, within one update step of an on-policy algorithm, the number of times
    the full collected data is applied for gradient updates, i.e. if the parameter is
    5, then the collected data shall be used five times to update the policy within the same
    update step.
    """


@dataclass(kw_only=True)
class OffPolicyTrainingConfig(OnlineTrainingConfig):
    batch_size: int = 64
    """
    the the number of environment steps/transitions to sample from the buffer for a gradient update.
    """

    update_step_num_gradient_steps_per_sample: float = 1.0
    """
    the number of gradient steps to perform per sample collected (see :attr:`collection_step_num_env_steps`).
    Specifically, if this is set to `u` and the number of samples collected in the preceding
    collection step is `n`, then `round(u * n)` gradient steps will be performed.
    """
