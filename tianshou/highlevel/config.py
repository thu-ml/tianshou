import logging
import multiprocessing
from dataclasses import dataclass

from sensai.util.string import ToStringMixin

log = logging.getLogger(__name__)


@dataclass
class SamplingConfig(ToStringMixin):
    """Configuration of sampling, epochs, parallelization, buffers, collectors, and batching."""

    num_epochs: int = 100
    """
    the number of epochs to run training for. An epoch is the outermost iteration level and each
    epoch consists of a number of training steps and a test step, where each training step

      * collects environment steps/transitions (collection step), adding them to the (replay)
        buffer (see :attr:`step_per_collect`)
      * performs one or more gradient updates (see :attr:`update_per_step`),

    and the test step collects :attr:`num_episodes_per_test` test episodes in order to evaluate
    agent performance.

    The number of training steps in each epoch is indirectly determined by
    :attr:`step_per_epoch`: As many training steps will be performed as are required in
    order to reach :attr:`step_per_epoch` total steps in the training environments.
    Specifically, if the number of transitions collected per step is `c` (see
    :attr:`step_per_collect`) and :attr:`step_per_epoch` is set to `s`, then the number
    of training steps per epoch is `ceil(s / c)`.

    Therefore, if `num_epochs = e`, the total number of environment steps taken during training
    can be computed as `e * ceil(s / c) * c`.
    """

    step_per_epoch: int = 30000
    """
    the total number of environment steps to be made per epoch. See :attr:`num_epochs` for
    an explanation of epoch semantics.
    """

    batch_size: int | None = 64
    """for off-policy algorithms, this is the number of environment steps/transitions to sample
    from the buffer for a gradient update; for on-policy algorithms, its use is algorithm-specific.
    On-policy algorithms use the full buffer that was collected in the preceding collection step
    but they may use this parameter to perform the gradient update using mini-batches of this size
    (causing the gradient to be less accurate, a form of regularization).

    ``batch_size=None`` means that the full buffer is used for the gradient update. This doesn't
    make much sense for off-policy algorithms and is not recommended then. For on-policy or offline algorithms,
    this means that the full buffer is used for the gradient update (no mini-batching), and
    may make sense in some cases.
    """

    num_train_envs: int = -1
    """the number of training environments to use. If set to -1, use number of CPUs/threads."""

    train_seed: int = 42
    """the seed to use for the training environments."""

    num_test_envs: int = 1
    """the number of test environments to use"""

    num_test_episodes: int = 1
    """the total number of episodes to collect in each test step (across all test environments).
    """

    buffer_size: int = 4096
    """the total size of the sample/replay buffer, in which environment steps (transitions) are
    stored"""

    step_per_collect: int | None = 2048
    """
    the number of environment steps/transitions to collect in each collection step before the
    network update within each training step.

    This is mutually exclusive with :attr:`episode_per_collect`, and one of the two must be set.

    Note that the exact number can be reached only if this is a multiple of the number of
    training environments being used, as each training environment will produce the same
    (non-zero) number of transitions.
    Specifically, if this is set to `n` and `m` training environments are used, then the total
    number of transitions collected per collection step is `ceil(n / m) * m =: c`.

    See :attr:`num_epochs` for information on the total number of environment steps being
    collected during training.
    """

    episode_per_collect: int | None = None
    """
    the number of episodes to collect in each collection step before the network update within
    each training step. If this is set, the number of environment steps collected in each
    collection step is the sum of the lengths of the episodes collected.

    This is mutually exclusive with :attr:`step_per_collect`, and one of the two must be set.
    """

    repeat_per_collect: int | None = 1
    """
    controls, within one gradient update step of an on-policy algorithm, the number of times an
    actual gradient update is applied using the full collected dataset, i.e. if the parameter is
    5, then the collected data shall be used five times to update the policy within the same
    training step.

    The parameter is ignored and may be set to None for off-policy and offline algorithms.
    """

    update_per_step: float = 1.0
    """
    for off-policy algorithms only: the number of gradient steps to perform per sample
    collected (see :attr:`step_per_collect`).
    Specifically, if this is set to `u` and the number of samples collected in the preceding
    collection step is `n`, then `round(u * n)` gradient steps will be performed.

    Note that for on-policy algorithms, only a single gradient update is usually performed,
    because thereafter, the samples no longer reflect the behavior of the updated policy.
    To change the number of gradient updates for an on-policy algorithm, use parameter
    :attr:`repeat_per_collect` instead.
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

    @property
    def test_seed(self) -> int:
        return self.train_seed + self.num_train_envs

    def __post_init__(self) -> None:
        if self.num_train_envs == -1:
            self.num_train_envs = multiprocessing.cpu_count()

        if self.num_test_episodes == 0 and self.num_test_envs != 0:
            log.warning(
                f"Number of test episodes is set to 0, "
                f"but number of test environments is ({self.num_test_envs}). "
                f"This can cause unnecessary memory usage.",
            )

        if self.num_test_episodes != 0 and self.num_test_episodes % self.num_test_envs != 0:
            log.warning(
                f"Number of test episodes ({self.num_test_episodes} "
                f"is not divisible by the number of test environments ({self.num_test_envs}). "
                f"This can cause unnecessary memory usage, it is recommended to adjust this.",
            )

        assert (
            sum([self.step_per_collect is not None, self.episode_per_collect is not None]) == 1
        ), ("Only one of `step_per_collect` and `episode_per_collect` can be set.",)
