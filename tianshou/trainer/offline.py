from typing import Any, Callable, Dict, Optional, Union

import numpy as np

from tianshou.data import Collector, ReplayBuffer
from tianshou.policy import BasePolicy
from tianshou.trainer.base import BaseTrainer
from tianshou.utils import BaseLogger, LazyLogger


class OfflineTrainer(BaseTrainer):
    """Create an iterator class for offline training procedure.

    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy` class.
    :param buffer: an instance of the :class:`~tianshou.data.ReplayBuffer` class.
        This buffer must be populated with experiences for offline RL.
    :param Collector test_collector: the collector used for testing. If it's None,
        then no testing will be performed.
    :param int max_epoch: the maximum number of epochs for training. The training
        process might be finished before reaching ``max_epoch`` if ``stop_fn`` is
        set.
    :param int update_per_epoch: the number of policy network updates, so-called
        gradient steps, per epoch.
    :param episode_per_test: the number of episodes for one policy evaluation.
    :param int batch_size: the batch size of sample data, which is going to feed in
        the policy network.
    :param function test_fn: a hook called at the beginning of testing in each
        epoch.
        It can be used to perform custom additional operations, with the signature
        ``f(num_epoch: int, step_idx: int) -> None``.
    :param function save_best_fn: a hook called when the undiscounted average mean
        reward in evaluation phase gets better, with the signature
        ``f(policy: BasePolicy) -> None``. It was ``save_fn`` previously.
    :param function save_checkpoint_fn: a function to save training process and
        return the saved checkpoint path, with the signature ``f(epoch: int,
        env_step: int, gradient_step: int) -> str``; you can save whatever you want.
        Because offline-RL doesn't have env_step, the env_step is always 0 here.
    :param bool resume_from_log: resume gradient_step and other metadata from
        existing tensorboard log. Default to False.
    :param function stop_fn: a function with signature ``f(mean_rewards: float) ->
        bool``, receives the average undiscounted returns of the testing result,
        returns a boolean which indicates whether reaching the goal.
    :param function reward_metric: a function with signature ``f(rewards:
        np.ndarray with shape (num_episode, agent_num)) -> np.ndarray with shape
        (num_episode,)``, used in multi-agent RL. We need to return a single scalar
        for each episode's result to monitor training in the multi-agent RL
        setting. This function specifies what is the desired metric, e.g., the
        reward of agent 1 or the average reward over all agents.
    :param BaseLogger logger: A logger that logs statistics during
        updating/testing. Default to a logger that doesn't log anything.
    :param bool verbose: whether to print the information. Default to True.
    :param bool show_progress: whether to display a progress bar when training.
        Default to True.
    """

    __doc__ = BaseTrainer.gen_doc("offline") + "\n".join(__doc__.split("\n")[1:])

    def __init__(
        self,
        policy: BasePolicy,
        buffer: ReplayBuffer,
        test_collector: Optional[Collector],
        max_epoch: int,
        update_per_epoch: int,
        episode_per_test: int,
        batch_size: int,
        test_fn: Optional[Callable[[int, Optional[int]], None]] = None,
        stop_fn: Optional[Callable[[float], bool]] = None,
        save_best_fn: Optional[Callable[[BasePolicy], None]] = None,
        save_checkpoint_fn: Optional[Callable[[int, int, int], str]] = None,
        resume_from_log: bool = False,
        reward_metric: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        logger: BaseLogger = LazyLogger(),
        verbose: bool = True,
        show_progress: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            learning_type="offline",
            policy=policy,
            buffer=buffer,
            test_collector=test_collector,
            max_epoch=max_epoch,
            update_per_epoch=update_per_epoch,
            step_per_epoch=update_per_epoch,
            episode_per_test=episode_per_test,
            batch_size=batch_size,
            test_fn=test_fn,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            save_checkpoint_fn=save_checkpoint_fn,
            resume_from_log=resume_from_log,
            reward_metric=reward_metric,
            logger=logger,
            verbose=verbose,
            show_progress=show_progress,
            **kwargs,
        )

    def policy_update_fn(
        self, data: Dict[str, Any], result: Optional[Dict[str, Any]] = None
    ) -> None:
        """Perform one off-line policy update."""
        assert self.buffer
        self.gradient_step += 1
        losses = self.policy.update(self.batch_size, self.buffer)
        data.update({"gradient_step": str(self.gradient_step)})
        self.log_update_data(data, losses)


def offline_trainer(*args, **kwargs) -> Dict[str, Union[float, str]]:  # type: ignore
    """Wrapper for offline_trainer run method.

    It is identical to ``OfflineTrainer(...).run()``.

    :return: See :func:`~tianshou.trainer.gather_info`.
    """
    return OfflineTrainer(*args, **kwargs).run()


offline_trainer_iter = OfflineTrainer
