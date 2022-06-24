from typing import Any, Callable, Dict, Optional, Union

import numpy as np

from tianshou.data import Collector
from tianshou.policy import BasePolicy
from tianshou.trainer.base import BaseTrainer
from tianshou.utils import BaseLogger, LazyLogger


class OnpolicyTrainer(BaseTrainer):
    """Create an iterator wrapper for on-policy training procedure.

    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy` class.
    :param Collector train_collector: the collector used for training.
    :param Collector test_collector: the collector used for testing. If it's None,
        then no testing will be performed.
    :param int max_epoch: the maximum number of epochs for training. The training
        process might be finished before reaching ``max_epoch`` if ``stop_fn`` is
        set.
    :param int step_per_epoch: the number of transitions collected per epoch.
    :param int repeat_per_collect: the number of repeat time for policy learning,
        for example, set it to 2 means the policy needs to learn each given batch
        data twice.
    :param int episode_per_test: the number of episodes for one policy evaluation.
    :param int batch_size: the batch size of sample data, which is going to feed in
        the policy network.
    :param int step_per_collect: the number of transitions the collector would
        collect before the network update, i.e., trainer will collect
        "step_per_collect" transitions and do some policy network update repeatedly
        in each epoch.
    :param int episode_per_collect: the number of episodes the collector would
        collect before the network update, i.e., trainer will collect
        "episode_per_collect" episodes and do some policy network update repeatedly
        in each epoch.
    :param function train_fn: a hook called at the beginning of training in each
        epoch. It can be used to perform custom additional operations, with the
        signature ``f(num_epoch: int, step_idx: int) -> None``.
    :param function test_fn: a hook called at the beginning of testing in each
        epoch. It can be used to perform custom additional operations, with the
        signature ``f(num_epoch: int, step_idx: int) -> None``.
    :param function save_best_fn: a hook called when the undiscounted average mean
        reward in evaluation phase gets better, with the signature
        ``f(policy: BasePolicy) -> None``. It was ``save_fn`` previously.
    :param function save_checkpoint_fn: a function to save training process and
        return the saved checkpoint path, with the signature ``f(epoch: int,
        env_step: int, gradient_step: int) -> str``; you can save whatever you want.
    :param bool resume_from_log: resume env_step/gradient_step and other metadata
        from existing tensorboard log. Default to False.
    :param function stop_fn: a function with signature ``f(mean_rewards: float) ->
        bool``, receives the average undiscounted returns of the testing result,
        returns a boolean which indicates whether reaching the goal.
    :param function reward_metric: a function with signature
        ``f(rewards: np.ndarray with shape (num_episode, agent_num)) ->
        np.ndarray with shape (num_episode,)``, used in multi-agent RL.
        We need to return a single scalar for each episode's result to monitor
        training in the multi-agent RL setting. This function specifies what is the
        desired metric, e.g., the reward of agent 1 or the average reward over
        all agents.
    :param BaseLogger logger: A logger that logs statistics during
        training/testing/updating. Default to a logger that doesn't log anything.
    :param bool verbose: whether to print the information. Default to True.
    :param bool show_progress: whether to display a progress bar when training.
        Default to True.
    :param bool test_in_train: whether to test in the training phase. Default to
        True.

    .. note::

        Only either one of step_per_collect and episode_per_collect can be specified.
    """

    __doc__ = BaseTrainer.gen_doc("onpolicy") + "\n".join(__doc__.split("\n")[1:])

    def __init__(
        self,
        policy: BasePolicy,
        train_collector: Collector,
        test_collector: Optional[Collector],
        max_epoch: int,
        step_per_epoch: int,
        repeat_per_collect: int,
        episode_per_test: int,
        batch_size: int,
        step_per_collect: Optional[int] = None,
        episode_per_collect: Optional[int] = None,
        train_fn: Optional[Callable[[int, int], None]] = None,
        test_fn: Optional[Callable[[int, Optional[int]], None]] = None,
        stop_fn: Optional[Callable[[float], bool]] = None,
        save_best_fn: Optional[Callable[[BasePolicy], None]] = None,
        save_checkpoint_fn: Optional[Callable[[int, int, int], str]] = None,
        resume_from_log: bool = False,
        reward_metric: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        logger: BaseLogger = LazyLogger(),
        verbose: bool = True,
        show_progress: bool = True,
        test_in_train: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            learning_type="onpolicy",
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=max_epoch,
            step_per_epoch=step_per_epoch,
            repeat_per_collect=repeat_per_collect,
            episode_per_test=episode_per_test,
            batch_size=batch_size,
            step_per_collect=step_per_collect,
            episode_per_collect=episode_per_collect,
            train_fn=train_fn,
            test_fn=test_fn,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            save_checkpoint_fn=save_checkpoint_fn,
            resume_from_log=resume_from_log,
            reward_metric=reward_metric,
            logger=logger,
            verbose=verbose,
            show_progress=show_progress,
            test_in_train=test_in_train,
            **kwargs,
        )

    def policy_update_fn(
        self, data: Dict[str, Any], result: Optional[Dict[str, Any]] = None
    ) -> None:
        """Perform one on-policy update."""
        assert self.train_collector is not None
        losses = self.policy.update(
            0,
            self.train_collector.buffer,
            batch_size=self.batch_size,
            repeat=self.repeat_per_collect,
        )
        self.train_collector.reset_buffer(keep_statistics=True)
        step = max([1] + [len(v) for v in losses.values() if isinstance(v, list)])
        self.gradient_step += step
        self.log_update_data(data, losses)


def onpolicy_trainer(*args, **kwargs) -> Dict[str, Union[float, str]]:  # type: ignore
    """Wrapper for OnpolicyTrainer run method.

    It is identical to ``OnpolicyTrainer(...).run()``.

    :return: See :func:`~tianshou.trainer.gather_info`.
    """
    return OnpolicyTrainer(*args, **kwargs).run()


onpolicy_trainer_iter = OnpolicyTrainer
