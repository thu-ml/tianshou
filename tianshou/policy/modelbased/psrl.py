from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from tianshou.data import Batch
from tianshou.policy import BasePolicy


class PSRLModel(object):
    """Implementation of Posterior Sampling Reinforcement Learning Model.

    :param np.ndarray trans_count_prior: dirichlet prior (alphas), with shape
        (n_state, n_action, n_state).
    :param np.ndarray rew_mean_prior: means of the normal priors of rewards,
        with shape (n_state, n_action).
    :param np.ndarray rew_std_prior: standard deviations of the normal priors
        of rewards, with shape (n_state, n_action).
    :param float discount_factor: in [0, 1].
    :param float epsilon: for precision control in value iteration.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).
    """

    def __init__(
        self,
        trans_count_prior: np.ndarray,
        rew_mean_prior: np.ndarray,
        rew_std_prior: np.ndarray,
        discount_factor: float,
        epsilon: float,
    ) -> None:
        self.trans_count = trans_count_prior
        self.n_state, self.n_action = rew_mean_prior.shape
        self.rew_mean = rew_mean_prior
        self.rew_std = rew_std_prior
        self.rew_square_sum = np.zeros_like(rew_mean_prior)
        self.rew_std_prior = rew_std_prior
        self.discount_factor = discount_factor
        self.rew_count = np.full(rew_mean_prior.shape, epsilon)  # no weight
        self.eps = epsilon
        self.policy: np.ndarray
        self.value = np.zeros(self.n_state)
        self.updated = False
        self.__eps = np.finfo(np.float32).eps.item()

    def observe(
        self,
        trans_count: np.ndarray,
        rew_sum: np.ndarray,
        rew_square_sum: np.ndarray,
        rew_count: np.ndarray,
    ) -> None:
        """Add data into memory pool.

        For rewards, we have a normal prior at first. After we observed a
        reward for a given state-action pair, we use the mean value of our
        observations instead of the prior mean as the posterior mean. The
        standard deviations are in inverse proportion to the number of the
        corresponding observations.

        :param np.ndarray trans_count: the number of observations, with shape
            (n_state, n_action, n_state).
        :param np.ndarray rew_sum: total rewards, with shape
            (n_state, n_action).
        :param np.ndarray rew_square_sum: total rewards' squares, with shape
            (n_state, n_action).
        :param np.ndarray rew_count: the number of rewards, with shape
            (n_state, n_action).
        """
        self.updated = False
        self.trans_count += trans_count
        sum_count = self.rew_count + rew_count
        self.rew_mean = (self.rew_mean * self.rew_count + rew_sum) / sum_count
        self.rew_square_sum += rew_square_sum
        raw_std2 = self.rew_square_sum / sum_count - self.rew_mean**2
        self.rew_std = np.sqrt(
            1 / (sum_count / (raw_std2 + self.__eps) + 1 / self.rew_std_prior**2)
        )
        self.rew_count = sum_count

    def sample_trans_prob(self) -> np.ndarray:
        sample_prob = torch.distributions.Dirichlet(
            torch.from_numpy(self.trans_count)
        ).sample().numpy()
        return sample_prob

    def sample_reward(self) -> np.ndarray:
        return np.random.normal(self.rew_mean, self.rew_std)

    def solve_policy(self) -> None:
        self.updated = True
        self.policy, self.value = self.value_iteration(
            self.sample_trans_prob(),
            self.sample_reward(),
            self.discount_factor,
            self.eps,
            self.value,
        )

    @staticmethod
    def value_iteration(
        trans_prob: np.ndarray,
        rew: np.ndarray,
        discount_factor: float,
        eps: float,
        value: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Value iteration solver for MDPs.

        :param np.ndarray trans_prob: transition probabilities, with shape
            (n_state, n_action, n_state).
        :param np.ndarray rew: rewards, with shape (n_state, n_action).
        :param float eps: for precision control.
        :param float discount_factor: in [0, 1].
        :param np.ndarray value: the initialize value of value array, with
            shape (n_state, ).

        :return: the optimal policy with shape (n_state, ).
        """
        Q = rew + discount_factor * trans_prob.dot(value)
        new_value = Q.max(axis=1)
        while not np.allclose(new_value, value, eps):
            value = new_value
            Q = rew + discount_factor * trans_prob.dot(value)
            new_value = Q.max(axis=1)
        # this is to make sure if Q(s, a1) == Q(s, a2) -> choose a1/a2 randomly
        Q += eps * np.random.randn(*Q.shape)
        return Q.argmax(axis=1), new_value

    def __call__(
        self,
        obs: np.ndarray,
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> np.ndarray:
        if not self.updated:
            self.solve_policy()
        return self.policy[obs]


class PSRLPolicy(BasePolicy):
    """Implementation of Posterior Sampling Reinforcement Learning.

    Reference: Strens M. A Bayesian framework for reinforcement learning [C]
    //ICML. 2000, 2000: 943-950.

    :param np.ndarray trans_count_prior: dirichlet prior (alphas), with shape
        (n_state, n_action, n_state).
    :param np.ndarray rew_mean_prior: means of the normal priors of rewards,
        with shape (n_state, n_action).
    :param np.ndarray rew_std_prior: standard deviations of the normal priors
        of rewards, with shape (n_state, n_action).
    :param float discount_factor: in [0, 1].
    :param float epsilon: for precision control in value iteration.
    :param bool add_done_loop: whether to add an extra self-loop for the
        terminal state in MDP. Default to False.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        trans_count_prior: np.ndarray,
        rew_mean_prior: np.ndarray,
        rew_std_prior: np.ndarray,
        discount_factor: float = 0.99,
        epsilon: float = 0.01,
        add_done_loop: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        assert (0.0 <= discount_factor <= 1.0), "discount factor should be in [0, 1]"
        self.model = PSRLModel(
            trans_count_prior, rew_mean_prior, rew_std_prior, discount_factor, epsilon
        )
        self._add_done_loop = add_done_loop

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data with PSRL model.

        :return: A :class:`~tianshou.data.Batch` with "act" key containing
            the action.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        act = self.model(batch.obs, state=state, info=batch.info)
        return Batch(act=act)

    def learn(self, batch: Batch, *args: Any, **kwargs: Any) -> Dict[str, float]:
        n_s, n_a = self.model.n_state, self.model.n_action
        trans_count = np.zeros((n_s, n_a, n_s))
        rew_sum = np.zeros((n_s, n_a))
        rew_square_sum = np.zeros((n_s, n_a))
        rew_count = np.zeros((n_s, n_a))
        for minibatch in batch.split(size=1):
            obs, act, obs_next = minibatch.obs, minibatch.act, minibatch.obs_next
            trans_count[obs, act, obs_next] += 1
            rew_sum[obs, act] += minibatch.rew
            rew_square_sum[obs, act] += minibatch.rew**2
            rew_count[obs, act] += 1
            if self._add_done_loop and minibatch.done:
                # special operation for terminal states: add a self-loop
                trans_count[obs_next, :, obs_next] += 1
                rew_count[obs_next, :] += 1
        self.model.observe(trans_count, rew_sum, rew_square_sum, rew_count)
        return {
            "psrl/rew_mean": float(self.model.rew_mean.mean()),
            "psrl/rew_std": float(self.model.rew_std.mean()),
        }
