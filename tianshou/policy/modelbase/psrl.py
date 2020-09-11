import numpy as np
from typing import Any, Dict, Union, Optional

from tianshou.policy import BasePolicy
from tianshou.data import Batch, ReplayBuffer


class PSRLModel(object):
    """Implementation of Posterior Sampling Reinforcement Learning Model.

    :param np.ndarray p_prior: dirichlet prior (alphas), with shape
        (n_state, n_action, n_state).
    :param np.ndarray rew_mean_prior: means of the normal priors of rewards,
        with shape (n_state, n_action).
    :param np.ndarray rew_std_prior: standard deviations of the normal priors
        of rewards, with shape (n_state, n_action).
    :param float epsilon: for precision control in value iteration.
    """

    def __init__(
        self,
        trans_count_prior: np.ndarray,
        rew_mean_prior: np.ndarray,
        rew_std_prior: np.ndarray,
        epsilon: float,
    ) -> None:
        self.trans_count = trans_count_prior
        self.n_state, self.n_action = rew_mean_prior.shape
        self.rew_mean = rew_mean_prior
        self.rew_std = rew_std_prior
        self.rew_count = np.full(rew_mean_prior.shape, epsilon)  # no weight
        self.eps = epsilon
        self.policy: Optional[np.ndarray] = None
        self.value = np.zeros(self.n_state)
        self.updated = False

    def observe(
        self,
        trans_count: np.ndarray,
        rew_sum: np.ndarray,
        rew_count: np.ndarray
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
        :param np.ndarray rew_count: the number of rewards, with shape
            (n_state, n_action).
        """
        self.updated = False
        self.trans_count += trans_count
        sum_count = self.rew_count + rew_count
        self.rew_mean = (self.rew_mean * self.rew_count + rew_sum) / sum_count
        self.rew_std *= self.rew_count / sum_count
        self.rew_count = sum_count

    @staticmethod
    def sample_from_prob(trans_count: np.ndarray) -> np.ndarray:
        sample_prob = np.zeros_like(trans_count)
        n_s, n_a = trans_count.shape[:2]
        for i in range(n_s):
            for j in range(n_a):  # numba does not support dirichlet :(
                sample_prob[i][j] = np.random.dirichlet(trans_count[i][j])
        return sample_prob

    def sample_from_rew(self) -> np.ndarray:
        return np.random.normal(self.rew_mean, self.rew_std)

    def solve_policy(self) -> None:
        self.updated = True
        self.policy, self.value = self.value_iteration(
            self.sample_from_prob(self.trans_count),
            self.sample_from_rew(),
            self.eps,
            self.value,
        )

    @staticmethod
    def value_iteration(
        trans_prob: np.ndarray, rew: np.ndarray, eps: float, value: np.ndarray
    ) -> np.ndarray:
        """Value iteration solver for MDPs.

        :param np.ndarray trans_prob: transition probabilities, with shape
            (n_action, n_state, n_state).
        :param np.ndarray rew: rewards, with shape (n_state, n_action).
        :param float eps: for precision control.
        :param np.ndarray value: the initialize value of value array, with
            shape (n_state, ).

        :return: the optimal policy with shape (n_state, ).
        """
        Q = rew + trans_prob.dot(value)
        new_value = Q.max(axis=1)
        while not np.allclose(new_value, value, eps):
            value = new_value
            Q = rew + trans_prob.dot(value)
            new_value = Q.max(axis=1)
        Q += eps * np.random.randn(*Q.shape)
        return Q.argmax(axis=1), new_value

    def __call__(self, obs: np.ndarray, state=None, info=None) -> np.ndarray:
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
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = PSRLModel(
            trans_count_prior, rew_mean_prior, rew_std_prior, epsilon)
        assert 0.0 <= discount_factor <= 1.0, \
            "discount factor should be in [0, 1]"
        self._gamma = discount_factor

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
        return Batch(act=self.model(batch.obs, state=state, info=batch.info))

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ) -> Batch:
        return self.compute_episodic_return(
            batch, gamma=self._gamma, gae_lambda=1.)

    def learn(  # type: ignore
        self, batch: Batch, *args: Any, **kwargs: Any
    ) -> Dict[str, float]:
        n_s, n_a = self.model.n_state, self.model.n_action
        trans_count = np.zeros((n_s, n_a, n_s))
        rew_sum = np.zeros((n_s, n_a))
        rew_count = np.zeros((n_s, n_a))
        act, rew = batch.act, batch.returns
        obs, obs_next = batch.obs, batch.obs_next
        for i in range(len(obs)):
            trans_count[obs[i], act[i], obs_next[i]] += 1
            rew_sum[obs[i], act[i]] += rew[i]
            rew_count[obs[i], act[i]] += 1
            if batch.done[i]:
                if hasattr(batch.info, "TimeLimit.truncated") \
                        and batch.info["TimeLimit.truncated"][i]:
                    continue
                trans_count[obs_next[i], :, obs_next[i]] += 1
                rew_count[obs_next[i], :] += 1
        self.model.observe(trans_count, rew_sum, rew_count)
        return {
            "psrl/rew_mean": self.model.rew_mean.mean(),
            "psrl/rew_std": self.model.rew_std.mean(),
        }
