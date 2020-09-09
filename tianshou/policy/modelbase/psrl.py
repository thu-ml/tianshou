import numpy as np
from typing import Any, Dict, Union, Optional

from tianshou.data import Batch
from tianshou.policy import BasePolicy


class PSRLModel(object):
    """Implementation of Posterior Sampling Reinforcement Learning Model.

    :param np.ndarray p_prior: dirichlet prior (alphas), with shape
        (n_state, n_action, n_state).
    :param np.ndarray rew_mean_prior: means of the normal priors of rewards,
        with shape (n_state, n_action).
    :param np.ndarray rew_std_prior: standard deviations of the normal priors
        of rewards, with shape (n_state, n_action).
    """

    def __init__(
        self,
        trans_count_prior: np.ndarray,
        rew_mean_prior: np.ndarray,
        rew_std_prior: np.ndarray,
    ) -> None:
        self.trans_count = trans_count_prior
        self.n_state, self.n_action = rew_mean_prior.shape
        self.rew_mean = rew_mean_prior
        self.rew_std = rew_std_prior
        self.rew_count = np.ones_like(rew_mean_prior)
        self.policy: Optional[np.ndarray] = None
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

    def sample_from_prob(self) -> np.ndarray:
        sample_prob = np.zeros_like(self.trans_count)
        for i in range(self.n_state):
            for j in range(self.n_action):
                sample_prob[i][j] = np.random.dirichlet(
                    self.trans_count[i][j])
        return sample_prob

    def sample_from_rew(self) -> np.ndarray:
        return np.random.normal(self.rew_mean, self.rew_std)

    def solve_policy(self) -> None:
        self.updated = True
        self.policy = self.value_iteration(
            self.sample_from_prob(), self.sample_from_rew())

    @staticmethod
    def value_iteration(
        trans_prob: np.ndarray, rew: np.ndarray, eps: float = 0.01
    ) -> np.ndarray:
        """Value iteration solver for MDPs.

        :param np.ndarray trans_prob: transition probabilities, with shape
            (n_action, n_state, n_state).
        :param np.ndarray rew: rewards, with shape (n_state, n_action).
        :param float eps: for precision control.

        :return: the optimal policy with shape (n_state, ).
        """
        value = np.zeros(len(rew))
        print(trans_prob.shape, value.shape)
        Q = rew + trans_prob.dot(value)  # (s, a) = (s, a) + (s, a, s) * (s)
        new_value = np.max(Q, axis=1)  # (s) = (s, a).max(axis=1)
        while not np.allclose(new_value, value, eps):
            value = new_value
            Q = rew + trans_prob.dot(value)
            new_value = np.max(Q, axis=1)
        return np.argmax(Q, axis=1)

    def __call__(self, obs: np.ndarray, state=None, info=None) -> np.ndarray:
        if self.updated is False:
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

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        trans_count_prior: np.ndarray,
        rew_mean_prior: np.ndarray,
        rew_std_prior: np.ndarray,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = PSRLModel(
            trans_count_prior, rew_mean_prior, rew_std_prior)

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        eps: Optional[float] = None,
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

    def learn(  # type: ignore
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, float]:
        n_s, n_a = self.model.n_state, self.model.n_action
        trans_count = np.zeros((n_s, n_a, n_s))
        rew_sum = np.zeros((n_s, n_a))
        rew_count = np.zeros((n_s, n_a))
        act, rew = batch.act, batch.rew
        obs, obs_next = batch.obs, batch.obs_next
        for i in range(len(obs)):
            trans_count[obs[i]][act[i]][obs_next[i]] += 1
            rew_sum[obs[i]][act[i]] += rew[i]
            rew_count[obs[i]][act[i]] += 1
            if batch.done[i]:
                if hasattr(batch.info, 'TimeLimit.truncated') \
                        and batch.info['TimeLimit.truncated'][i]:
                    continue
                trans_count[obs_next[i], :, obs_next[i]] += 1
        self.model.observe(trans_count, rew_sum, rew_count)
        return {}
