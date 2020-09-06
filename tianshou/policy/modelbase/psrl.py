import numpy as np
from typing import Any, Dict, Union, Optional

from tianshou.data import Batch
from tianshou.policy import BasePolicy


class PSRLModel(object):
    """Implementation of Posterior Sampling Reinforcement Learning Model.

    :param np.ndarray p_prior: dirichlet prior (alphas), with shape
        (n_action, n_state, n_state).
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
        p_prior: np.ndarray,
        rew_mean_prior: np.ndarray,
        rew_std_prior: np.ndarray,
    ) -> None:
        self.__eps = np.finfo(np.float32).eps.item()
        self.p = p_prior
        self.n_action, self.n_state, _ = p_prior.shape
        self.rew_mean = rew_mean_prior
        self.rew_std = rew_std_prior
        self.rew_count = np.zeros_like(rew_mean_prior) + self.__eps
        self.policy: Optional[np.ndarray] = None
        self.updated = False

    def observe(
        self, p: np.ndarray, rew_sum: np.ndarray, rew_count: np.ndarray
    ) -> None:
        """Add data into memory pool.

        For rewards, we have a normal prior at first. After we observed a
        reward for a given state-action pair, we use the mean value of our
        observations instead of the prior mean as the posterior mean. The
        standard deviations are in inverse proportion to the number of the
        corresponding observations.

        :param np.ndarray p: the number of observations, with shape
            (n_action, n_state, n_state).
        :param np.ndarray rew_sum: total rewards, with shape
            (n_state, n_action).
        :param np.ndarray rew_count: the number of rewards, with shape
            (n_state, n_action).
        """
        self.updated = False
        self.p += p
        sum_count = self.rew_count + rew_count
        self.rew_mean = (self.rew_mean * self.rew_count + rew_sum) / sum_count
        self.rew_std *= self.rew_count / sum_count
        self.rew_count = sum_count

    def get_p_ml(self) -> np.ndarray:
        return self.p / np.sum(self.p, axis=-1, keepdims=True)

    def sample_from_rew(self) -> np.ndarray:
        sample_rew = np.random.randn(*self.rew_mean.shape)
        sample_rew = sample_rew * self.rew_std + self.rew_mean
        return sample_rew

    def solve_policy(self) -> None:
        self.updated = True
        self.policy = self.value_iteration(
            self.get_p_ml(), self.sample_from_rew())

    @staticmethod
    def value_iteration(
        p: np.ndarray, rew: np.ndarray, eps: float = 0.01
    ) -> np.ndarray:
        """Value iteration solver for MDPs.

        :param np.ndarray p: transition probabilities, with shape
            (n_action, n_state, n_state).
        :param np.ndarray rew: rewards, with shape (n_state, n_action).
        :param float eps: for precision control.

        :return: the optimal policy with shape (n_state, ).
        """
        value = np.zeros(len(rew))
        while True:
            Q = rew + np.matmul(p, value).T
            new_value = np.max(Q, axis=1)
            if np.allclose(new_value, value, eps):
                return np.argmax(Q, axis=1)
            else:
                value = new_value

    def __call__(self, obs: np.ndarray, state=None, info=None) -> np.ndarray:
        if self.updated is False:
            self.solve_policy()
        return self.policy[obs]


class PSRLPolicy(BasePolicy):
    """Implementation of Posterior Sampling Reinforcement Learning.

    Reference: Strens M. A Bayesian framework for reinforcement learning [C]
    //ICML. 2000, 2000: 943-950.

    :param np.ndarray p_prior: dirichlet prior (alphas).
    :param np.ndarray rew_mean_prior: means of the normal priors of rewards.
    :param np.ndarray rew_std_prior: standard deviations of the normal
        priors of rewards.
    :param torch.distributions.Distribution dist_fn: for computing the action.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        p_prior: np.ndarray,
        rew_mean_prior: np.ndarray,
        rew_std_prior: np.ndarray,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = PSRLModel(p_prior, rew_mean_prior, rew_std_prior)
        self.eps = 0.0

    def set_eps(self, eps: float) -> None:
        """Set the eps for epsilon-greedy exploration."""
        self.eps = eps

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
        act = self.model(batch.obs, state=state, info=batch.info)
        if eps is None:
            eps = self.eps
        if not np.isclose(eps, 0):
            for i in range(len(act)):
                if np.random.rand() < eps:
                    act[i] = np.random.randint(0, self.model.n_action)
        return Batch(act=act)

    def learn(  # type: ignore
        self, batch: Batch, **kwargs: Any
    ) -> Dict[str, float]:
        p = np.zeros((self.model.n_action, self.model.n_state,
                      self.model.n_state))
        rew_sum = np.zeros((self.model.n_state, self.model.n_action))
        rew_count = np.zeros_like(rew_sum)
        a, r = batch.act, batch.returns
        obs, obs_next = batch.obs, batch.obs_next
        for i in range(len(obs)):
            p[a[i]][obs[i]][obs_next[i]] += 1
            rew_sum[obs[i]][a[i]] += r[i]
            rew_count[obs[i]][a[i]] += 1
        self.model.observe(p, rew_sum, rew_count)
        return {}
