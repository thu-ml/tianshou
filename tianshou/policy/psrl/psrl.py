import torch
import numpy as np
from typing import Dict, List, Union, Optional
import mdptoolbox

from tianshou.policy import BasePolicy
from tianshou.data import Batch, ReplayBuffer, to_torch_as


class PSRLModel(object):
    """Implementation of Posterior Sampling Reinforcement Learning Model.

    :param np.ndarray p_prior: dirichlet prior (alphas).
    :param np.ndarray rew_mean_prior: means of the normal priors of rewards.
    :param np.ndarray rew_std_prior: standard deviations of the normal priors of rewards.
    :param float discount_factor: in [0, 1].

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(self, p_prior, rew_mean_prior, rew_std_prior, discount_factor: float = 0.99):
        self.p = p_prior
        self.n_action = len(self.p)
        self.n_state = len(self.p[0])
        self.rew_mean = rew_mean_prior
        self.rew_std = rew_std_prior
        self.discount_factor = discount_factor
        self.rew_count = np.zeros_like(rew_mean_prior)
        self.sample_p = None
        self.sample_rew = None
        self.policy = None
        self.updated = False

    def observe(self, p, rew_sum, rew_count):
        """Add data."""
        self.updated = False
        self.p += p
        sum_count_nonzero = np.where(self.rew_count + rew_count == 0, 1, self.rew_count + rew_count)
        rew_count_nonzero = np.where(rew_count == 0, 1, rew_count)
        self.rew_mean = np.where(self.rew_count == 0,
                                 np.where(rew_count == 0, self.rew_mean, rew_sum / rew_count_nonzero),
                                 (self.rew_mean * self.rew_count + rew_sum) / sum_count_nonzero)
        self.rew_std *= np.where(self.rew_count == 0, 1, self.rew_count) / sum_count_nonzero
        self.rew_count += rew_count

    def sample_from_p(self):
        sample_p = []
        for a in range(self.n_action):
            for i in range(self.n_state):
                param = self.p[a][i] + 1e-5 * np.random.randn(len(self.p[a][i]))
                sample_p.append(param / np.sum(param))
        sample_p = np.array(sample_p).reshape(self.n_action, self.n_state, self.n_state)
        return sample_p

    def sample_from_rew(self):
        sample_rew = np.random.randn(len(self.rew_mean), len(self.rew_mean[0]))
        sample_rew = sample_rew * self.rew_std + self.rew_mean
        return sample_rew

    def solve_policy(self):
        self.updated = True
        self.sample_p = self.sample_from_p()
        self.sample_rew = self.sample_from_rew()
        problem = mdptoolbox.mdp.ValueIteration(self.sample_p, self.sample_rew, self.discount_factor)
        problem.run()
        self.policy = np.array(problem.policy)
        return self.policy

    def __call__(self, obs, state=None, info=None):
        if self.updated is False:
            self.solve_policy()
        act = self.policy[obs]
        return act


class PSRLPolicy(BasePolicy):
    """Implementation of Posterior Sampling Reinforcement Learning.

    :param PSRLModel model: a model following the rules in
        :class:`PSRLModel`.
    :param torch.distributions.Distribution dist_fn: for computing the action.
    :param float discount_factor: in [0, 1].
    :param bool reward_normalization: normalize the reward to Normal(0, 1),
        defaults to ``False``.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(self,
                 model: PSRLModel,
                 dist_fn: torch.distributions.Distribution
                 = torch.distributions.Categorical,
                 discount_factor: float = 0.2,
                 reward_normalization: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.dist_fn = dist_fn
        assert 0 <= discount_factor <= 1, 'discount factor should in [0, 1]'
        self._gamma = discount_factor
        self._rew_norm = reward_normalization
        self.eps = 0

    def set_eps(self, eps: float) -> None:
        """Set the eps for epsilon-greedy exploration."""
        self.eps = eps

    def process_fn(self, batch: Batch, buffer: ReplayBuffer,
                   indice: np.ndarray) -> Batch:
        r"""Compute the discounted returns for each frame:

        .. math::
            G_t = \sum_{i=t}^T \gamma^{i-t}r_i

        , where :math:`T` is the terminal time step, :math:`\gamma` is the
        discount factor, :math:`\gamma \in [0, 1]`.
        """
        # batch.returns = self._vanilla_returns(batch)
        # batch.returns = self._vectorized_returns(batch)
        # return batch
        return self.compute_episodic_return(
               batch, gamma=self._gamma, gae_lambda=1.)

    def forward(self, batch: Batch,
                state: Optional[Union[dict, Batch, np.ndarray]] = None,
                eps: Optional[float] = None,
                **kwargs) -> Batch:
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which has 1 key:

            * ``act`` the action.

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

    def learn(self, batch: Batch, **kwargs) -> Dict[str, List[float]]:
        r = batch.returns
        if self._rew_norm and not np.isclose(r.std(), 0):
            batch.returns = (r - r.mean()) / r.std()
        p = np.zeros((self.model.n_action, self.model.n_state, self.model.n_state))
        rew_sum = np.zeros((self.model.n_state, self.model.n_action))
        rew_count = np.zeros((self.model.n_state, self.model.n_action))
        a = batch.act
        r = batch.returns
        obs = batch.obs
        obs_next = batch.obs_next
        for i in range(len(obs)):
            p[a[i]][obs[i]][obs_next[i]] += 1
            rew_sum[obs[i]][a[i]] += r[i]
            rew_count[obs[i]][a[i]] += 1
        self.model.observe(p, rew_sum, rew_count)
        return {'loss': [0.0]}


if __name__ == "__main__":
    n_action = 3
    n_state = 4
    p_pri = np.ones((n_action, n_state, n_state))
    rew_mean = np.ones((n_state, n_action))
    rew_std = np.ones((n_state, n_action))
    model = PSRLModel(p_pri, rew_mean, rew_std)
    policy = PSRLPolicy(model)
    import pdb
    pdb.set_trace()
