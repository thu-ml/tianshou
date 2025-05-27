from dataclasses import dataclass
from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch

from tianshou.algorithm.algorithm_base import (
    OnPolicyAlgorithm,
    Policy,
    TrainingStats,
)
from tianshou.data import Batch
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import ActBatchProtocol, ObsBatchProtocol, RolloutBatchProtocol


@dataclass(kw_only=True)
class PSRLTrainingStats(TrainingStats):
    psrl_rew_mean: float = 0.0
    psrl_rew_std: float = 0.0


class PSRLModel:
    """Implementation of Posterior Sampling Reinforcement Learning Model."""

    def __init__(
        self,
        trans_count_prior: np.ndarray,
        rew_mean_prior: np.ndarray,
        rew_std_prior: np.ndarray,
        gamma: float,
        epsilon: float,
    ) -> None:
        """
        :param trans_count_prior: dirichlet prior (alphas), with shape
            (n_state, n_action, n_state).
        :param rew_mean_prior: means of the normal priors of rewards,
            with shape (n_state, n_action).
        :param rew_std_prior: standard deviations of the normal priors
            of rewards, with shape (n_state, n_action).
        :param gamma: the discount factor in [0, 1] for future rewards.
            This determines how much future rewards are valued compared to immediate ones.
            Lower values (closer to 0) make the agent focus on immediate rewards, creating "myopic"
            behavior. Higher values (closer to 1) make the agent value long-term rewards more,
            potentially improving performance in tasks where delayed rewards are important but
            increasing training variance by incorporating more environmental stochasticity.
            Typically set between 0.9 and 0.99 for most reinforcement learning tasks
        :param epsilon: for precision control in value iteration.
        """
        self.trans_count = trans_count_prior
        self.n_state, self.n_action = rew_mean_prior.shape
        self.rew_mean = rew_mean_prior
        self.rew_std = rew_std_prior
        self.rew_square_sum = np.zeros_like(rew_mean_prior)
        self.rew_std_prior = rew_std_prior
        self.gamma = gamma
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

        :param trans_count: the number of observations, with shape
            (n_state, n_action, n_state).
        :param rew_sum: total rewards, with shape
            (n_state, n_action).
        :param rew_square_sum: total rewards' squares, with shape
            (n_state, n_action).
        :param rew_count: the number of rewards, with shape
            (n_state, n_action).
        """
        self.updated = False
        self.trans_count += trans_count
        sum_count = self.rew_count + rew_count
        self.rew_mean = (self.rew_mean * self.rew_count + rew_sum) / sum_count
        self.rew_square_sum += rew_square_sum
        raw_std2 = self.rew_square_sum / sum_count - self.rew_mean**2
        self.rew_std = np.sqrt(
            1 / (sum_count / (raw_std2 + self.__eps) + 1 / self.rew_std_prior**2),
        )
        self.rew_count = sum_count

    def sample_trans_prob(self) -> np.ndarray:
        return torch.distributions.Dirichlet(torch.from_numpy(self.trans_count)).sample().numpy()

    def sample_reward(self) -> np.ndarray:
        return np.random.normal(self.rew_mean, self.rew_std)

    def solve_policy(self) -> None:
        self.updated = True
        self.policy, self.value = self.value_iteration(
            self.sample_trans_prob(),
            self.sample_reward(),
            self.gamma,
            self.eps,
            self.value,
        )

    @staticmethod
    def value_iteration(
        trans_prob: np.ndarray,
        rew: np.ndarray,
        gamma: float,
        eps: float,
        value: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Value iteration solver for MDPs.

        :param trans_prob: transition probabilities, with shape
            (n_state, n_action, n_state).
        :param rew: rewards, with shape (n_state, n_action).
        :param eps: for precision control.
        :param gamma: the discount factor in [0, 1] for future rewards.
            This determines how much future rewards are valued compared to immediate ones.
            Lower values (closer to 0) make the agent focus on immediate rewards, creating "myopic"
            behavior. Higher values (closer to 1) make the agent value long-term rewards more,
            potentially improving performance in tasks where delayed rewards are important but
            increasing training variance by incorporating more environmental stochasticity.
            Typically set between 0.9 and 0.99 for most reinforcement learning tasks
        :param value: the initialize value of value array, with
            shape (n_state, ).

        :return: the optimal policy with shape (n_state, ).
        """
        Q = rew + gamma * trans_prob.dot(value)
        new_value = Q.max(axis=1)
        while not np.allclose(new_value, value, eps):
            value = new_value
            Q = rew + gamma * trans_prob.dot(value)
            new_value = Q.max(axis=1)
        # this is to make sure if Q(s, a1) == Q(s, a2) -> choose a1/a2 randomly
        Q += eps * np.random.randn(*Q.shape)
        return Q.argmax(axis=1), new_value

    def __call__(
        self,
        obs: np.ndarray,
        state: Any = None,
        info: Any = None,
    ) -> np.ndarray:
        if not self.updated:
            self.solve_policy()
        return self.policy[obs]


class PSRLPolicy(Policy):
    def __init__(
        self,
        *,
        trans_count_prior: np.ndarray,
        rew_mean_prior: np.ndarray,
        rew_std_prior: np.ndarray,
        action_space: gym.spaces.Discrete,
        discount_factor: float = 0.99,
        epsilon: float = 0.01,
        observation_space: gym.Space | None = None,
    ) -> None:
        """
        :param trans_count_prior: dirichlet prior (alphas), with shape
            (n_state, n_action, n_state).
        :param rew_mean_prior: means of the normal priors of rewards,
            with shape (n_state, n_action).
        :param rew_std_prior: standard deviations of the normal priors
            of rewards, with shape (n_state, n_action).
        :param action_space: the environment's action_space.
        :param epsilon: for precision control in value iteration.
        :param observation_space: the environment's observation space
        """
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            action_scaling=False,
            action_bound_method=None,
        )
        self.model = PSRLModel(
            trans_count_prior,
            rew_mean_prior,
            rew_std_prior,
            discount_factor,
            epsilon,
        )

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        **kwargs: Any,
    ) -> ActBatchProtocol:
        """Compute action over the given batch data with PSRL model.

        :return: A :class:`~tianshou.data.Batch` with "act" key containing
            the action.
        """
        assert isinstance(batch.obs, np.ndarray), "only support np.ndarray observation"
        # TODO: shouldn't the model output a state as well if state is passed (i.e. RNNs are involved)?
        act = self.model(batch.obs, state=state, info=batch.info)
        return cast(ActBatchProtocol, Batch(act=act))


class PSRL(OnPolicyAlgorithm[PSRLPolicy]):
    """Implementation of Posterior Sampling Reinforcement Learning (PSRL).

    Reference: Strens M., A Bayesian Framework for Reinforcement Learning, ICML, 2000.
    """

    def __init__(
        self,
        *,
        policy: PSRLPolicy,
        add_done_loop: bool = False,
    ) -> None:
        """
        :param policy: the policy
        :param add_done_loop: a flag indicating whether to add a self-loop transition for terminal states
            in the MDP.
            If True, whenever an episode terminates, an artificial transition from the terminal state
            back to itself is added to the transition counts for all actions.
            This modification can help stabilize learning for terminal states that have limited samples.
            Setting to True can be beneficial in environments where episodes frequently terminate,
            ensuring that terminal states receive sufficient updates to their value estimates.
            Default is False, which preserves the standard MDP formulation without artificial self-loops.
        """
        super().__init__(
            policy=policy,
        )
        self._add_done_loop = add_done_loop

    def _update_with_batch(
        self, batch: RolloutBatchProtocol, batch_size: int | None, repeat: int
    ) -> PSRLTrainingStats:
        # NOTE: In contrast to other on-policy algorithms, this algorithm ignores
        #   the batch_size and repeat arguments.
        #   PSRL, being a Bayesian approach, updates its posterior distribution of
        #   the MDP parameters based on the collected transition data as a whole,
        #   rather than performing gradient-based updates that benefit from mini-batching.
        n_s, n_a = self.policy.model.n_state, self.policy.model.n_action
        trans_count = np.zeros((n_s, n_a, n_s))
        rew_sum = np.zeros((n_s, n_a))
        rew_square_sum = np.zeros((n_s, n_a))
        rew_count = np.zeros((n_s, n_a))
        for minibatch in batch.split(size=1):
            obs, act, obs_next = minibatch.obs, minibatch.act, minibatch.obs_next
            obs_next = cast(np.ndarray, obs_next)
            assert not isinstance(obs, BatchProtocol), "Observations cannot be Batches here"
            trans_count[obs, act, obs_next] += 1
            rew_sum[obs, act] += minibatch.rew
            rew_square_sum[obs, act] += minibatch.rew**2
            rew_count[obs, act] += 1
            if self._add_done_loop and minibatch.done:
                # special operation for terminal states: add a self-loop
                trans_count[obs_next, :, obs_next] += 1
                rew_count[obs_next, :] += 1
        self.policy.model.observe(trans_count, rew_sum, rew_square_sum, rew_count)

        return PSRLTrainingStats(
            psrl_rew_mean=float(self.policy.model.rew_mean.mean()),
            psrl_rew_std=float(self.policy.model.rew_std.mean()),
        )
