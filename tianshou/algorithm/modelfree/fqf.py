from dataclasses import dataclass
from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from overrides import override

from tianshou.algorithm import QRDQN, Algorithm
from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
from tianshou.algorithm.modelfree.qrdqn import QRDQNPolicy
from tianshou.algorithm.modelfree.reinforce import SimpleLossTrainingStats
from tianshou.algorithm.optim import OptimizerFactory
from tianshou.data import Batch, ReplayBuffer, to_numpy
from tianshou.data.types import FQFBatchProtocol, ObsBatchProtocol, RolloutBatchProtocol
from tianshou.utils.net.discrete import FractionProposalNetwork, FullQuantileFunction


@dataclass(kw_only=True)
class FQFTrainingStats(SimpleLossTrainingStats):
    quantile_loss: float
    fraction_loss: float
    entropy_loss: float


class FQFPolicy(QRDQNPolicy):
    def __init__(
        self,
        *,
        model: FullQuantileFunction,
        fraction_model: FractionProposalNetwork,
        action_space: gym.spaces.Space,
        observation_space: gym.Space | None = None,
        eps_training: float = 0.0,
        eps_inference: float = 0.0,
    ):
        """
        :param model: a model following the rules (s_B -> action_values_BA)
        :param fraction_model: a FractionProposalNetwork for
            proposing fractions/quantiles given state.
        :param action_space: the environment's action space
        :param observation_space: the environment's observation space.
        :param eps_training: the epsilon value for epsilon-greedy exploration during training.
            When collecting data for training, this is the probability of choosing a random action
            instead of the action chosen by the policy.
            A value of 0.0 means no exploration (fully greedy) and a value of 1.0 means full
            exploration (fully random).
        :param eps_inference: the epsilon value for epsilon-greedy exploration during inference,
            i.e. non-training cases (such as evaluation during test steps).
            The epsilon value is the probability of choosing a random action instead of the action
            chosen by the policy.
            A value of 0.0 means no exploration (fully greedy) and a value of 1.0 means full
            exploration (fully random).
        """
        assert isinstance(action_space, gym.spaces.Discrete)
        super().__init__(
            model=model,
            action_space=action_space,
            observation_space=observation_space,
            eps_training=eps_training,
            eps_inference=eps_inference,
        )
        self.fraction_model = fraction_model

    def forward(  # type: ignore
        self,
        batch: ObsBatchProtocol,
        state: dict | Batch | np.ndarray | None = None,
        model: FullQuantileFunction | None = None,
        fractions: Batch | None = None,
        **kwargs: Any,
    ) -> FQFBatchProtocol:
        if model is None:
            model = self.model
        obs = batch.obs
        # TODO: this is convoluted! See also other places where this is done
        obs_next = obs.obs if hasattr(obs, "obs") else obs
        if fractions is None:
            (logits, fractions, quantiles_tau), hidden = model(
                obs_next,
                propose_model=self.fraction_model,
                state=state,
                info=batch.info,
            )
        else:
            (logits, _, quantiles_tau), hidden = model(
                obs_next,
                propose_model=self.fraction_model,
                fractions=fractions,
                state=state,
                info=batch.info,
            )
        weighted_logits = (fractions.taus[:, 1:] - fractions.taus[:, :-1]).unsqueeze(1) * logits
        q = DiscreteQLearningPolicy.compute_q_value(
            self, weighted_logits.sum(2), getattr(obs, "mask", None)
        )
        act = to_numpy(q.max(dim=1)[1])
        result = Batch(
            logits=logits,
            act=act,
            state=hidden,
            fractions=fractions,
            quantiles_tau=quantiles_tau,
        )
        return cast(FQFBatchProtocol, result)


class FQF(QRDQN[FQFPolicy]):
    """Implementation of Fully Parameterized Quantile Function for Distributional Reinforcement Learning. arXiv:1911.02140."""

    def __init__(
        self,
        *,
        policy: FQFPolicy,
        optim: OptimizerFactory,
        fraction_optim: OptimizerFactory,
        gamma: float = 0.99,
        # TODO: used as num_quantiles in QRDQNPolicy, but num_fractions in FQFPolicy.
        #  Rename? Or at least explain what happens here.
        num_fractions: int = 32,
        ent_coef: float = 0.0,
        n_step_return_horizon: int = 1,
        target_update_freq: int = 0,
    ) -> None:
        """
        :param policy: the policy
        :param optim: the optimizer factory for the policy's main Q-function model
        :param fraction_optim: the optimizer factory for the policy's fraction model
        :param action_space: the environment's action space.
        :param gamma: the discount factor in [0, 1] for future rewards.
            This determines how much future rewards are valued compared to immediate ones.
            Lower values (closer to 0) make the agent focus on immediate rewards, creating "myopic"
            behavior. Higher values (closer to 1) make the agent value long-term rewards more,
            potentially improving performance in tasks where delayed rewards are important but
            increasing training variance by incorporating more environmental stochasticity.
            Typically set between 0.9 and 0.99 for most reinforcement learning tasks
        :param num_fractions: the number of fractions to use.
        :param ent_coef: coefficient that weights the entropy bonus relative to the actor loss.
            Controls the exploration-exploitation trade-off by encouraging policy entropy.
            Higher values promote more exploration by encouraging a more uniform action distribution.
            Lower values focus more on exploitation of the current policy's knowledge.
            Typically set between 0.01 and 0.05 for most actor-critic implementations.
        :param n_step_return_horizon: the number of future steps (> 0) to consider when computing temporal
            difference (TD) targets. Controls the balance between TD learning and Monte Carlo methods:
            higher values reduce bias (by relying less on potentially inaccurate value estimates)
            but increase variance (by incorporating more environmental stochasticity and reducing
            the averaging effect). A value of 1 corresponds to standard TD learning with immediate
            bootstrapping, while very large values approach Monte Carlo-like estimation that uses
            complete episode returns.
        :param target_update_freq: the number of training iterations between each complete update of
            the target network.
            Controls how frequently the target Q-network parameters are updated with the current
            Q-network values.
            A value of 0 disables the target network entirely, using only a single network for both
            action selection and bootstrap targets.
            Higher values provide more stable learning targets but slow down the propagation of new
            value estimates. Lower positive values allow faster learning but may lead to instability
            due to rapidly changing targets.
            Typically set between 100-10000 for DQN variants, with exact values depending on environment
            complexity.
        """
        super().__init__(
            policy=policy,
            optim=optim,
            gamma=gamma,
            num_quantiles=num_fractions,
            n_step_return_horizon=n_step_return_horizon,
            target_update_freq=target_update_freq,
        )
        self.ent_coef = ent_coef
        self.fraction_optim = self._create_optimizer(self.policy.fraction_model, fraction_optim)

    @override
    def _create_policy_optimizer(self, optim: OptimizerFactory) -> Algorithm.Optimizer:
        # Override to leave out the fraction model (use main model only), as we want
        # to use a separate optimizer for the fraction model
        return self._create_optimizer(self.policy.model, optim)

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        obs_next_batch = Batch(
            obs=buffer[indices].obs_next,
            info=[None] * len(indices),
        )  # obs_next: s_{t+n}
        if self.use_target_network:
            result = self.policy(obs_next_batch)
            act, fractions = result.act, result.fractions
            next_dist = self.policy(
                obs_next_batch, model=self.model_old, fractions=fractions
            ).logits
        else:
            next_batch = self.policy(obs_next_batch)
            act = next_batch.act
            next_dist = next_batch.logits
        return next_dist[np.arange(len(act)), act, :]

    def _update_with_batch(
        self,
        batch: RolloutBatchProtocol,
    ) -> FQFTrainingStats:
        self._periodically_update_lagged_network_weights()
        weight = batch.pop("weight", 1.0)
        out = self.policy(batch)
        curr_dist_orig = out.logits
        taus, tau_hats = out.fractions.taus, out.fractions.tau_hats
        act = batch.act
        curr_dist = curr_dist_orig[np.arange(len(act)), act, :].unsqueeze(2)
        target_dist = batch.returns.unsqueeze(1)
        # calculate each element's difference between curr_dist and target_dist
        dist_diff = F.smooth_l1_loss(target_dist, curr_dist, reduction="none")
        huber_loss = (
            (
                dist_diff
                * (tau_hats.unsqueeze(2) - (target_dist - curr_dist).detach().le(0.0).float()).abs()
            )
            .sum(-1)
            .mean(1)
        )
        quantile_loss = (huber_loss * weight).mean()
        # ref: https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/
        # blob/master/fqf_iqn_qrdqn/agent/qrdqn_agent.py L130
        batch.weight = dist_diff.detach().abs().sum(-1).mean(1)  # prio-buffer
        # calculate fraction loss
        with torch.no_grad():
            sa_quantile_hats = curr_dist_orig[np.arange(len(act)), act, :]
            sa_quantiles = out.quantiles_tau[np.arange(len(act)), act, :]
            # ref: https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/
            # blob/master/fqf_iqn_qrdqn/agent/fqf_agent.py L169
            values_1 = sa_quantiles - sa_quantile_hats[:, :-1]
            signs_1 = sa_quantiles > torch.cat(
                [sa_quantile_hats[:, :1], sa_quantiles[:, :-1]],
                dim=1,
            )

            values_2 = sa_quantiles - sa_quantile_hats[:, 1:]
            signs_2 = sa_quantiles < torch.cat(
                [sa_quantiles[:, 1:], sa_quantile_hats[:, -1:]],
                dim=1,
            )

            gradient_of_taus = torch.where(signs_1, values_1, -values_1) + torch.where(
                signs_2,
                values_2,
                -values_2,
            )
        fraction_loss = (gradient_of_taus * taus[:, 1:-1]).sum(1).mean()
        # calculate entropy loss
        entropy_loss = out.fractions.entropies.mean()
        fraction_entropy_loss = fraction_loss - self.ent_coef * entropy_loss
        self.fraction_optim.step(fraction_entropy_loss, retain_graph=True)
        self.optim.step(quantile_loss)

        return FQFTrainingStats(
            loss=quantile_loss.item() + fraction_entropy_loss.item(),
            quantile_loss=quantile_loss.item(),
            fraction_loss=fraction_loss.item(),
            entropy_loss=entropy_loss.item(),
        )
