from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from tianshou.data import Batch, ReplayBuffer, to_numpy
from tianshou.policy import DQNPolicy, QRDQNPolicy
from tianshou.utils.net.discrete import FractionProposalNetwork, FullQuantileFunction


class FQFPolicy(QRDQNPolicy):
    """Implementation of Fully-parameterized Quantile Function. arXiv:1911.02140.

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param FractionProposalNetwork fraction_model: a FractionProposalNetwork for
        proposing fractions/quantiles given state.
    :param torch.optim.Optimizer fraction_optim: a torch.optim for optimizing
        the fraction model above.
    :param float discount_factor: in [0, 1].
    :param int num_fractions: the number of fractions to use. Default to 32.
    :param float ent_coef: the coefficient for entropy loss. Default to 0.
    :param int estimation_step: the number of steps to look ahead. Default to 1.
    :param int target_update_freq: the target network update frequency (0 if
        you do not use the target network).
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).

    .. seealso::

        Please refer to :class:`~tianshou.policy.QRDQNPolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        model: FullQuantileFunction,
        optim: torch.optim.Optimizer,
        fraction_model: FractionProposalNetwork,
        fraction_optim: torch.optim.Optimizer,
        discount_factor: float = 0.99,
        num_fractions: int = 32,
        ent_coef: float = 0.0,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model, optim, discount_factor, num_fractions, estimation_step,
            target_update_freq, reward_normalization, **kwargs
        )
        self.propose_model = fraction_model
        self._ent_coef = ent_coef
        self._fraction_optim = fraction_optim

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]  # batch.obs_next: s_{t+n}
        if self._target:
            result = self(batch, input="obs_next")
            act, fractions = result.act, result.fractions
            next_dist = self(
                batch, model="model_old", input="obs_next", fractions=fractions
            ).logits
        else:
            next_batch = self(batch, input="obs_next")
            act = next_batch.act
            next_dist = next_batch.logits
        next_dist = next_dist[np.arange(len(act)), act, :]
        return next_dist  # shape: [bsz, num_quantiles]

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "model",
        input: str = "obs",
        fractions: Optional[Batch] = None,
        **kwargs: Any,
    ) -> Batch:
        model = getattr(self, model)
        obs = batch[input]
        obs_next = obs.obs if hasattr(obs, "obs") else obs
        if fractions is None:
            (logits, fractions, quantiles_tau), hidden = model(
                obs_next,
                propose_model=self.propose_model,
                state=state,
                info=batch.info
            )
        else:
            (logits, _, quantiles_tau), hidden = model(
                obs_next,
                propose_model=self.propose_model,
                fractions=fractions,
                state=state,
                info=batch.info
            )
        weighted_logits = (fractions.taus[:, 1:] -
                           fractions.taus[:, :-1]).unsqueeze(1) * logits
        q = DQNPolicy.compute_q_value(
            self, weighted_logits.sum(2), getattr(obs, "mask", None)
        )
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q.shape[1]
        act = to_numpy(q.max(dim=1)[1])
        return Batch(
            logits=logits,
            act=act,
            state=hidden,
            fractions=fractions,
            quantiles_tau=quantiles_tau
        )

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._target and self._iter % self._freq == 0:
            self.sync_weight()
        weight = batch.pop("weight", 1.0)
        out = self(batch)
        curr_dist_orig = out.logits
        taus, tau_hats = out.fractions.taus, out.fractions.tau_hats
        act = batch.act
        curr_dist = curr_dist_orig[np.arange(len(act)), act, :].unsqueeze(2)
        target_dist = batch.returns.unsqueeze(1)
        # calculate each element's difference between curr_dist and target_dist
        dist_diff = F.smooth_l1_loss(target_dist, curr_dist, reduction="none")
        huber_loss = (
            dist_diff * (
                tau_hats.unsqueeze(2) -
                (target_dist - curr_dist).detach().le(0.).float()
            ).abs()
        ).sum(-1).mean(1)
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
                [sa_quantile_hats[:, :1], sa_quantiles[:, :-1]], dim=1
            )

            values_2 = sa_quantiles - sa_quantile_hats[:, 1:]
            signs_2 = sa_quantiles < torch.cat(
                [sa_quantiles[:, 1:], sa_quantile_hats[:, -1:]], dim=1
            )

            gradient_of_taus = (
                torch.where(signs_1, values_1, -values_1) +
                torch.where(signs_2, values_2, -values_2)
            )
        fraction_loss = (gradient_of_taus * taus[:, 1:-1]).sum(1).mean()
        # calculate entropy loss
        entropy_loss = out.fractions.entropies.mean()
        fraction_entropy_loss = fraction_loss - self._ent_coef * entropy_loss
        self._fraction_optim.zero_grad()
        fraction_entropy_loss.backward(retain_graph=True)
        self._fraction_optim.step()
        self.optim.zero_grad()
        quantile_loss.backward()
        self.optim.step()
        self._iter += 1
        return {
            "loss": quantile_loss.item() + fraction_entropy_loss.item(),
            "loss/quantile": quantile_loss.item(),
            "loss/fraction": fraction_loss.item(),
            "loss/entropy": entropy_loss.item()
        }
