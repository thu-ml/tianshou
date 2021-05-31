import torch
import numpy as np
import torch.nn.functional as F
from typing import Any, Dict, Optional, Union

from tianshou.policy import QRDQNPolicy
from tianshou.data import Batch, to_numpy


class FQFPolicy(QRDQNPolicy):
    """Implementation of Fully-parameterized Quantile Function. arXiv:1911.02140.

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param float discount_factor: in [0, 1].
    :param int num_fractions: the number of fractions to use. Default to 32.
    :param float ent_coef: the coefficient for entropy loss. Default to 0.
    :param int estimation_step: the number of steps to look ahead. Default to 1.
    :param int target_update_freq: the target network update frequency (0 if
        you do not use the target network).
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.

    .. seealso::

        Please refer to :class:`~tianshou.policy.QRDQNPolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
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
        self._ent_coef = ent_coef
        self._fraction_optim = fraction_optim

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "model",
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        model = getattr(self, model)
        obs = batch[input]
        obs_ = obs.obs if hasattr(obs, "obs") else obs
        (logits, taus, tau_hats, quantiles, quantiles_tau, entropies), h = model(
            obs_, state=state, info=batch.info
        )
        q = self.compute_q_value(logits, getattr(obs, "mask", None))
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q.shape[1]
        act = to_numpy(q.max(dim=1)[1])
        return Batch(
            logits=logits, act=act, state=h, taus=taus, tau_hats=tau_hats,
            quantiles=quantiles, quantiles_tau=quantiles_tau, entropies=entropies
        )

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._target and self._iter % self._freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        self._fraction_optim.zero_grad()
        weight = batch.pop("weight", 1.0)
        out = self(batch)
        curr_dist_orig = out.logits
        taus, tau_hats = out.taus, out.tau_hats
        act = batch.act
        curr_dist = curr_dist_orig[np.arange(len(act)), act, :].unsqueeze(2)
        target_dist = batch.returns.unsqueeze(1)
        # calculate each element's difference between curr_dist and target_dist
        u = F.smooth_l1_loss(target_dist, curr_dist, reduction="none")
        huber_loss = (u * (
            tau_hats.unsqueeze(2) - (target_dist - curr_dist).detach().le(0.).float()
        ).abs()).sum(-1).mean(1)
        quantile_loss = (huber_loss * weight).mean()
        # ref: https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/
        # blob/master/fqf_iqn_qrdqn/agent/qrdqn_agent.py L130
        batch.weight = u.detach().abs().sum(-1).mean(1)  # prio-buffer
        # calculate fraction loss
        sa_quantile_hats = out.quantiles[np.arange(len(act)), act, :]
        sa_quantiles = out.quantiles_tau[np.arange(len(act)), act, :]
        # ref: https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/
        # blob/master/fqf_iqn_qrdqn/agent/fqf_agent.py L169
        values_1 = sa_quantiles - sa_quantile_hats[:, :-1]
        signs_1 = sa_quantiles > torch.cat([
            sa_quantile_hats[:, :1], sa_quantiles[:, :-1]], dim=1)
        assert values_1.shape == signs_1.shape

        values_2 = sa_quantiles - sa_quantile_hats[:, 1:]
        signs_2 = sa_quantiles < torch.cat([
            sa_quantiles[:, 1:], sa_quantile_hats[:, -1:]], dim=1)
        assert values_2.shape == signs_2.shape

        gradient_of_taus = (
            torch.where(signs_1, values_1, -values_1)
            + torch.where(signs_2, values_2, -values_2)
        ).view(taus.shape[0], taus.shape[1] - 2)
        fraction_loss = (gradient_of_taus.detach() * taus[:, 1:-1]).sum(1).mean()
        # calculate entropy loss
        entropy_loss = out.entropies.mean()
        fraction_entropy_loss = fraction_loss - self._ent_coef * entropy_loss
        fraction_entropy_loss.backward(retain_graph=True)
        self._fraction_optim.step()
        quantile_loss.backward()
        self.optim.step()
        self._iter += 1
        return {
            "loss": quantile_loss.item() + fraction_entropy_loss.item(),
            "loss/quantile": quantile_loss.item(),
            "loss/fraction": fraction_loss.item(),
            "loss/entropy": entropy_loss.item()
        }
