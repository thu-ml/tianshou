import torch
import numpy as np
import torch.nn.functional as F
from typing import Any, Dict

from tianshou.policy import QRDQNPolicy
from tianshou.data import Batch


class IQNPolicy(QRDQNPolicy):
    """Implementation of Implicit Quantile Network. arXiv:1806.06923.

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param float discount_factor: in [0, 1].
    :param int K: the number of samples for policy evaluation. Default to 32.
    :param int N: the number of samples for online model in training. Default to 8.
    :param int N_prime: the number of samples for target model in training.
        Default to 8.
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
        discount_factor: float = 0.99,
        K: int = 32,
        N: int = 8,
        N_prime: int = 8,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, optim, discount_factor, K, estimation_step,
                         target_update_freq, reward_normalization, **kwargs)
        assert K > 1, "K should be greater than 1"
        assert N > 1, "N should be greater than 1"
        assert N_prime > 1, "N_prime should be greater than 1"
        self._K = K  # for policy eval
        self._N = N  # for online model
        self._N_prime = N_prime  # for target model
        # set sample size for online and target model
        self.model.sample_size = self._N
        if self._target:
            self.model_old.sample_size = self._N_prime

    def train(self, mode: bool = True) -> "IQNPolicy":
        self.model.sample_size = self._N if mode else self._K
        super().train(mode)
        return self

    def sync_weight(self) -> None:
        """Synchronize the weight for the target network."""
        self.model_old.load_state_dict(self.model.state_dict())  # type: ignore
        self.model_old.sample_size = self._N_prime

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._target and self._iter % self._freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        weight = batch.pop("weight", 1.0)
        out = self(batch)
        curr_dist, taus = out.logits, out.state
        act = batch.act
        curr_dist = curr_dist[np.arange(len(act)), act, :].unsqueeze(2)
        target_dist = batch.returns.unsqueeze(1)
        # calculate each element's difference between curr_dist and target_dist
        u = F.smooth_l1_loss(target_dist, curr_dist, reduction="none")
        huber_loss = (u * (
            taus.unsqueeze(2) - (target_dist - curr_dist).detach().le(0.).float()
        ).abs()).sum(-1).mean(1)
        loss = (huber_loss * weight).mean()
        # ref: https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/
        # blob/master/fqf_iqn_qrdqn/agent/qrdqn_agent.py L130
        batch.weight = u.detach().abs().sum(-1).mean(1)  # prio-buffer
        loss.backward()
        self.optim.step()
        self._iter += 1
        return {"loss": loss.item()}
