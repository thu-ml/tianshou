import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Type

from tianshou.data import Batch
from tianshou.policy import A2CPolicy
from tianshou.utils import KFACOptimizer


class ACKTRPolicy(A2CPolicy):
    """Implementation of Actor Critic using Kronecker-Factored Trust Region, \
        arXiv:1708.05144.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.nn.Module critic: the critic network. (s -> V(s))
    :param torch.optim.Optimizer optim: the optimizer for actor and critic network.
    :param dist_fn: distribution class for computing the action.
    :type dist_fn: Type[torch.distributions.Distribution]
    :param bool advantage_normalization: whether to do per mini-batch advantage
        normalization. Default to True.
    :param float gae_lambda: in [0, 1], param for Generalized Advantage Estimation.
        Default to 0.95.
    :param bool reward_normalization: normalize estimated values to have std close to
        1. Default to False.
    :param int max_batchsize: the maximum size of the batch when computing GAE,
        depends on the size of available memory and the memory cost of the
        model; should be as large as possible within the memory constraint.
        Default to 256.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action), "tanh" (for applying tanh
        squashing) for now, or empty string for no bounding. Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        optim: KFACOptimizer,
        dist_fn: Type[torch.distributions.Distribution],
        advantage_normalization: bool = True,
        **kwargs: Any,
    ) -> None:
        assert isinstance(optim, KFACOptimizer)
        super().__init__(actor, critic, optim, dist_fn, **kwargs)
        del self._grad_norm
        self.optim: KFACOptimizer
        self._norm_adv = advantage_normalization

    def learn(  # type: ignore
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        losses, actor_losses, vf_losses, ent_losses = [], [], [], []
        for _ in range(repeat):
            for b in batch.split(batch_size, merge_last=True):
                # calculate loss for actor
                dist = self(b).dist
                # print(dist.mean[0][0], dist.stddev[0][0])
                # print(self.ret_rms.var)
                if self._norm_adv and False:
                    mean, std = b.adv.mean(), b.adv.std()
                    b.adv = (b.adv - mean) / std  # per-batch norm
                log_prob = dist.log_prob(b.act).reshape(len(b.adv), -1).transpose(0, 1)
                actor_loss = -(log_prob * b.adv).mean()
                # calculate loss for critic
                value = self.critic(b.obs).flatten()
                vf_loss = F.mse_loss(b.returns, value)
                # calculate regularization and overall loss
                ent_loss = dist.entropy().mean()
                loss = actor_loss + self._weight_vf * vf_loss \
                    - self._weight_ent * ent_loss
                if self.optim.steps % self.optim.Ts == 0:
                    # Compute fisher, see Martens 2014
                    self.optim.model.zero_grad()
                    pg_fisher_loss = -log_prob.mean()
                    value_noise = torch.randn(value.size(), device=value.device)
                    sample_value = value + value_noise
                    vf_fisher_loss = -(value - sample_value.detach()).pow(2).mean()
                    fisher_loss = pg_fisher_loss + vf_fisher_loss
                    self.optim.acc_stats = True
                    fisher_loss.backward(retain_graph=True)
                    self.optim.acc_stats = False
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                actor_losses.append(actor_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
                losses.append(loss.item())
        # update learning rate if lr_scheduler is given
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {
            "loss": losses,
            "loss/actor": actor_losses,
            "loss/vf": vf_losses,
            "loss/ent": ent_losses,
        }
