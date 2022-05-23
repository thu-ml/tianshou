from typing import Any, Dict, List, Optional, Type

import numpy as np
import torch
import torch.nn.functional as F

from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch
from tianshou.policy import PPOPolicy


class GAILPolicy(PPOPolicy):
    r"""Implementation of Generative Adversarial Imitation Learning. arXiv:1606.03476.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.nn.Module critic: the critic network. (s -> V(s))
    :param torch.optim.Optimizer optim: the optimizer for actor and critic network.
    :param dist_fn: distribution class for computing the action.
    :type dist_fn: Type[torch.distributions.Distribution]
    :param ReplayBuffer expert_buffer: the replay buffer contains expert experience.
    :param torch.nn.Module disc_net: the discriminator network with input dim equals
        state dim plus action dim and output dim equals 1.
    :param torch.optim.Optimizer disc_optim: the optimizer for the discriminator
        network.
    :param int disc_update_num: the number of discriminator grad steps per model grad
        step. Default to 4.
    :param float discount_factor: in [0, 1]. Default to 0.99.
    :param float eps_clip: :math:`\epsilon` in :math:`L_{CLIP}` in the original
        paper. Default to 0.2.
    :param float dual_clip: a parameter c mentioned in arXiv:1912.09729 Equ. 5,
        where c > 1 is a constant indicating the lower bound.
        Default to 5.0 (set None if you do not want to use it).
    :param bool value_clip: a parameter mentioned in arXiv:1811.02553 Sec. 4.1.
        Default to True.
    :param bool advantage_normalization: whether to do per mini-batch advantage
        normalization. Default to True.
    :param bool recompute_advantage: whether to recompute advantage every update
        repeat according to https://arxiv.org/pdf/2006.05990.pdf Sec. 3.5.
        Default to False.
    :param float vf_coef: weight for value loss. Default to 0.5.
    :param float ent_coef: weight for entropy loss. Default to 0.01.
    :param float max_grad_norm: clipping gradients in back propagation. Default to
        None.
    :param float gae_lambda: in [0, 1], param for Generalized Advantage Estimation.
        Default to 0.95.
    :param bool reward_normalization: normalize estimated values to have std close
        to 1, also normalize the advantage to Normal(0, 1). Default to False.
    :param int max_batchsize: the maximum size of the batch when computing GAE,
        depends on the size of available memory and the memory cost of the model;
        should be as large as possible within the memory constraint. Default to 256.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action), "tanh" (for applying tanh
        squashing) for now, or empty string for no bounding. Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).
    :param bool deterministic_eval: whether to use deterministic action instead of
        stochastic action sampled by the policy. Default to False.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).

    .. seealso::

        Please refer to :class:`~tianshou.policy.PPOPolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        optim: torch.optim.Optimizer,
        dist_fn: Type[torch.distributions.Distribution],
        expert_buffer: ReplayBuffer,
        disc_net: torch.nn.Module,
        disc_optim: torch.optim.Optimizer,
        disc_update_num: int = 4,
        eps_clip: float = 0.2,
        dual_clip: Optional[float] = None,
        value_clip: bool = False,
        advantage_normalization: bool = True,
        recompute_advantage: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            actor, critic, optim, dist_fn, eps_clip, dual_clip, value_clip,
            advantage_normalization, recompute_advantage, **kwargs
        )
        self.disc_net = disc_net
        self.disc_optim = disc_optim
        self.disc_update_num = disc_update_num
        self.expert_buffer = expert_buffer
        self.action_dim = actor.output_dim

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        """Pre-process the data from the provided replay buffer.

        Used in :meth:`update`. Check out :ref:`process_fn` for more information.
        """
        # update reward
        with torch.no_grad():
            batch.rew = to_numpy(-F.logsigmoid(-self.disc(batch)).flatten())
        return super().process_fn(batch, buffer, indices)

    def disc(self, batch: Batch) -> torch.Tensor:
        obs = to_torch(batch.obs, device=self.disc_net.device)
        act = to_torch(batch.act, device=self.disc_net.device)
        return self.disc_net(torch.cat([obs, act], dim=1))

    def learn(  # type: ignore
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        # update discriminator
        losses = []
        acc_pis = []
        acc_exps = []
        bsz = len(batch) // self.disc_update_num
        for b in batch.split(bsz, merge_last=True):
            logits_pi = self.disc(b)
            exp_b = self.expert_buffer.sample(bsz)[0]
            logits_exp = self.disc(exp_b)
            loss_pi = -F.logsigmoid(-logits_pi).mean()
            loss_exp = -F.logsigmoid(logits_exp).mean()
            loss_disc = loss_pi + loss_exp
            self.disc_optim.zero_grad()
            loss_disc.backward()
            self.disc_optim.step()
            losses.append(loss_disc.item())
            acc_pis.append((logits_pi < 0).float().mean().item())
            acc_exps.append((logits_exp > 0).float().mean().item())
        # update policy
        res = super().learn(batch, batch_size, repeat, **kwargs)
        res["loss/disc"] = losses
        res["stats/acc_pi"] = acc_pis
        res["stats/acc_exp"] = acc_exps
        return res
