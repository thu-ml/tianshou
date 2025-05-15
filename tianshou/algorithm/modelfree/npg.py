from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import kl_divergence

from tianshou.algorithm.algorithm_base import TrainingStats
from tianshou.algorithm.modelfree.a2c import ActorCriticOnPolicyAlgorithm
from tianshou.algorithm.modelfree.pg import ActorPolicyProbabilistic
from tianshou.algorithm.optim import OptimizerFactory
from tianshou.data import ReplayBuffer, SequenceSummaryStats, to_torch_as
from tianshou.data.types import BatchWithAdvantagesProtocol, RolloutBatchProtocol
from tianshou.utils.net.continuous import ContinuousCritic
from tianshou.utils.net.discrete import DiscreteCritic


@dataclass(kw_only=True)
class NPGTrainingStats(TrainingStats):
    actor_loss: SequenceSummaryStats
    vf_loss: SequenceSummaryStats
    kl: SequenceSummaryStats


class NPG(ActorCriticOnPolicyAlgorithm):
    """Implementation of Natural Policy Gradient.

    https://proceedings.neurips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf
    """

    def __init__(
        self,
        *,
        policy: ActorPolicyProbabilistic,
        critic: torch.nn.Module | ContinuousCritic | DiscreteCritic,
        optim: OptimizerFactory,
        optim_critic_iters: int = 5,
        actor_step_size: float = 0.5,
        advantage_normalization: bool = True,
        gae_lambda: float = 0.95,
        max_batchsize: int = 256,
        gamma: float = 0.99,
        return_scaling: bool = False,
    ) -> None:
        """
        :param policy: the policy containing the actor network.
        :param critic: the critic network. (s -> V(s))
        :param optim: the optimizer factory for the critic network.
        :param optim_critic_iters: the number of optimization steps performed on the critic network
            for each policy (actor) update.
            Controls the learning rate balance between critic and actor.
            Higher values prioritize critic accuracy by training the value function more
            extensively before each policy update, which can improve stability but slow down
            training. Lower values maintain a more even learning pace between policy and value
            function but may lead to less reliable advantage estimates.
            Typically set between 1 and 10, depending on the complexity of the value function.
        :param actor_step_size: the scalar multiplier for policy updates in the natural gradient direction.
            Controls how far the policy parameters move in the calculated direction
            during each update. Higher values allow for faster learning but may cause instability
            or policy deterioration; lower values provide more stable but slower learning. Unlike
            regular policy gradients, natural gradients already account for the local geometry of
            the parameter space, making this step size more robust to different parameterizations.
            Typically set between 0.1 and 1.0 for most reinforcement learning tasks.
        :param advantage_normalization: whether to do per mini-batch advantage
            normalization.
        :param gae_lambda: the lambda parameter in [0, 1] for generalized advantage estimation (GAE).
            Controls the bias-variance tradeoff in advantage estimates, acting as a
            weighting factor for combining different n-step advantage estimators. Higher values
            (closer to 1) reduce bias but increase variance by giving more weight to longer
            trajectories, while lower values (closer to 0) reduce variance but increase bias
            by relying more on the immediate TD error and value function estimates. At λ=0,
            GAE becomes equivalent to the one-step TD error (high bias, low variance); at λ=1,
            it becomes equivalent to Monte Carlo advantage estimation (low bias, high variance).
            Intermediate values create a weighted average of n-step returns, with exponentially
            decaying weights for longer-horizon returns. Typically set between 0.9 and 0.99 for
            most policy gradient methods.
        :param max_batchsize: the maximum number of samples to process at once when computing
            generalized advantage estimation (GAE) and value function predictions.
            Controls memory usage by breaking large batches into smaller chunks processed sequentially.
            Higher values may increase speed but require more GPU/CPU memory; lower values
            reduce memory requirements but may increase computation time. Should be adjusted
            based on available hardware resources and total batch size of your training data.
        :param gamma: the discount factor in [0, 1] for future rewards.
            This determines how much future rewards are valued compared to immediate ones.
            Lower values (closer to 0) make the agent focus on immediate rewards, creating "myopic"
            behavior. Higher values (closer to 1) make the agent value long-term rewards more,
            potentially improving performance in tasks where delayed rewards are important but
            increasing training variance by incorporating more environmental stochasticity.
            Typically set between 0.9 and 0.99 for most reinforcement learning tasks
        :param return_scaling: flag indicating whether to enable scaling of estimated returns by
            dividing them by their running standard deviation without centering the mean.
            This reduces the magnitude variation of advantages across different episodes while
            preserving their signs and relative ordering.
            The use of running statistics (rather than batch-specific scaling) means that early
            training experiences may be scaled differently than later ones as the statistics evolve.
            When enabled, this improves training stability in environments with highly variable
            reward scales and makes the algorithm less sensitive to learning rate settings.
            However, it may reduce the algorithm's ability to distinguish between episodes with
            different absolute return magnitudes.
            Best used in environments where the relative ordering of actions is more important
            than the absolute scale of returns.
        """
        super().__init__(
            policy=policy,
            critic=critic,
            optim=optim,
            optim_include_actor=False,
            gae_lambda=gae_lambda,
            max_batchsize=max_batchsize,
            gamma=gamma,
            return_scaling=return_scaling,
        )
        self.norm_adv = advantage_normalization
        self.optim_critic_iters = optim_critic_iters
        self.actor_step_size = actor_step_size
        # adjusts Hessian-vector product calculation for numerical stability
        self._damping = 0.1

    def _preprocess_batch(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> BatchWithAdvantagesProtocol:
        batch = self._add_returns_and_advantages(batch, buffer, indices)
        batch.act = to_torch_as(batch.act, batch.v_s)
        old_log_prob = []
        with torch.no_grad():
            for minibatch in batch.split(self.max_batchsize, shuffle=False, merge_last=True):
                old_log_prob.append(self.policy(minibatch).dist.log_prob(minibatch.act))
        batch.logp_old = torch.cat(old_log_prob, dim=0)
        if self.norm_adv:
            batch.adv = (batch.adv - batch.adv.mean()) / batch.adv.std()
        return batch

    def _update_with_batch(  # type: ignore[override]
        self,
        batch: BatchWithAdvantagesProtocol,
        batch_size: int | None,
        repeat: int,
    ) -> NPGTrainingStats:
        actor_losses, vf_losses, kls = [], [], []
        split_batch_size = batch_size or -1
        for _ in range(repeat):
            for minibatch in batch.split(split_batch_size, merge_last=True):
                # optimize actor
                # direction: calculate villia gradient
                dist = self.policy(minibatch).dist
                log_prob = dist.log_prob(minibatch.act)
                log_prob = log_prob.reshape(log_prob.size(0), -1).transpose(0, 1)
                actor_loss = -(log_prob * minibatch.adv).mean()
                flat_grads = self._get_flat_grad(
                    actor_loss, self.policy.actor, retain_graph=True
                ).detach()

                # direction: calculate natural gradient
                with torch.no_grad():
                    old_dist = self.policy(minibatch).dist

                kl = kl_divergence(old_dist, dist).mean()
                # calculate first order gradient of kl with respect to theta
                flat_kl_grad = self._get_flat_grad(kl, self.policy.actor, create_graph=True)
                search_direction = -self._conjugate_gradients(flat_grads, flat_kl_grad, nsteps=10)

                # step
                with torch.no_grad():
                    flat_params = torch.cat(
                        [param.data.view(-1) for param in self.policy.actor.parameters()],
                    )
                    new_flat_params = flat_params + self.actor_step_size * search_direction
                    self._set_from_flat_params(self.policy.actor, new_flat_params)
                    new_dist = self.policy(minibatch).dist
                    kl = kl_divergence(old_dist, new_dist).mean()

                # optimize critic
                for _ in range(self.optim_critic_iters):
                    value = self.critic(minibatch.obs).flatten()
                    vf_loss = F.mse_loss(minibatch.returns, value)
                    self.optim.step(vf_loss)

                actor_losses.append(actor_loss.item())
                vf_losses.append(vf_loss.item())
                kls.append(kl.item())

        return NPGTrainingStats(
            actor_loss=SequenceSummaryStats.from_sequence(actor_losses),
            vf_loss=SequenceSummaryStats.from_sequence(vf_losses),
            kl=SequenceSummaryStats.from_sequence(kls),
        )

    def _MVP(self, v: torch.Tensor, flat_kl_grad: torch.Tensor) -> torch.Tensor:
        """Matrix vector product."""
        # caculate second order gradient of kl with respect to theta
        kl_v = (flat_kl_grad * v).sum()
        flat_kl_grad_grad = self._get_flat_grad(kl_v, self.policy.actor, retain_graph=True).detach()
        return flat_kl_grad_grad + v * self._damping

    def _conjugate_gradients(
        self,
        minibatch: torch.Tensor,
        flat_kl_grad: torch.Tensor,
        nsteps: int = 10,
        residual_tol: float = 1e-10,
    ) -> torch.Tensor:
        x = torch.zeros_like(minibatch)
        r, p = minibatch.clone(), minibatch.clone()
        # Note: should be 'r, p = minibatch - MVP(x)', but for x=0, MVP(x)=0.
        # Change if doing warm start.
        rdotr = r.dot(r)
        for _ in range(nsteps):
            z = self._MVP(p, flat_kl_grad)
            alpha = rdotr / p.dot(z)
            x += alpha * p
            r -= alpha * z
            new_rdotr = r.dot(r)
            if new_rdotr < residual_tol:
                break
            p = r + new_rdotr / rdotr * p
            rdotr = new_rdotr
        return x

    def _get_flat_grad(self, y: torch.Tensor, model: nn.Module, **kwargs: Any) -> torch.Tensor:
        grads = torch.autograd.grad(y, model.parameters(), **kwargs)  # type: ignore
        return torch.cat([grad.reshape(-1) for grad in grads])

    def _set_from_flat_params(self, model: nn.Module, flat_params: torch.Tensor) -> nn.Module:
        prev_ind = 0
        for param in model.parameters():
            flat_size = int(np.prod(list(param.size())))
            param.data.copy_(flat_params[prev_ind : prev_ind + flat_size].view(param.size()))
            prev_ind += flat_size
        return model
