import torch
import torch.nn.functional as F

from tianshou.algorithm import TD3
from tianshou.algorithm.algorithm_base import OfflineAlgorithm
from tianshou.algorithm.modelfree.ddpg import ContinuousDeterministicPolicy
from tianshou.algorithm.modelfree.td3 import TD3TrainingStats
from tianshou.algorithm.optim import OptimizerFactory
from tianshou.data import to_torch_as
from tianshou.data.types import RolloutBatchProtocol


# NOTE: This uses diamond inheritance to convert from off-policy to offline
class TD3BC(OfflineAlgorithm[ContinuousDeterministicPolicy], TD3):  # type: ignore
    """Implementation of TD3+BC. arXiv:2106.06860."""

    def __init__(
        self,
        *,
        policy: ContinuousDeterministicPolicy,
        policy_optim: OptimizerFactory,
        critic: torch.nn.Module,
        critic_optim: OptimizerFactory,
        critic2: torch.nn.Module | None = None,
        critic2_optim: OptimizerFactory | None = None,
        tau: float = 0.005,
        gamma: float = 0.99,
        policy_noise: float = 0.2,
        update_actor_freq: int = 2,
        noise_clip: float = 0.5,
        alpha: float = 2.5,
        n_step_return_horizon: int = 1,
    ) -> None:
        """
        :param policy: the policy
        :param policy_optim: the optimizer factory for the policy's model.
        :param critic: the first critic network. (s, a -> Q(s, a))
        :param critic_optim: the optimizer factory for the first critic network.
        :param critic2: the second critic network. (s, a -> Q(s, a)).
            If None, copy the first critic (via deepcopy).
        :param critic2_optim: the optimizer factory for the second critic network.
            If None, use the first critic's factory.
        :param tau: the soft update coefficient for target networks, controlling the rate at which
            target networks track the learned networks.
            When the parameters of the target network are updated with the current (source) network's
            parameters, a weighted average is used: target = tau * source + (1 - tau) * target.
            Smaller values (closer to 0) create more stable but slower learning as target networks
            change more gradually. Higher values (closer to 1) allow faster learning but may reduce
            stability.
            Typically set to a small value (0.001 to 0.01) for most reinforcement learning tasks.
        :param gamma: the discount factor in [0, 1] for future rewards.
            This determines how much future rewards are valued compared to immediate ones.
            Lower values (closer to 0) make the agent focus on immediate rewards, creating "myopic"
            behavior. Higher values (closer to 1) make the agent value long-term rewards more,
            potentially improving performance in tasks where delayed rewards are important but
            increasing training variance by incorporating more environmental stochasticity.
            Typically set between 0.9 and 0.99 for most reinforcement learning tasks
        :param policy_noise: standard deviation of the Gaussian noise before optional clipping
            by ``noise_clip``. TD3+BC adds the noise to the target actor's action when computing
            the critic target (target policy smoothing).
        :param update_actor_freq: the frequency of actor network updates relative to critic network updates
            (the actor network is only updated once for every `update_actor_freq` critic updates).
            This implements the "delayed" policy updates from the TD3 algorithm, where the actor is
            updated less frequently than the critics.
            Higher values (e.g., 2-5) help stabilize training by allowing the critic to become more
            accurate before updating the policy.
            The default value of 2 follows the original TD3 paper's recommendation of updating the
            policy at half the rate of the Q-functions.
        :param noise_clip: clip target policy noise to ``[-noise_clip, noise_clip]``.
            A nonpositive value disables clipping.
        :param alpha: the value of alpha, which controls the weight for TD3 learning
            relative to behavior cloning.
        """
        TD3.__init__(
            self,
            policy=policy,
            policy_optim=policy_optim,
            critic=critic,
            critic_optim=critic_optim,
            critic2=critic2,
            critic2_optim=critic2_optim,
            tau=tau,
            gamma=gamma,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            update_actor_freq=update_actor_freq,
            n_step_return_horizon=n_step_return_horizon,
        )
        self.alpha = alpha

    def _update_with_batch(self, batch: RolloutBatchProtocol) -> TD3TrainingStats:
        # critic 1&2
        td1, critic1_loss = self._minimize_critic_squared_loss(
            batch, self.critic, self.critic_optim
        )
        td2, critic2_loss = self._minimize_critic_squared_loss(
            batch, self.critic2, self.critic2_optim
        )
        batch.weight = (td1 + td2) / 2.0  # prio-buffer

        # actor
        if self._cnt % self.update_actor_freq == 0:
            act = self.policy(batch, eps=0.0).act
            q_value = self.critic(batch.obs, act)
            lmbda = self.alpha / q_value.abs().mean().detach()
            actor_loss = -lmbda * q_value.mean() + F.mse_loss(act, to_torch_as(batch.act, act))
            self._last = actor_loss.item()
            self.policy_optim.step(actor_loss)
            self._update_lagged_network_weights()
        self._cnt += 1

        return TD3TrainingStats(
            actor_loss=self._last,
            critic1_loss=critic1_loss.item(),
            critic2_loss=critic2_loss.item(),
        )
