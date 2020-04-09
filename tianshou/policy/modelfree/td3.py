import torch
from copy import deepcopy
import torch.nn.functional as F

from tianshou.policy import DDPGPolicy


class TD3Policy(DDPGPolicy):
    """Implementation of Twin Delayed Deep Deterministic Policy Gradient,
    arXiv:1802.09477

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic1: the first critic network. (s, a -> Q(s,
        a))
    :param torch.optim.Optimizer critic1_optim: the optimizer for the first
        critic network.
    :param torch.nn.Module critic2: the second critic network. (s, a -> Q(s,
        a))
    :param torch.optim.Optimizer critic2_optim: the optimizer for the second
        critic network.
    :param float tau: param for soft update of the target network, defaults to
        0.005.
    :param float gamma: discount factor, in [0, 1], defaults to 0.99.
    :param float exploration_noise: the noise intensity, add to the action,
        defaults to 0.1.
    :param float policy_noise: the noise used in updating policy network,
        default to 0.2.
    :param int update_actor_freq: the update frequency of actor network,
        default to 2.
    :param float noise_clip: the clipping range used in updating policy
        network, default to 0.5.
    :param action_range: the action range (minimum, maximum).
    :type action_range: [float, float]
    :param bool reward_normalization: normalize the reward to Normal(0, 1),
        defaults to ``False``.
    :param bool ignore_done: ignore the done flag while training the policy,
        defaults to ``False``.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(self, actor, actor_optim, critic1, critic1_optim,
                 critic2, critic2_optim, tau=0.005, gamma=0.99,
                 exploration_noise=0.1, policy_noise=0.2, update_actor_freq=2,
                 noise_clip=0.5, action_range=None,
                 reward_normalization=False, ignore_done=False, **kwargs):
        super().__init__(actor, actor_optim, None, None, tau, gamma,
                         exploration_noise, action_range, reward_normalization,
                         ignore_done, **kwargs)
        self.critic1, self.critic1_old = critic1, deepcopy(critic1)
        self.critic1_old.eval()
        self.critic1_optim = critic1_optim
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)
        self.critic2_old.eval()
        self.critic2_optim = critic2_optim
        self._policy_noise = policy_noise
        self._freq = update_actor_freq
        self._noise_clip = noise_clip
        self._cnt = 0
        self._last = 0

    def train(self):
        self.training = True
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval(self):
        self.training = False
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()

    def sync_weight(self):
        for o, n in zip(self.actor_old.parameters(), self.actor.parameters()):
            o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)
        for o, n in zip(
                self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)
        for o, n in zip(
                self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)

    def learn(self, batch, **kwargs):
        with torch.no_grad():
            a_ = self(batch, model='actor_old', input='obs_next').act
            dev = a_.device
            noise = torch.randn(size=a_.shape, device=dev) * self._policy_noise
            if self._noise_clip >= 0:
                noise = noise.clamp(-self._noise_clip, self._noise_clip)
            a_ += noise
            a_ = a_.clamp(self._range[0], self._range[1])
            target_q = torch.min(
                self.critic1_old(batch.obs_next, a_),
                self.critic2_old(batch.obs_next, a_))
            rew = torch.tensor(batch.rew,
                               dtype=torch.float, device=dev)[:, None]
            done = torch.tensor(batch.done,
                                dtype=torch.float, device=dev)[:, None]
            target_q = (rew + (1. - done) * self._gamma * target_q)
        # critic 1
        current_q1 = self.critic1(batch.obs, batch.act)
        critic1_loss = F.mse_loss(current_q1, target_q)
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()
        # critic 2
        current_q2 = self.critic2(batch.obs, batch.act)
        critic2_loss = F.mse_loss(current_q2, target_q)
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()
        if self._cnt % self._freq == 0:
            actor_loss = -self.critic1(
                batch.obs, self(batch, eps=0).act).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self._last = actor_loss.item()
            self.actor_optim.step()
            self.sync_weight()
        self._cnt += 1
        return {
            'loss/actor': self._last,
            'loss/critic1': critic1_loss.item(),
            'loss/critic2': critic2_loss.item(),
        }
