import torch
from copy import deepcopy
import torch.nn.functional as F

from tianshou.data import Batch
from tianshou.policy import BasePolicy
# from tianshou.exploration import OUNoise


class DDPGPolicy(BasePolicy):
    """docstring for DDPGPolicy"""

    def __init__(self, actor, actor_optim,
                 critic, critic_optim, action_range,
                 tau=0.005, gamma=0.99, exploration_noise=0.1):
        super().__init__()
        self.actor = actor
        self.actor_old = deepcopy(actor)
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.actor_old.eval()
        self.actor_optim = actor_optim
        self.critic = critic
        self.critic_old = deepcopy(critic)
        self.critic_old.load_state_dict(self.critic.state_dict())
        self.critic_old.eval()
        self.critic_optim = critic_optim
        assert 0 < tau <= 1, 'tau should in (0, 1]'
        self._tau = tau
        assert 0 < gamma <= 1, 'gamma should in (0, 1]'
        self._gamma = gamma
        assert 0 <= exploration_noise, 'noise should not be negative'
        self._eps = exploration_noise
        self._range = action_range
        # self.noise = OUNoise()

    def set_eps(self, eps):
        self._eps = eps

    def train(self):
        self.training = True
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.training = False
        self.actor.eval()
        self.critic.eval()

    def sync_weight(self):
        for o, n in zip(self.actor_old.parameters(), self.actor.parameters()):
            o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)
        for o, n in zip(
                self.critic_old.parameters(), self.critic.parameters()):
            o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)

    def process_fn(self, batch, buffer, indice):
        return batch

    def __call__(self, batch, state=None,
                 model='actor', input='obs', eps=None):
        model = getattr(self, model)
        obs = getattr(batch, input)
        logits, h = model(obs, state=state, info=batch.info)
        # noise = np.random.normal(0, self._eps, size=logits.shape)
        logits += torch.randn(
            size=logits.shape, device=logits.device) * self._eps
        # noise = self.noise(logits.shape, self._eps)
        # logits += torch.tensor(noise, device=logits.device)
        logits = logits.clamp(self._range[0], self._range[1])
        return Batch(act=logits, state=h)

    def learn(self, batch, batch_size=None):
        target_q = self.critic_old(
            batch.obs_next, self.actor_old(batch.obs_next, state=None)[0])
        dev = target_q.device
        rew = torch.tensor(batch.rew, dtype=torch.float, device=dev)
        done = torch.tensor(batch.done, dtype=torch.float, device=dev)
        target_q = rew[:, None] + ((
            1. - done[:, None]) * self._gamma * target_q).detach()
        current_q = self.critic(batch.obs, batch.act)
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        actor_loss = -self.critic(
            batch.obs, self.actor(batch.obs, state=None)[0]).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.sync_weight()
        return {
            'loss/actor': actor_loss.detach().cpu().numpy(),
            'loss/critic': critic_loss.detach().cpu().numpy(),
        }
