import gym
import torch
import pprint
import argparse
import datetime
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import PPOPolicy
from tianshou.env import SubprocVectorEnv
from tianshou.data import Collector, ReplayBuffer

if __name__ == '__main__':
    from .net import NetWithView, Actor, Critic
    from ..onpolicy import onpolicy_trainer
else:  # pytest
    from peer.discrete.net import NetWithView, Actor, Critic
    from peer.onpolicy import onpolicy_trainer


class View(object):
    def __init__(self, args, mask=None, name='full'):
        env = gym.make(args.task)
        self.stop_fn = lambda x: x >= env.spec.reward_threshold
        self.state_shape = env.observation_space.shape or env.observation_space.n
        self.action_shape = env.action_space.shape or env.action_space.n
        self.buffer = ReplayBuffer(400000)

        # Env
        # train_envs = gym.make(args.task)
        self.train_envs = SubprocVectorEnv(
            [lambda: gym.make(args.task) for _ in range(args.training_num)])
        # test_envs = gym.make(args.task)
        self.test_envs = SubprocVectorEnv(
            [lambda: gym.make(args.task) for _ in range(args.test_num)])

        # Mask
        state_dim = int(np.prod(self.state_shape))
        self._view_mask = torch.ones(state_dim)
        if mask == 'even':
            for i in range(0, state_dim, 2):
                self._view_mask[i] = 0
        elif mask == "odd":
            for i in range(1, state_dim, 2):
                self._view_mask[i] = 0
        elif type(mask) == int:
            self._view_mask[mask] = 0

        # Model
        net = NetWithView(args.layer_num, self.state_shape, device=args.device,
                          mask=self._view_mask)
        self.actor = Actor(net, self.action_shape).to(args.device)
        self.critic = Critic(net).to(args.device)
        optim = torch.optim.Adam(list(
            self.actor.parameters()) + list(self.critic.parameters()), lr=args.lr)
        dist = torch.distributions.Categorical
        self.policy = PPOPolicy(
            self.actor, self.critic, optim, dist, args.gamma,
            max_grad_norm=args.max_grad_norm,
            eps_clip=args.eps_clip,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            action_range=None)

        # Collector
        self.train_collector = Collector(
            self.policy, self.train_envs, ReplayBuffer(args.buffer_size))
        self.test_collector = Collector(self.policy, self.test_envs)

        # Log
        self.writer = SummaryWriter(f'{args.logdir}/{args.task}/ppo/{args.note}/{name}')

    def seed(self, _seed):
        self.train_envs.seed(_seed)
        self.test_envs.seed(_seed)

    def close(self):
        self.train_collector.close()
        self.test_collector.close()

    def train(self):
        self.actor.train()
        self.critic.train()

    def learn_from_demos(self, batch, demo, peer=0):
        logits = self.policy(batch).logits
        demo = demo.act.detach()
        loss = F.cross_entropy(logits, demo)
        if peer != 0:
            peer_demo = demo[torch.randperm(len(demo))]
            loss -= peer * F.cross_entropy(logits, peer_demo)
        self.policy.optim.zero_grad()
        loss.backward()
        self.policy.optim.step()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--note', type=str, default=None)
    parser.add_argument('--peer', type=float, default=0)
    parser.add_argument('--copier', action='store_true')
    parser.add_argument('--task', type=str, default='Acrobot-v1')
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--copier-batch-size', type=int, default=4000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--collect-per-step', type=int, default=10)
    parser.add_argument('--repeat-per-collect', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--layer-num', type=int, default=1)
    parser.add_argument('--training-num', type=int, default=32)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--device', type=str, default='cpu')
    # ppo special
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    args = parser.parse_known_args()[0]
    args.note = args.note or datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    return args


def test_ppo(args=get_args()):
    A = View(args, mask=0, name='A')
    B = View(args, mask=1, name='B')

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    A.seed(args.seed)
    B.seed(args.seed)

    # Trainer
    result = onpolicy_trainer(
        A, B, args.epoch, args.step_per_epoch, args.collect_per_step,
        args.repeat_per_collect, args.test_num, args.batch_size,
        task=args.task, copier=args.copier, peer=args.peer,
        copier_batch_size=args.copier_batch_size)
    print(result)
    A.close()
    B.close()


if __name__ == '__main__':
    test_ppo()
