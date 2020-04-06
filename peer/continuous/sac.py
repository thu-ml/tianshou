import gym
import torch
import pprint
import argparse
import datetime
import numpy as np

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import SACPolicy
from tianshou.data import Collector, ReplayBuffer
from tianshou.env import VectorEnv, SubprocVectorEnv

from peer.offpolicy import offpolicy_trainer_with_views
from peer.continuous.net import ActorProbWithView, CriticWithView


class View(object):
    def __init__(self, args, mask=None, name=None):
        env = gym.make(args.task)
        if args.task == 'Pendulum-v0':
            env.spec.reward_threshold = -250
        self.state_shape = env.observation_space.shape or env.observation_space.n
        self.action_shape = env.action_space.shape or env.action_space.n
        self.max_action = env.action_space.high[0]

        self.stop_fn = lambda x: x >= env.spec.reward_threshold

        # env
        self.train_envs = VectorEnv(
            [lambda: gym.make(args.task) for _ in range(args.training_num)])
        self.test_envs = SubprocVectorEnv(
            [lambda: gym.make(args.task) for _ in range(args.test_num)])

        # mask
        state_dim = int(np.prod(self.state_shape))
        self._view_mask = torch.ones(state_dim)
        if mask == 'even':
            for i in range(0, state_dim, 2):
                self._view_mask[i] = 0
        elif mask == "odd":
            for i in range(1, state_dim, 2):
                self._view_mask[i] = 1

        # policy
        self.actor = ActorProbWithView(
            args.layer_num, self.state_shape, self.action_shape,
            self.max_action, self._view_mask, args.device
        ).to(args.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic1 = CriticWithView(
            args.layer_num, self.state_shape, self._view_mask, self.action_shape, args.device
        ).to(args.device)
        self.critic1_optim = torch.optim.Adam(self.critic1.parameters(), lr=args.critic_lr)
        self.critic2 = CriticWithView(
            args.layer_num, self.state_shape, self._view_mask, self.action_shape, args.device
        ).to(args.device)
        self.critic2_optim = torch.optim.Adam(self.critic2.parameters(), lr=args.critic_lr)
        self.policy = SACPolicy(
            self.actor, self.actor_optim, self.critic1, self.critic1_optim, self.critic2,
            self.critic2_optim, args.tau, args.gamma, args.alpha,
            [env.action_space.low[0], env.action_space.high[0]],
            reward_normalization=True, ignore_done=True)

        # collector
        self.train_collector = Collector(self.policy, self.train_envs,
                                         ReplayBuffer(args.buffer_size))
        self.test_collector = Collector(self.policy, self.test_envs)

        # log
        self.writer = SummaryWriter(f"{args.logdir}/{args.task}/sac/{args.note}/{name}")

    def seed(self, _seed):
        self.train_envs.seed(_seed)
        self.test_envs.seed(_seed)

    def close(self):
        self.train_collector.close()
        self.test_collector.close()

    def train(self):
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def learn_from_demos(self, batch, demo_acts, peer=0):
        acts = self.policy(batch).act
        loss = F.mse_loss(acts, demo_acts)
        if peer != 0:
            peer_demos = demo_acts[torch.randperm(len(demo_acts))]
            loss += peer * F.mse_loss(acts, peer_demos)
        self.policy.actor_optim.zero_grad()
        loss.backward()
        self.policy.actor_optim.step()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--note', type=str, default=None)
    parser.add_argument('--peer', type=float, default=0)
    parser.add_argument('--copier', action='store_true')
    parser.add_argument('--task', type=str, default='Pendulum-v0')
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step-per-epoch', type=int, default=2400)
    parser.add_argument('--collect-per-step', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--layer-num', type=int, default=1)
    parser.add_argument('--training-num', type=int, default=8)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_known_args()[0]
    args.note = args.note or datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    return args


def test_sac(args=get_args()):
    A = View(args, mask='even', name='A')
    B = View(args, mask='odd', name='B')

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    A.seed(args.seed)
    B.seed(args.seed)

    # trainer
    result = offpolicy_trainer_with_views(
        A, B, args.epoch, args.step_per_epoch, args.collect_per_step,
        args.test_num, args.batch_size, copier=args.copier,
        peer=args.peer, task=args.task)
    # assert A.stop_fn(result['best_reward'])

    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        # env = gym.make(args.task)
        # collector = Collector(policy, env)
        # result = collector.collect(n_episode=1, render=args.render)
        # print(f'Final reward: {result["rew"]}, length: {result["len"]}')
        # collector.close()


if __name__ == '__main__':
    test_sac()
