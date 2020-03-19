import gym
import torch
import argparse
import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import A2CPolicy
from tianshou.env import SubprocVectorEnv
from tianshou.trainer import episodic_trainer
from tianshou.data import Collector, ReplayBuffer


class Net(nn.Module):
    def __init__(self, layer_num, state_shape, device='cpu'):
        super().__init__()
        self.device = device
        self.model = [
            nn.Linear(np.prod(state_shape), 128),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(128, 128), nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*self.model)

    def forward(self, s):
        s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        return logits


class Actor(nn.Module):
    def __init__(self, preprocess_net, action_shape):
        super().__init__()
        self.model = nn.Sequential(*[
            preprocess_net,
            nn.Linear(128, np.prod(action_shape)),
        ])

    def forward(self, s, **kwargs):
        logits = self.model(s)
        return logits, None


class Critic(nn.Module):
    def __init__(self, preprocess_net):
        super().__init__()
        self.model = nn.Sequential(*[
            preprocess_net,
            nn.Linear(128, 1),
        ])

    def forward(self, s):
        logits = self.model(s)
        return logits


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='CartPole-v0')
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step-per-epoch', type=int, default=320)
    parser.add_argument('--collect-per-step', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--layer-num', type=int, default=2)
    parser.add_argument('--training-num', type=int, default=32)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    # a2c special
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--entropy-coef', type=float, default=0.001)
    parser.add_argument('--max-grad-norm', type=float, default=None)
    args = parser.parse_known_args()[0]
    return args


def test_a2c(args=get_args()):
    env = gym.make(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # train_envs = gym.make(args.task)
    train_envs = SubprocVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.training_num)],
        reset_after_done=True)
    # test_envs = gym.make(args.task)
    test_envs = SubprocVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.test_num)],
        reset_after_done=False)
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net = Net(args.layer_num, args.state_shape, args.device)
    actor = Actor(net, args.action_shape).to(args.device)
    critic = Critic(net).to(args.device)
    optim = torch.optim.Adam(list(
        actor.parameters()) + list(critic.parameters()), lr=args.lr)
    dist = torch.distributions.Categorical
    policy = A2CPolicy(
        actor, critic, optim, dist, args.gamma, vf_coef=args.vf_coef,
        entropy_coef=args.entropy_coef, max_grad_norm=args.max_grad_norm)
    # collector
    train_collector = Collector(
        policy, train_envs, ReplayBuffer(args.buffer_size))
    test_collector = Collector(policy, test_envs, stat_size=args.test_num)
    # log
    writer = SummaryWriter(args.logdir)

    def stop_fn(x):
        return x >= env.spec.reward_threshold

    # trainer
    train_step, train_episode, test_step, test_episode, best_rew, duration = \
        episodic_trainer(
            policy, train_collector, test_collector, args.epoch,
            args.step_per_epoch, args.collect_per_step, args.test_num,
            args.batch_size, stop_fn=stop_fn, writer=writer)
    assert stop_fn(best_rew)
    train_collector.close()
    test_collector.close()
    if __name__ == '__main__':
        print(f'Collect {train_step} frame / {train_episode} episode during '
              f'training and {test_step} frame / {test_episode} episode during'
              f' test in {duration:.2f}s, best_reward: {best_rew}, speed: '
              f'{(train_step + test_step) / duration:.2f}it/s')
        # Let's watch its performance!
        env = gym.make(args.task)
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=1 / 35)
        print(f'Final reward: {result["rew"]}, length: {result["len"]}')
        collector.close()


if __name__ == '__main__':
    test_a2c()
