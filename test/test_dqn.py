import gym
import time
import tqdm
import torch
import argparse
import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import DQNPolicy
from tianshou.env import SubprocVectorEnv
from tianshou.utils import tqdm_config, MovAvg
from tianshou.data import Collector, ReplayBuffer


class Net(nn.Module):
    def __init__(self, layer_num, state_shape, action_shape, device):
        super().__init__()
        self.device = device
        self.model = [
            nn.Linear(np.prod(state_shape), 128),
            nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(128, 128), nn.ReLU(inplace=True)]
        self.model += [nn.Linear(128, np.prod(action_shape))]
        self.model = nn.Sequential(*self.model)

    def forward(self, s, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.Tensor(s)
        s = s.to(self.device)
        batch = s.shape[0]
        q = self.model(s.view(batch, -1))
        return q, None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='CartPole-v0')
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step-per-epoch', type=int, default=320)
    parser.add_argument('--collect-per-step', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--layer-num', type=int, default=3)
    parser.add_argument('--training-num', type=int, default=8)
    parser.add_argument('--test-num', type=int, default=20)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


def test_dqn(args=get_args()):
    env = gym.make(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
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
    net = Net(args.layer_num, args.state_shape, args.action_shape, args.device)
    net = net.to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss = nn.MSELoss()
    policy = DQNPolicy(net, optim, loss, args.gamma, args.n_step)
    # collector
    training_collector = Collector(
        policy, train_envs, ReplayBuffer(args.buffer_size))
    test_collector = Collector(
        policy, test_envs, ReplayBuffer(args.buffer_size), args.test_num)
    training_collector.collect(n_step=args.batch_size)
    # log
    stat_loss = MovAvg()
    global_step = 0
    writer = SummaryWriter(args.logdir)
    best_epoch = -1
    best_reward = -1e10
    for epoch in range(args.epoch):
        desc = f"Epoch #{epoch + 1}"
        # train
        policy.train()
        policy.sync_weight()
        policy.set_eps(args.eps_train)
        with tqdm.trange(
                0, args.step_per_epoch, desc=desc, **tqdm_config) as t:
            for _ in t:
                training_collector.collect(n_step=args.collect_per_step)
                global_step += 1
                result = training_collector.stat()
                loss = policy.learn(training_collector.sample(args.batch_size))
                stat_loss.add(loss)
                writer.add_scalar(
                    'reward', result['reward'], global_step=global_step)
                writer.add_scalar(
                    'length', result['length'], global_step=global_step)
                writer.add_scalar(
                    'loss', stat_loss.get(), global_step=global_step)
                t.set_postfix(loss=f'{stat_loss.get():.6f}',
                              reward=f'{result["reward"]:.6f}',
                              length=f'{result["length"]:.6f}')
        # eval
        test_collector.reset_env()
        test_collector.reset_buffer()
        policy.eval()
        policy.set_eps(args.eps_test)
        test_collector.collect(n_episode=args.test_num)
        result = test_collector.stat()
        if best_reward < result['reward']:
            best_reward = result['reward']
            best_epoch = epoch
        print(f'Epoch #{epoch + 1} test_reward: {result["reward"]:.6f}, '
              f'best_reward: {best_reward:.6f} in #{best_epoch}')
        if args.task == 'CartPole-v0' and best_reward >= 200:
            break
    assert best_reward >= 200
    if __name__ == '__main__':
        # let's watch its performance!
        env = gym.make(args.task)
        obs = env.reset()
        done = False
        total = 0
        while not done:
            q, _ = net([obs])
            action = q.max(dim=1)[1]
            obs, rew, done, info = env.step(action[0].detach().cpu().numpy())
            total += rew
            env.render()
            time.sleep(1 / 100)
        env.close()
        print(f'Final test: {total}')
    return best_reward


if __name__ == '__main__':
    test_dqn(get_args())
