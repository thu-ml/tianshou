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
    def __init__(self, layer_num, state_shape, action_shape, device='cpu'):
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
        s = torch.tensor(s, device=self.device, dtype=torch.float)
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
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step-per-epoch', type=int, default=320)
    parser.add_argument('--collect-per-step', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--layer-num', type=int, default=3)
    parser.add_argument('--training-num', type=int, default=8)
    parser.add_argument('--test-num', type=int, default=100)
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
    start_time = time.time()
    for epoch in range(1, 1 + args.epoch):
        desc = f"Epoch #{epoch}"
        # train
        policy.train()
        policy.sync_weight()
        policy.set_eps(args.eps_train)
        with tqdm.tqdm(
                total=args.step_per_epoch, desc=desc, **tqdm_config) as t:
            while t.n < t.total:
                result = training_collector.collect(
                    n_step=args.collect_per_step)
                for i in range(min(
                        result['n_step'] // args.collect_per_step,
                        t.total - t.n)):
                    t.update(1)
                    global_step += 1
                    loss = policy.learn(
                        training_collector.sample(args.batch_size))
                    stat_loss.add(loss)
                    writer.add_scalar(
                        'reward', result['reward'], global_step=global_step)
                    writer.add_scalar(
                        'length', result['length'], global_step=global_step)
                    writer.add_scalar(
                        'loss', stat_loss.get(), global_step=global_step)
                    writer.add_scalar(
                        'speed', result['speed'], global_step=global_step)
                    t.set_postfix(loss=f'{stat_loss.get():.6f}',
                                  reward=f'{result["reward"]:.6f}',
                                  length=f'{result["length"]:.2f}',
                                  speed=f'{result["speed"]:.2f}')
        # eval
        test_collector.reset_env()
        test_collector.reset_buffer()
        policy.eval()
        policy.set_eps(args.eps_test)
        result = test_collector.collect(n_episode=args.test_num)
        if best_reward < result['reward']:
            best_reward = result['reward']
            best_epoch = epoch
        print(f'Epoch #{epoch}: test_reward: {result["reward"]:.6f}, '
              f'best_reward: {best_reward:.6f} in #{best_epoch}')
        if best_reward >= env.spec.reward_threshold:
            break
    assert best_reward >= env.spec.reward_threshold
    training_collector.close()
    test_collector.close()
    if __name__ == '__main__':
        train_cnt = training_collector.collect_step
        test_cnt = test_collector.collect_step
        duration = time.time() - start_time
        print(f'Collect {train_cnt} training frame and {test_cnt} test frame '
              f'in {duration:.2f}s, '
              f'speed: {(train_cnt + test_cnt) / duration:.2f}it/s')
        # Let's watch its performance!
        env = gym.make(args.task)
        test_collector = Collector(policy, env)
        result = test_collector.collect(n_episode=1, render=1 / 35)
        print(f'Final reward: {result["reward"]}, length: {result["length"]}')
        test_collector.close()


if __name__ == '__main__':
    test_dqn(get_args())
