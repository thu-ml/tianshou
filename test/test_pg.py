import gym
import time
import tqdm
import torch
import argparse
import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import PGPolicy
from tianshou.env import SubprocVectorEnv
from tianshou.utils import tqdm_config, MovAvg
from tianshou.data import Batch, Collector, ReplayBuffer


def compute_return_base(batch, aa=None, bb=None, gamma=0.1):
    returns = np.zeros_like(batch.rew)
    last = 0
    for i in reversed(range(len(batch.rew))):
        returns[i] = batch.rew[i]
        if not batch.done[i]:
            returns[i] += last * gamma
        last = returns[i]
    batch.update(returns=returns)
    return batch


def test_fn(size=2560):
    policy = PGPolicy(None, None, None, discount_factor=0.1)
    fn = policy.process_fn
    # fn = compute_return_base
    batch = Batch(
        done=np.array([1, 0, 0, 1, 0, 1, 0, 1.]),
        rew=np.array([0, 1, 2, 3, 4, 5, 6, 7.]),
    )
    batch = fn(batch, None, None)
    ans = np.array([0, 1.23, 2.3, 3, 4.5, 5, 6.7, 7])
    assert abs(batch.returns - ans).sum() <= 1e-5
    batch = Batch(
        done=np.array([0, 1, 0, 1, 0, 1, 0.]),
        rew=np.array([7, 6, 1, 2, 3, 4, 5.]),
    )
    batch = fn(batch, None, None)
    ans = np.array([7.6, 6, 1.2, 2, 3.4, 4, 5])
    assert abs(batch.returns - ans).sum() <= 1e-5
    batch = Batch(
        done=np.array([0, 1, 0, 1, 0, 0, 1.]),
        rew=np.array([7, 6, 1, 2, 3, 4, 5.]),
    )
    batch = fn(batch, None, None)
    ans = np.array([7.6, 6, 1.2, 2, 3.45, 4.5, 5])
    assert abs(batch.returns - ans).sum() <= 1e-5
    if __name__ == '__main__':
        batch = Batch(
            done=np.random.randint(100, size=size) == 0,
            rew=np.random.random(size),
        )
        cnt = 3000
        t = time.time()
        for _ in range(cnt):
            compute_return_base(batch)
        print(f'vanilla: {(time.time() - t) / cnt}')
        t = time.time()
        for _ in range(cnt):
            policy.process_fn(batch, None, None)
        print(f'policy: {(time.time() - t) / cnt}')


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
        logits = self.model(s.view(batch, -1))
        return logits, None


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
    parser.add_argument('--layer-num', type=int, default=3)
    parser.add_argument('--training-num', type=int, default=8)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


def test_pg(args=get_args()):
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
    dist = torch.distributions.Categorical
    policy = PGPolicy(net, optim, dist, args.gamma)
    # collector
    training_collector = Collector(
        policy, train_envs, ReplayBuffer(args.buffer_size))
    test_collector = Collector(policy, test_envs, stat_size=args.test_num)
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
        with tqdm.tqdm(
                total=args.step_per_epoch, desc=desc, **tqdm_config) as t:
            while t.n < t.total:
                result = training_collector.collect(
                    n_episode=args.collect_per_step)
                losses = policy.learn(
                    training_collector.sample(0), args.batch_size)
                training_collector.reset_buffer()
                global_step += len(losses)
                t.update(len(losses))
                stat_loss.add(losses)
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
            if t.n <= t.total:
                t.update()
        # eval
        test_collector.reset_env()
        test_collector.reset_buffer()
        policy.eval()
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
    # test_fn()
    test_pg()
