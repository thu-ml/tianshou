import os
import gym
import time
import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.env import VectorEnv
from tianshou.policy import PGPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.data import Batch, Collector, ReplayBuffer

if __name__ == '__main__':
    from net import Net
else:  # pytest
    from test.discrete.net import Net


def compute_return_base(batch, aa=None, bb=None, gamma=0.1):
    returns = np.zeros_like(batch.rew)
    last = 0
    for i in reversed(range(len(batch.rew))):
        returns[i] = batch.rew[i]
        if not batch.done[i]:
            returns[i] += last * gamma
        last = returns[i]
    batch.returns = returns
    return batch


def test_fn(size=2560):
    policy = PGPolicy(None, None, None, discount_factor=0.1)
    buf = ReplayBuffer(100)
    buf.add(1, 1, 1, 1, 1)
    fn = policy.process_fn
    # fn = compute_return_base
    batch = Batch(
        done=np.array([1, 0, 0, 1, 0, 1, 0, 1.]),
        rew=np.array([0, 1, 2, 3, 4, 5, 6, 7.]),
    )
    batch = fn(batch, buf, 0)
    ans = np.array([0, 1.23, 2.3, 3, 4.5, 5, 6.7, 7])
    assert abs(batch.returns - ans).sum() <= 1e-5
    batch = Batch(
        done=np.array([0, 1, 0, 1, 0, 1, 0.]),
        rew=np.array([7, 6, 1, 2, 3, 4, 5.]),
    )
    batch = fn(batch, buf, 0)
    ans = np.array([7.6, 6, 1.2, 2, 3.4, 4, 5])
    assert abs(batch.returns - ans).sum() <= 1e-5
    batch = Batch(
        done=np.array([0, 1, 0, 1, 0, 0, 1.]),
        rew=np.array([7, 6, 1, 2, 3, 4, 5.]),
    )
    batch = fn(batch, buf, 0)
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
            policy.process_fn(batch, buf, 0)
        print(f'policy: {(time.time() - t) / cnt}')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='CartPole-v0')
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--collect-per-step', type=int, default=10)
    parser.add_argument('--repeat-per-collect', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--layer-num', type=int, default=3)
    parser.add_argument('--training-num', type=int, default=8)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
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
    # you can also use tianshou.env.SubprocVectorEnv
    train_envs = VectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.training_num)])
    # test_envs = gym.make(args.task)
    test_envs = VectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net = Net(
        args.layer_num, args.state_shape, args.action_shape,
        device=args.device, softmax=True)
    net = net.to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    dist = torch.distributions.Categorical
    policy = PGPolicy(net, optim, dist, args.gamma)
    # collector
    train_collector = Collector(
        policy, train_envs, ReplayBuffer(args.buffer_size))
    test_collector = Collector(policy, test_envs)
    # log
    log_path = os.path.join(args.logdir, args.task, 'pg')
    writer = SummaryWriter(log_path)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(x):
        return x >= env.spec.reward_threshold

    # trainer
    result = onpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.repeat_per_collect,
        args.test_num, args.batch_size, stop_fn=stop_fn, save_fn=save_fn,
        writer=writer)
    assert stop_fn(result['best_reward'])
    train_collector.close()
    test_collector.close()
    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        env = gym.make(args.task)
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        print(f'Final reward: {result["rew"]}, length: {result["len"]}')
        collector.close()


if __name__ == '__main__':
    test_fn()
    test_pg()
