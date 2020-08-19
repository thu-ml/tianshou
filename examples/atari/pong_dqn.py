import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import DQNPolicy
from tianshou.env import SubprocVectorEnv
from tianshou.utils.net.discrete import DQN
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, ReplayBuffer

from atari import create_atari_environment, preprocess_fn


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Pong')
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--target-update-freq', type=int, default=320)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--collect-per-step', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--layer-num', type=int, default=3)
    parser.add_argument('--training-num', type=int, default=8)
    parser.add_argument('--test-num', type=int, default=8)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def test_dqn(args=get_args()):
    env = create_atari_environment(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.env.action_space.shape or env.env.action_space.n
    # train_envs = gym.make(args.task)
    train_envs = SubprocVectorEnv([
        lambda: create_atari_environment(args.task)
        for _ in range(args.training_num)])
    # test_envs = gym.make(args.task)
    test_envs = SubprocVectorEnv([
        lambda: create_atari_environment(args.task)
        for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net = DQN(
        args.state_shape[0], args.state_shape[1],
        args.action_shape, args.device)
    net = net.to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    policy = DQNPolicy(
        net, optim, args.gamma, args.n_step,
        target_update_freq=args.target_update_freq)
    # collector
    train_collector = Collector(
        policy, train_envs, ReplayBuffer(args.buffer_size),
        preprocess_fn=preprocess_fn)
    test_collector = Collector(policy, test_envs, preprocess_fn=preprocess_fn)
    # policy.set_eps(1)
    train_collector.collect(n_step=args.batch_size * 4)
    print(len(train_collector.buffer))
    # log
    writer = SummaryWriter(args.logdir + '/' + 'dqn')

    def stop_fn(x):
        if env.env.spec.reward_threshold:
            return x >= env.spec.reward_threshold
        else:
            return False

    def train_fn(x):
        policy.set_eps(args.eps_train)

    def test_fn(x):
        policy.set_eps(args.eps_test)

    # trainer
    result = offpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.test_num,
        args.batch_size, train_fn=train_fn, test_fn=test_fn,
        stop_fn=stop_fn, writer=writer)

    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        env = create_atari_environment(args.task)
        collector = Collector(policy, env, preprocess_fn=preprocess_fn)
        result = collector.collect(n_episode=1, render=args.render)
        print(f'Final reward: {result["rew"]}, length: {result["len"]}')


if __name__ == '__main__':
    test_dqn(get_args())
