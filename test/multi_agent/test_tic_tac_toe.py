import os
import pprint
from torch.utils.tensorboard import SummaryWriter
import torch
import argparse
import numpy as np

from tianshou.env import VectorEnv
from tianshou.policy import \
    (MultiAgentDQNPolicy, MultiAgentPolicyManager, RandomMultiAgentPolicy)
from tianshou.utils.net.common import Net
from tianshou.data import Collector, ReplayBuffer
from tianshou.trainer import offpolicy_trainer

from tic_tac_toe_env import TicTacToeEnv


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=320)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--collect-per-step', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--layer-num', type=int, default=3)
    parser.add_argument('--training-num', type=int, default=8)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.1)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


def test_tic_tac_toe(args=get_args()):
    env = TicTacToeEnv()
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    train_envs = VectorEnv(
        [lambda: TicTacToeEnv() for _ in range(args.training_num)])
    test_envs = VectorEnv(
        [lambda: TicTacToeEnv() for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net = Net(args.layer_num, args.state_shape, args.action_shape, args.device)
    net = net.to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    agent_1 = MultiAgentDQNPolicy(
        net, optim, args.gamma, args.n_step,
        use_target_network=args.target_update_freq > 0,
        target_update_freq=args.target_update_freq)
    agent_2 = RandomMultiAgentPolicy()
    policy = MultiAgentPolicyManager([agent_1, agent_2])

    # collector
    train_collector = Collector(
        policy, train_envs, ReplayBuffer(args.buffer_size), reward_length=2)
    test_collector = Collector(policy, test_envs, reward_length=2)
    # policy.set_eps(1)
    train_collector.collect(n_step=args.batch_size)
    # log
    log_path = os.path.join(args.logdir, 'tic_tac_toe', 'dqn')
    writer = SummaryWriter(log_path)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(x):
        return x >= 0.95

    def train_fn(x):
        policy.policies[0].set_eps(args.eps_train)

    def test_fn(x):
        policy.policies[0].set_eps(args.eps_test)

    # trainer
    result = offpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.test_num,
        args.batch_size, train_fn=train_fn, test_fn=test_fn,
        stop_fn=stop_fn, save_fn=save_fn, writer=writer)

    assert stop_fn(result["best_reward"])
    train_collector.close()
    test_collector.close()
    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        env = TicTacToeEnv()
        collector = Collector(policy, env, reward_length=2)
        result = collector.collect(n_episode=1, render=args.render)
        print(f'Final reward: {result["rew"]}, length: {result["len"]}')
        collector.close()


if __name__ == '__main__':
    test_tic_tac_toe(get_args())
