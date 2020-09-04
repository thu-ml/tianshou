import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.env import SubprocVectorEnv
from tianshou.trainer import onpolicy_trainer, offpolicy_trainer
from tianshou.data import Collector, ReplayBuffer

import gym
from tianshou.env import VectorEnv
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.split(__file__)[0], os.pardir)) + '/tianshou/policy/psrl')
from psrl import PSRLPolicy, PSRLModel


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='FrozenLake-v0')
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--eps-test', type=float, default=0)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--buffer-size', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step-per-epoch', type=int, default=100)
    parser.add_argument('--collect-per-step', type=int, default=100)
    parser.add_argument('--repeat-per-collect', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--training-num', type=int, default=8)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    args = parser.parse_known_args()[0]
    return args


def test_psrl(args=get_args()):
    env = gym.make(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.env.action_space.shape or env.env.action_space.n
    # train_envs = gym.make(args.task)
    # train_envs = gym.make(args.task)
    train_envs = VectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.training_num)])
    # test_envs = gym.make(args.task)
    test_envs = SubprocVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    n_action = args.action_shape
    n_state = args.state_shape
    p_pri = 1e-3 * np.ones((n_action, n_state, n_state))
    rew_mean = np.zeros((n_state, n_action))
    rew_std = np.ones((n_state, n_action))
    model = PSRLModel(p_pri, rew_mean, rew_std)
    policy = PSRLPolicy(model)
    # collector
    train_collector = Collector(
        policy, train_envs, ReplayBuffer(args.buffer_size))
    test_collector = Collector(policy, test_envs)
    # log
    writer = SummaryWriter(args.logdir + '/' + 'FrozenLake')

    def train_fn(x):
        policy.set_eps(args.eps_train)

    def test_fn(x):
        policy.set_eps(args.eps_test)

    def stop_fn(x):
        if env.env.spec.reward_threshold:
            return x >= env.spec.reward_threshold
        else:
            return False
    # trainer
    result = offpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step,
        args.test_num, args.batch_size, train_fn=train_fn, test_fn=test_fn,
        stop_fn=stop_fn, writer=writer)

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
    test_psrl()
