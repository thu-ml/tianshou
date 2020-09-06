import gym
import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import PSRLPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.data import Collector, ReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='NChain-v0')
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--eps-test', type=float, default=0.0)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--step-per-epoch', type=int, default=1)
    parser.add_argument('--collect-per-step', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--training-num', type=int, default=8)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    return parser.parse_known_args()[0]


def test_psrl(args=get_args()):
    env = gym.make(args.task)
    if args.task == "NChain-v0":
        env.spec.reward_threshold = 3650  # discribed in PSRL paper
    print("reward threshold:", env.spec.reward_threshold)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.env.action_space.shape or env.env.action_space.n
    # train_envs = gym.make(args.task)
    # train_envs = gym.make(args.task)
    train_envs = DummyVectorEnv(
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
    trans_count_prior = 1e-3 * np.ones((n_action, n_state, n_state))
    rew_mean_prior = np.zeros((n_state, n_action))
    rew_std_prior = np.ones((n_state, n_action))
    policy = PSRLPolicy(trans_count_prior,
                        rew_mean_prior, rew_std_prior)
    # collector
    train_collector = Collector(
        policy, train_envs, ReplayBuffer(args.buffer_size))
    test_collector = Collector(policy, test_envs)
    # log
    writer = SummaryWriter(args.logdir + '/' + args.task)

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
    result = onpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, 1,
        args.test_num, args.batch_size, train_fn=train_fn, test_fn=test_fn,
        stop_fn=stop_fn, writer=writer)

    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        policy.eval()
        policy.set_eps(args.eps_test)
        test_envs.seed(args.seed)
        test_collector.reset()
        result = test_collector.collect(n_episode=[1] * args.test_num,
                                        render=args.render)
        print(f'Final reward: {result["rew"]}, length: {result["len"]}')


if __name__ == '__main__':
    test_psrl()
