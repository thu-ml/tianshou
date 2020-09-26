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
    parser.add_argument('--buffer-size', type=int, default=50000)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--step-per-epoch', type=int, default=5)
    parser.add_argument('--collect-per-step', type=int, default=1)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.0)
    parser.add_argument('--rew-mean-prior', type=float, default=0.0)
    parser.add_argument('--rew-std-prior', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--eps', type=float, default=0.01)
    parser.add_argument('--add-done-loop', action='store_true')
    return parser.parse_known_args()[0]


def test_psrl(args=get_args()):
    env = gym.make(args.task)
    if args.task == "NChain-v0":
        env.spec.reward_threshold = 3647  # described in PSRL paper
    print("reward threshold:", env.spec.reward_threshold)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
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
    trans_count_prior = np.ones((n_state, n_action, n_state))
    rew_mean_prior = np.full((n_state, n_action), args.rew_mean_prior)
    rew_std_prior = np.full((n_state, n_action), args.rew_std_prior)
    policy = PSRLPolicy(
        trans_count_prior, rew_mean_prior, rew_std_prior, args.gamma, args.eps,
        args.add_done_loop)
    # collector
    train_collector = Collector(
        policy, train_envs, ReplayBuffer(args.buffer_size))
    test_collector = Collector(policy, test_envs)
    # log
    writer = SummaryWriter(args.logdir + '/' + args.task)

    def stop_fn(mean_rewards):
        if env.spec.reward_threshold:
            return mean_rewards >= env.spec.reward_threshold
        else:
            return False

    train_collector.collect(n_step=args.buffer_size, random=True)
    # trainer
    result = onpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, 1,
        args.test_num, 0, stop_fn=stop_fn, writer=writer,
        test_in_train=False)

    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        policy.eval()
        test_envs.seed(args.seed)
        test_collector.reset()
        result = test_collector.collect(n_episode=[1] * args.test_num,
                                        render=args.render)
        print(f'Final reward: {result["rew"]}, length: {result["len"]}')
    elif env.spec.reward_threshold:
        assert result["best_reward"] >= env.spec.reward_threshold


if __name__ == '__main__':
    test_psrl()
