import argparse
import os
import pprint

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import PSRLPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import LazyLogger, TensorboardLogger, WandbLogger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='NChain-v0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--buffer-size', type=int, default=50000)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--episode-per-collect', type=int, default=1)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.0)
    parser.add_argument('--rew-mean-prior', type=float, default=0.0)
    parser.add_argument('--rew-std-prior', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--eps', type=float, default=0.01)
    parser.add_argument('--add-done-loop', action="store_true", default=False)
    parser.add_argument(
        '--logger',
        type=str,
        default="wandb",
        choices=["wandb", "tensorboard", "none"],
    )
    return parser.parse_known_args()[0]


def test_psrl(args=get_args()):
    env = gym.make(args.task)
    if args.task == "NChain-v0":
        env.spec.reward_threshold = 3400
        # env.spec.reward_threshold = 3647  # described in PSRL paper
    print("reward threshold:", env.spec.reward_threshold)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # train_envs = gym.make(args.task)
    train_envs = DummyVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.training_num)]
    )
    # test_envs = gym.make(args.task)
    test_envs = SubprocVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.test_num)]
    )
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
        args.add_done_loop
    )
    # collector
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True
    )
    test_collector = Collector(policy, test_envs)
    # Logger
    if args.logger == "wandb":
        logger = WandbLogger(
            save_interval=1, project='psrl', name='wandb_test', config=args
        )
    elif args.logger == "tensorboard":
        log_path = os.path.join(args.logdir, args.task, 'psrl')
        writer = SummaryWriter(log_path)
        writer.add_text("args", str(args))
        logger = TensorboardLogger(writer)
    else:
        logger = LazyLogger()

    def stop_fn(mean_rewards):
        if env.spec.reward_threshold:
            return mean_rewards >= env.spec.reward_threshold
        else:
            return False

    train_collector.collect(n_step=args.buffer_size, random=True)
    # trainer, test it without logger
    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        1,
        args.test_num,
        0,
        episode_per_collect=args.episode_per_collect,
        stop_fn=stop_fn,
        logger=logger,
        test_in_train=False,
    )

    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        policy.eval()
        test_envs.seed(args.seed)
        test_collector.reset()
        result = test_collector.collect(n_episode=args.test_num, render=args.render)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")
    elif env.spec.reward_threshold:
        assert result["best_reward"] >= env.spec.reward_threshold


if __name__ == '__main__':
    test_psrl()
