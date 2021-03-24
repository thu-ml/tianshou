#!/usr/bin/env python3

import os
import gym
import torch
import datetime
import argparse
import numpy as np
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Independent, Normal

from tianshou.policy import PGPolicy
from tianshou.utils import BasicLogger
from tianshou.env import SubprocVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.continuous import ActorProb
from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='HalfCheetah-v3')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=4096)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[64, 64])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step-per-epoch', type=int, default=30000)
    parser.add_argument('--step-per-collect', type=int, default=2048)
    parser.add_argument('--repeat-per-collect', type=int, default=1)
    # batch-size >> step-per-collect means caculating all data in one singe forward.
    parser.add_argument('--batch-size', type=int, default=99999)
    parser.add_argument('--training-num', type=int, default=64)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume-path', type=str, default=None)
    # reinforce special
    parser.add_argument('--rew-norm', type=int, default=True)
    # "clipping" also works well.
    parser.add_argument('--action-bound-method', type=str, default="tanh")
    parser.add_argument('--lr-decay', type=int, default=True)
    return parser.parse_args()


def test_reinforce(args=get_args()):
    env = gym.make(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low),
          np.max(env.action_space.high))
    # train_envs = gym.make(args.task)
    train_envs = SubprocVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.training_num)],
        norm_obs=True)
    # test_envs = gym.make(args.task)
    test_envs = SubprocVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.test_num)],
        norm_obs=True, obs_rms=train_envs.obs_rms, update_obs_rms=False)

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net_a = Net(args.state_shape, hidden_sizes=args.hidden_sizes,
                activation=nn.Tanh, device=args.device)
    actor = ActorProb(net_a, args.action_shape, max_action=args.max_action,
                      unbounded=True, device=args.device).to(args.device)
    torch.nn.init.constant_(actor.sigma_param, -0.5)
    for m in actor.modules():
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
    # do last policy layer scaling, this will make initial actions have (close to)
    # 0 mean and std, and will help boost performances,
    # see https://arxiv.org/abs/2006.05990, Fig.24 for details
    for m in actor.mu.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)

    optim = torch.optim.Adam(actor.parameters(), lr=args.lr)
    lr_scheduler = None
    if args.lr_decay:
        # decay learning rate to 0 linearly
        max_update_num = np.ceil(
            args.step_per_epoch / args.step_per_collect) * args.epoch

        lr_scheduler = LambdaLR(
            optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num)

    def dist(*logits):
        return Independent(Normal(*logits), 1)

    policy = PGPolicy(actor, optim, dist, discount_factor=args.gamma,
                      reward_normalization=args.rew_norm, action_scaling=True,
                      action_bound_method=args.action_bound_method,
                      lr_scheduler=lr_scheduler, action_space=env.action_space)

    # collector
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(args.buffer_size)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    # log
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'seed_{args.seed}_{t0}-{args.task.replace("-", "_")}_reinforce'
    log_path = os.path.join(args.logdir, args.task, 'reinforce', log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = BasicLogger(writer, update_interval=10)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    # trainer
    result = onpolicy_trainer(
        policy, train_collector, test_collector, args.epoch, args.step_per_epoch,
        args.repeat_per_collect, args.test_num, args.batch_size,
        step_per_collect=args.step_per_collect, save_fn=save_fn, logger=logger,
        test_in_train=False)

    # Let's watch its performance!
    policy.eval()
    test_envs.seed(args.seed)
    test_collector.reset()
    result = test_collector.collect(n_episode=args.test_num, render=args.render)
    print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')


if __name__ == '__main__':
    test_reinforce()
