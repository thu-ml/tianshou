#!/usr/bin/env python3

import argparse
import datetime
import os
import pprint

import gym
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import (
    Collector,
    ReplayBuffer,
    RolloutsCollector,
    SimpleReplayBuffer,
    VectorReplayBuffer,
)
from tianshou.env import SubprocVectorEnv
from tianshou.env.fake import FakeEnv, GaussianModel
from tianshou.env.mujoco.static import TERMINAL_FUNCTIONS
from tianshou.policy import MBPOPolicy
from tianshou.trainer import dyna_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import EnsembleMLPGaussian, Net
from tianshou.utils.net.continuous import ActorProb, Critic


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Hopper-v2')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=1000000)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[256, 256])
    parser.add_argument(
        '--model-hidden-sizes', type=int, nargs='*', default=[200, 200, 200, 200]
    )
    parser.add_argument(
        '--model-net-decays',
        type=float,
        nargs='*',
        default=[0.000025, 0.00005, 0.000075, 0.000075, 0.0001]
    )
    parser.add_argument('--ensemble-size', type=int, default=7)
    parser.add_argument('--num-elites', type=int, default=5)
    parser.add_argument('--actor-lr', type=float, default=1e-3)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--model-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--auto-alpha', default=False, action='store_true')
    parser.add_argument('--alpha-lr', type=float, default=3e-4)
    parser.add_argument("--start-timesteps", type=int, default=5000)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--step-per-collect', type=int, default=1)
    parser.add_argument('--update-per-step', type=float, default=20.)
    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--real-ratio', type=float, default=0.1)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--rollout-batch-size', type=int, default=100000)
    parser.add_argument(
        '--rollout-schedule', type=int, nargs='*', default=[1, 100, 1, 1]
    )
    parser.add_argument('--model-train-freq', type=int, default=250)
    parser.add_argument('--model-retain-epochs', type=int, default=1)
    parser.add_argument('--deterministic', default=False, action='store_true')
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
    )
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument(
        '--watch',
        default=False,
        action='store_true',
        help='watch the play of pre-trained policy only'
    )
    return parser.parse_args()


def test_mbpo(args=get_args()):
    env = gym.make(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))
    if args.training_num > 1:
        train_envs = SubprocVectorEnv(
            [lambda: gym.make(args.task) for _ in range(args.training_num)]
        )
    else:
        train_envs = gym.make(args.task)
    test_envs = SubprocVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.test_num)]
    )
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net_a = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = ActorProb(
        net_a,
        args.action_shape,
        max_action=args.max_action,
        device=args.device,
        unbounded=True,
        conditioned_sigma=True
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c1 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device
    )
    net_c2 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device
    )
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)
    model_net = EnsembleMLPGaussian(
        args.ensemble_size,
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.model_hidden_sizes,
        activation=nn.SiLU,
        device=args.device
    ).to(args.device)
    assert len(args.model_net_decays) == len(args.model_hidden_sizes) + 1
    parameters = []
    layer = -1
    for name, param in model_net.named_parameters():
        if name.endswith('.weight'):
            layer += 1
            option = {
                'params': param,
                'weight_decay': args.model_net_decays[layer],
            }
            parameters.append(option)
        else:
            parameters.append({'params': param})
    model_net_optim = torch.optim.Adam(
        parameters,
        lr=args.model_lr,
    )
    model = GaussianModel(
        args.ensemble_size,
        model_net,
        model_net_optim,
        device=args.device,
        num_elites=args.num_elites,
        batch_size=args.batch_size,
        deterministic=args.deterministic
    )
    domain = args.task.split("-")[0]
    terminal_fn = TERMINAL_FUNCTIONS[domain]

    if args.auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    policy = MBPOPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        estimation_step=args.n_step,
        action_space=env.action_space
    )

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    # collector
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(args.buffer_size)
    fake_env = FakeEnv(model, buffer, terminal_fn, args.rollout_batch_size)
    model_buffer = SimpleReplayBuffer(args.rollout_batch_size)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    model_collector = RolloutsCollector(policy, fake_env, model_buffer)
    # log
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'seed_{args.seed}_{t0}-{args.task.replace("-", "_")}_mbpo'
    log_path = os.path.join(args.logdir, args.task, 'mbpo', log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    if not args.watch:
        # trainer
        result = dyna_trainer(
            policy,
            model,
            train_collector,
            test_collector,
            model_collector,
            args.epoch,
            args.step_per_epoch,
            args.step_per_collect,
            args.test_num,
            args.batch_size,
            args.rollout_batch_size,
            args.rollout_schedule,
            args.real_ratio,
            args.start_timesteps,
            model_train_freq=args.model_train_freq,
            model_retain_epochs=args.model_retain_epochs,
            update_per_step=args.update_per_step,
            save_fn=save_fn,
            logger=logger,
            test_in_train=False
        )
        torch.save(model.network.state_dict(), os.path.join(log_path, 'model.pth'))
        pprint.pprint(result)

    # Let's watch its performance!
    policy.eval()
    test_envs.seed(args.seed)
    test_collector.reset()
    result = test_collector.collect(n_episode=args.test_num, render=args.render)
    print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')


if __name__ == '__main__':
    test_mbpo()
