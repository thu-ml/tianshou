import argparse
import os
import pprint

import numpy as np
import torch
from env import Env
from network import DQN
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import ShmemVectorEnv
from tianshou.policy import A2CPolicy, ICMPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.discrete import Actor, Critic, IntrinsicCuriosityModule


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='D2_navigation')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=2000000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--step-per-epoch', type=int, default=100000)
    parser.add_argument('--episode-per-collect', type=int, default=10)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--update-per-step', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[512])
    parser.add_argument('--training-num', type=int, default=10)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument('--frames-stack', type=int, default=4)
    parser.add_argument('--skip-num', type=int, default=4)
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument(
        '--watch',
        default=False,
        action='store_true',
        help='watch the play of pre-trained policy only'
    )
    parser.add_argument(
        '--save-lmp',
        default=False,
        action='store_true',
        help='save lmp file for replay whole episode'
    )
    parser.add_argument('--save-buffer-name', type=str, default=None)
    parser.add_argument(
        '--icm-lr-scale',
        type=float,
        default=0.,
        help='use intrinsic curiosity module with this lr scale'
    )
    parser.add_argument(
        '--icm-reward-scale',
        type=float,
        default=0.01,
        help='scaling factor for intrinsic curiosity reward'
    )
    parser.add_argument(
        '--icm-forward-loss-weight',
        type=float,
        default=0.2,
        help='weight for the forward model loss in ICM'
    )
    return parser.parse_args()


def test_a2c(args=get_args()):
    args.cfg_path = f"maps/{args.task}.cfg"
    args.wad_path = f"maps/{args.task}.wad"
    args.res = (args.skip_num, 84, 84)
    env = Env(args.cfg_path, args.frames_stack, args.res)
    args.state_shape = args.res
    args.action_shape = env.action_space.shape or env.action_space.n
    # should be N_FRAMES x H x W
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    # make environments
    train_envs = ShmemVectorEnv(
        [
            lambda: Env(args.cfg_path, args.frames_stack, args.res)
            for _ in range(args.training_num)
        ]
    )
    test_envs = ShmemVectorEnv(
        [
            lambda: Env(args.cfg_path, args.frames_stack, args.res, args.save_lmp)
            for _ in range(min(os.cpu_count() - 1, args.test_num))
        ]
    )
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # define model
    net = DQN(
        *args.state_shape, args.action_shape, device=args.device, features_only=True
    )
    actor = Actor(
        net, args.action_shape, hidden_sizes=args.hidden_sizes, device=args.device
    )
    critic = Critic(net, hidden_sizes=args.hidden_sizes, device=args.device)
    optim = torch.optim.Adam(ActorCritic(actor, critic).parameters(), lr=args.lr)
    # define policy
    dist = torch.distributions.Categorical
    policy = A2CPolicy(actor, critic, optim, dist).to(args.device)
    if args.icm_lr_scale > 0:
        feature_net = DQN(
            *args.state_shape,
            args.action_shape,
            device=args.device,
            features_only=True
        )
        action_dim = np.prod(args.action_shape)
        feature_dim = feature_net.output_dim
        icm_net = IntrinsicCuriosityModule(
            feature_net.net,
            feature_dim,
            action_dim,
            hidden_sizes=args.hidden_sizes,
            device=args.device
        )
        icm_optim = torch.optim.adam(icm_net.parameters(), lr=args.lr)
        policy = ICMPolicy(
            policy, icm_net, icm_optim, args.icm_lr_scale, args.icm_reward_scale,
            args.icm_forward_loss_weight
        ).to(args.device)
    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)
    # replay buffer: `save_last_obs` and `stack_num` can be removed together
    # when you have enough RAM
    buffer = VectorReplayBuffer(
        args.buffer_size,
        buffer_num=len(train_envs),
        ignore_obs_next=True,
        save_only_last_obs=True,
        stack_num=args.frames_stack
    )
    # collector
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # log
    log_path = os.path.join(args.logdir, args.task, 'a2c')
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(mean_rewards):
        return False

    def watch():
        # watch agent's performance
        print("Setup test envs ...")
        policy.eval()
        test_envs.seed(args.seed)
        if args.save_buffer_name:
            print(f"Generate buffer with size {args.buffer_size}")
            buffer = VectorReplayBuffer(
                args.buffer_size,
                buffer_num=len(test_envs),
                ignore_obs_next=True,
                save_only_last_obs=True,
                stack_num=args.frames_stack
            )
            collector = Collector(policy, test_envs, buffer, exploration_noise=True)
            result = collector.collect(n_step=args.buffer_size)
            print(f"Save buffer into {args.save_buffer_name}")
            # Unfortunately, pickle will cause oom with 1M buffer size
            buffer.save_hdf5(args.save_buffer_name)
        else:
            print("Testing agent ...")
            test_collector.reset()
            result = test_collector.collect(
                n_episode=args.test_num, render=args.render
            )
        rew = result["rews"].mean()
        lens = result["lens"].mean() * args.skip_num
        print(f'Mean reward (over {result["n/ep"]} episodes): {rew}')
        print(f'Mean length (over {result["n/ep"]} episodes): {lens}')

    if args.watch:
        watch()
        exit(0)

    # test train_collector and start filling replay buffer
    train_collector.collect(n_step=args.batch_size * args.training_num)
    # trainer
    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.repeat_per_collect,
        args.test_num,
        args.batch_size,
        episode_per_collect=args.episode_per_collect,
        stop_fn=stop_fn,
        save_fn=save_fn,
        logger=logger,
        test_in_train=False
    )

    pprint.pprint(result)
    watch()


if __name__ == '__main__':
    test_a2c(get_args())
