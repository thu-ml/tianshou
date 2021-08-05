import os
import torch
import pickle
import pprint
import datetime
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical

from tianshou.utils import TensorboardLogger
from tianshou.env import SubprocVectorEnv
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.discrete import Actor, Critic, Discriminator
from tianshou.policy import GAILPolicy
from tianshou.data import Collector, VectorReplayBuffer

from atari_network import DQN
from atari_wrapper import wrap_deepmind


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="PongNoFrameskip-v4")
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument('--buffer-size', type=int, default=10000)
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--disc-lr", type=float, default=0.0001)
    parser.add_argument("--disc-repeat", type=int, default=5)
    parser.add_argument("--disc-hidden-sizes", type=int, nargs="*", default=[128])
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument('--step-per-epoch', type=int, default=100000)
    parser.add_argument('--step-per-collect', type=int, default=500)
    parser.add_argument('--repeat-per-collect', type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[512])
    parser.add_argument('--training-num', type=int, default=10)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument('--frames-stack', type=int, default=4)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--watch", default=False, action="store_true",
                        help="watch the play of pre-trained policy only")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument(
        "--load-buffer-name", type=str,
        default="./expert_DQN_PongNoFrameskip-v4.hdf5")
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_known_args()[0]
    return args


def make_atari_env(args):
    return wrap_deepmind(args.task, frame_stack=args.frames_stack)


def make_atari_env_watch(args):
    return wrap_deepmind(args.task, frame_stack=args.frames_stack,
                         episode_life=False, clip_rewards=False)


def test_gail(args=get_args()):
    # envs
    env = make_atari_env(args)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # should be N_FRAMES x H x W
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    # make environments
    train_envs = SubprocVectorEnv([lambda: make_atari_env(args)
                                   for _ in range(args.training_num)])
    test_envs = SubprocVectorEnv([lambda: make_atari_env_watch(args)
                                  for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    test_envs.seed(args.seed)
    # model
    feature_net = DQN(*args.state_shape, args.action_shape,
                      device=args.device, features_only=True).to(args.device)
    actor = Actor(feature_net, args.action_shape, device=args.device,
                  hidden_sizes=args.hidden_sizes, softmax_output=False).to(args.device)
    critic = Critic(feature_net, hidden_sizes=args.hidden_sizes,
                    device=args.device).to(args.device)
    optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()),
                             lr=args.lr)
    disc = Discriminator(feature_net, args.action_shape, hidden_sizes=args.disc_hidden_sizes,
                         device=args.device).to(args.device)
    disc_optim = torch.optim.Adam(disc.parameters(), lr=args.disc_lr)
    # buffer
    assert os.path.exists(args.load_buffer_name), \
        "Please run atari_qrdqn.py first to get expert's data buffer."
    if args.load_buffer_name.endswith('.pkl'):
        expert_buffer = pickle.load(open(args.load_buffer_name, "rb"))
    elif args.load_buffer_name.endswith('.hdf5'):
        expert_buffer = VectorReplayBuffer.load_hdf5(args.load_buffer_name)
    else:
        print(f"Unknown buffer format: {args.load_buffer_name}")
        exit(0)
    # define policy

    def dist(logits):
        return Categorical(logits=logits)

    policy = GAILPolicy(
        actor, critic, optim, dist, expert_buffer, disc,
        disc_optim, disc_repeat=args.disc_repeat,
        action_space=env.action_space, reward_normalization=True
    ).to(args.device)
    print("action_type=", policy.action_type, "rew_norm=", policy._rew_norm)
    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(
            args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)
    # buffer
    buffer = VectorReplayBuffer(
        args.buffer_size, buffer_num=args.training_num, ignore_obs_next=True,
        save_only_last_obs=True, stack_num=args.frames_stack)

    # collector
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    # log
    log_path = os.path.join(
        args.logdir, args.task, 'gail',
        f'seed_{args.seed}_{datetime.datetime.now().strftime("%m%d-%H%M%S")}')
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(mean_rewards):
        return False

    # watch agent's performance
    def watch():
        print("Setup test envs ...")
        policy.eval()
        test_envs.seed(args.seed)
        print("Testing agent ...")
        test_collector.reset()
        result = test_collector.collect(n_episode=args.test_num,
                                        render=args.render)
        pprint.pprint(result)
        rew = result["rews"].mean()
        print(f'Mean reward (over {result["n/ep"]} episodes): {rew}')

    if args.watch:
        watch()
        exit(0)

    result = onpolicy_trainer(
        policy, train_collector, test_collector, args.epoch, args.step_per_epoch,
        args.repeat_per_collect, args.test_num, args.batch_size,
        step_per_collect=args.step_per_collect,
        stop_fn=stop_fn, save_fn=save_fn, logger=logger)

    pprint.pprint(result)
    watch()


if __name__ == "__main__":
    test_gail(get_args())
