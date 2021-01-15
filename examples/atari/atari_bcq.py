import os
import gym
import torch
import pickle
import pprint
import argparse
import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from tianshou.env import SubprocVectorEnv
from tianshou.utils.net.discrete import DQN
from tianshou.trainer import offline_trainer
from tianshou.policy import DiscreteBCQPolicy
from tianshou.data import Collector, ReplayBuffer

from atari_wrapper import wrap_deepmind


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="PongNoFrameskip-v4")
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--eps-test", type=float, default=0.001)
    parser.add_argument("--lr", type=float, default=6.25e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--target-update-freq", type=int, default=8000)
    parser.add_argument("--unlikely-action-threshold", type=float, default=0.3)
    parser.add_argument("--imitation-logits-penalty", type=float, default=0.01)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--layer-num", type=int, default=2)
    parser.add_argument("--hidden-layer-size", type=int, default=512)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument('--frames_stack', type=int, default=4)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.)
    parser.add_argument(
        "--load-buffer-name", type=str,
        default="./expert_DQN_PongNoFrameskip-v4.hdf5",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_known_args()[0]
    return args


def make_atari_env(args):
    return wrap_deepmind(args.task, frame_stack=args.frames_stack)


def make_atari_env_watch(args):
    return wrap_deepmind(args.task, frame_stack=args.frames_stack,
                         episode_life=False, clip_rewards=False)


class Net(nn.Module):
    def __init__(self, preprocess_net, action_shape, hidden_layer_size):
        super().__init__()
        self.preprocess = preprocess_net
        self.last = nn.Sequential(
            nn.Linear(3136, hidden_layer_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_layer_size, np.prod(action_shape))
        )

    def forward(self, s, state=None, **kwargs):
        feature, h = self.preprocess(s, state)
        return self.last(feature), h


def test_discrete_bcq(args=get_args()):
    # envs
    env = make_atari_env(args)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # should be N_FRAMES x H x W
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    # make environments
    test_envs = SubprocVectorEnv([lambda: make_atari_env_watch(args)
                                  for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    test_envs.seed(args.seed)
    # model
    feature_net = DQN(*args.state_shape, args.action_shape,
                      args.device, features_only=True).to(args.device)
    policy_net = Net(feature_net, args.action_shape,
                     args.hidden_layer_size).to(args.device)
    imitation_net = Net(feature_net, args.action_shape,
                        args.hidden_layer_size).to(args.device)
    print(feature_net)
    print(policy_net)
    print(imitation_net)
    optim = torch.optim.Adam(
        list(set(policy_net).union(imitation_net)), lr=args.lr
    )

    policy = DiscreteBCQPolicy(
        policy_net, imitation_net, optim, args.gamma, args.n_step,
        args.target_update_freq, args.eps_test,
        args.unlikely_action_threshold, args.imitation_logits_penalty,
    )
    # buffer
    assert os.path.exists(args.load_buffer_name), \
        "Please run atari_dqn.py first to get expert's data buffer."
    if args.load_buffer_name.endswith('.pkl'):
        buffer = pickle.load(open(args.load_buffer_name, "rb"))
    elif args.load_buffer_name.endswith('.hdf5'):
        buffer = ReplayBuffer.load_hdf5(args.load_buffer_name)
    else:
        print(f"Unknown buffer format: {args.load_buffer_name}")
        exit(0)

    # collector
    test_collector = Collector(policy, test_envs)

    log_path = os.path.join(args.logdir, args.task, 'discrete_bcq')
    writer = SummaryWriter(log_path)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(mean_rewards):
        return False

    result = offline_trainer(
        policy, buffer, test_collector,
        args.epoch, args.step_per_epoch, args.test_num, args.batch_size,
        stop_fn=stop_fn, save_fn=save_fn, writer=writer,
    )

    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        env = gym.make(args.task)
        policy.eval()
        policy.set_eps(args.eps_test)
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        print(f'Final reward: {result["rew"]}, length: {result["len"]}')


if __name__ == "__main__":
    test_discrete_bcq(get_args())
