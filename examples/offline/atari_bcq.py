#!/usr/bin/env python3

import argparse
import datetime
import os
import pickle
import pprint

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from examples.atari.atari_network import DQN
from examples.atari.atari_wrapper import make_atari_env
from examples.offline.utils import load_buffer
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import DiscreteBCQPolicy
from tianshou.trainer import offline_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.discrete import Actor


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
    parser.add_argument("--update-per-epoch", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[512])
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--frames-stack", type=int, default=4)
    parser.add_argument("--scale-obs", type=int, default=0)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="offline_atari.benchmark")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only"
    )
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument(
        "--load-buffer-name", type=str, default="./expert_DQN_PongNoFrameskip-v4.hdf5"
    )
    parser.add_argument(
        "--buffer-from-rl-unplugged", action="store_true", default=False
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_known_args()[0]
    return args


def test_discrete_bcq(args=get_args()):
    # envs
    env, _, test_envs = make_atari_env(
        args.task,
        args.seed,
        1,
        args.test_num,
        scale=args.scale_obs,
        frame_stack=args.frames_stack,
    )
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # should be N_FRAMES x H x W
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # model
    feature_net = DQN(
        *args.state_shape, args.action_shape, device=args.device, features_only=True
    ).to(args.device)
    policy_net = Actor(
        feature_net,
        args.action_shape,
        device=args.device,
        hidden_sizes=args.hidden_sizes,
        softmax_output=False,
    ).to(args.device)
    imitation_net = Actor(
        feature_net,
        args.action_shape,
        device=args.device,
        hidden_sizes=args.hidden_sizes,
        softmax_output=False,
    ).to(args.device)
    actor_critic = ActorCritic(policy_net, imitation_net)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)
    # define policy
    policy = DiscreteBCQPolicy(
        policy_net, imitation_net, optim, args.gamma, args.n_step,
        args.target_update_freq, args.eps_test, args.unlikely_action_threshold,
        args.imitation_logits_penalty
    )
    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)
    # buffer
    if args.buffer_from_rl_unplugged:
        buffer = load_buffer(args.load_buffer_name)
    else:
        assert os.path.exists(args.load_buffer_name), \
            "Please run atari_dqn.py first to get expert's data buffer."
        if args.load_buffer_name.endswith(".pkl"):
            buffer = pickle.load(open(args.load_buffer_name, "rb"))
        elif args.load_buffer_name.endswith(".hdf5"):
            buffer = VectorReplayBuffer.load_hdf5(args.load_buffer_name)
        else:
            print(f"Unknown buffer format: {args.load_buffer_name}")
            exit(0)
    print("Replay buffer size:", len(buffer), flush=True)

    # collector
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "bcq"
    log_name = os.path.join(args.task, args.algo_name, str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)

    # logger
    if args.logger == "wandb":
        logger = WandbLogger(
            save_interval=1,
            name=log_name.replace(os.path.sep, "__"),
            run_id=args.resume_id,
            config=args,
            project=args.wandb_project,
        )
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    if args.logger == "tensorboard":
        logger = TensorboardLogger(writer)
    else:  # wandb
        logger.load(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards):
        return False

    # watch agent's performance
    def watch():
        print("Setup test envs ...")
        policy.eval()
        policy.set_eps(args.eps_test)
        test_envs.seed(args.seed)
        print("Testing agent ...")
        test_collector.reset()
        result = test_collector.collect(n_episode=args.test_num, render=args.render)
        pprint.pprint(result)
        rew = result["rews"].mean()
        print(f'Mean reward (over {result["n/ep"]} episodes): {rew}')

    if args.watch:
        watch()
        exit(0)

    result = offline_trainer(
        policy,
        buffer,
        test_collector,
        args.epoch,
        args.update_per_epoch,
        args.test_num,
        args.batch_size,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
    )

    pprint.pprint(result)
    watch()


if __name__ == "__main__":
    test_discrete_bcq(get_args())
