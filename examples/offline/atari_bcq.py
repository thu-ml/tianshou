#!/usr/bin/env python3

import argparse
import datetime
import os
import pickle
import pprint
import sys

import numpy as np
import torch
from gymnasium.spaces import Discrete

from examples.offline.utils import load_buffer
from tianshou.data import Collector, CollectStats, VectorReplayBuffer
from tianshou.env.atari.atari_network import DQNet
from tianshou.env.atari.atari_wrapper import make_atari_env
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.algorithm import DiscreteBCQ
from tianshou.algorithm.base import Algorithm
from tianshou.algorithm.imitation.discrete_bcq import DiscreteBCQPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.trainer import OfflineTrainerParams
from tianshou.utils.net.discrete import DiscreteActor


def get_args() -> argparse.Namespace:
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
    parser.add_argument("--render", type=float, default=0.0)
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
        help="watch the play of pre-trained policy only",
    )
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument(
        "--load-buffer-name",
        type=str,
        default="./expert_DQN_PongNoFrameskip-v4.hdf5",
    )
    parser.add_argument("--buffer-from-rl-unplugged", action="store_true", default=False)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_known_args()[0]


def main(args: argparse.Namespace = get_args()) -> None:
    # envs
    env, _, test_envs = make_atari_env(
        args.task,
        args.seed,
        1,
        args.test_num,
        scale=args.scale_obs,
        frame_stack=args.frames_stack,
    )
    assert isinstance(env.action_space, Discrete)
    args.state_shape = env.observation_space.shape
    args.action_shape = int(env.action_space.n)
    # should be N_FRAMES x H x W
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # model
    assert args.state_shape is not None
    assert len(args.state_shape) == 3
    c, h, w = args.state_shape
    feature_net = DQNet(
        c=c,
        h=h,
        w=w,
        action_shape=args.action_shape,
        features_only=True,
    ).to(args.device)
    policy_net = DiscreteActor(
        preprocess_net=feature_net,
        action_shape=args.action_shape,
        hidden_sizes=args.hidden_sizes,
        softmax_output=False,
    ).to(args.device)
    imitation_net = DiscreteActor(
        preprocess_net=feature_net,
        action_shape=args.action_shape,
        hidden_sizes=args.hidden_sizes,
        softmax_output=False,
    ).to(args.device)
    optim = AdamOptimizerFactory(lr=args.lr)
    # define policy and algorithm
    policy = DiscreteBCQPolicy(
        model=policy_net,
        imitator=imitation_net,
        action_space=env.action_space,
        unlikely_action_threshold=args.unlikely_action_threshold,
        eps_inference=args.eps_test,
    )
    algorithm: DiscreteBCQ = DiscreteBCQ(
        policy=policy,
        optim=optim,
        gamma=args.gamma,
        estimation_step=args.n_step,
        target_update_freq=args.target_update_freq,
        imitation_logits_penalty=args.imitation_logits_penalty,
    )
    # load a previous policy
    if args.resume_path:
        algorithm.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)
    # buffer
    if args.buffer_from_rl_unplugged:
        buffer = load_buffer(args.load_buffer_name)
    else:
        assert os.path.exists(
            args.load_buffer_name,
        ), "Please run atari_dqn.py first to get expert's data buffer."
        if args.load_buffer_name.endswith(".pkl"):
            with open(args.load_buffer_name, "rb") as f:
                buffer = pickle.load(f)
        elif args.load_buffer_name.endswith(".hdf5"):
            buffer = VectorReplayBuffer.load_hdf5(args.load_buffer_name)
        else:
            print(f"Unknown buffer format: {args.load_buffer_name}")
            sys.exit(0)
    print("Replay buffer size:", len(buffer), flush=True)

    # collector
    test_collector = Collector[CollectStats](algorithm, test_envs, exploration_noise=True)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "bcq"
    log_name = os.path.join(args.task, args.algo_name, str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)

    # logger
    logger_factory = LoggerFactoryDefault()
    if args.logger == "wandb":
        logger_factory.logger_type = "wandb"
        logger_factory.wandb_project = args.wandb_project
    else:
        logger_factory.logger_type = "tensorboard"

    logger = logger_factory.create_logger(
        log_dir=log_path,
        experiment_name=log_name,
        run_id=args.resume_id,
        config_dict=vars(args),
    )

    def save_best_fn(policy: Algorithm) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards: float) -> bool:
        return False

    # watch agent's performance
    def watch() -> None:
        print("Setup test envs ...")
        policy.set_eps(args.eps_test)
        test_envs.seed(args.seed)
        print("Testing agent ...")
        test_collector.reset()
        result = test_collector.collect(n_episode=args.test_num, render=args.render)
        result.pprint_asdict()

    if args.watch:
        watch()
        sys.exit(0)

    result = algorithm.run_training(
        OfflineTrainerParams(
            buffer=buffer,
            test_collector=test_collector,
            max_epoch=args.epoch,
            step_per_epoch=args.update_per_epoch,
            episode_per_test=args.test_num,
            batch_size=args.batch_size,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
        )
    )

    pprint.pprint(result)
    watch()


if __name__ == "__main__":
    main(get_args())
