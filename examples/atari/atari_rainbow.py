import argparse
import datetime
import os
import pprint
import sys

import numpy as np
import torch

from tianshou.algorithm import C51, RainbowDQN
from tianshou.algorithm.algorithm_base import Algorithm
from tianshou.algorithm.modelfree.c51 import C51Policy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import (
    Collector,
    CollectStats,
    PrioritizedVectorReplayBuffer,
    VectorReplayBuffer,
)
from tianshou.env.atari.atari_network import RainbowNet
from tianshou.env.atari.atari_wrapper import make_atari_env
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.trainer import OffPolicyTrainerParams


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="PongNoFrameskip-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale_obs", type=int, default=0)
    parser.add_argument("--eps_test", type=float, default=0.005)
    parser.add_argument("--eps_train", type=float, default=1.0)
    parser.add_argument("--eps_train_final", type=float, default=0.05)
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.0000625)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--num_atoms", type=int, default=51)
    parser.add_argument("--v_min", type=float, default=-10.0)
    parser.add_argument("--v_max", type=float, default=10.0)
    parser.add_argument("--noisy_std", type=float, default=0.1)
    parser.add_argument("--no_dueling", action="store_true", default=False)
    parser.add_argument("--no_noisy", action="store_true", default=False)
    parser.add_argument("--no_priority", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--beta_final", type=float, default=1.0)
    parser.add_argument("--beta_anneal_step", type=int, default=5000000)
    parser.add_argument("--no_weight_norm", action="store_true", default=False)
    parser.add_argument("--n_step", type=int, default=3)
    parser.add_argument("--target_update_freq", type=int, default=500)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--epoch_num_steps", type=int, default=100000)
    parser.add_argument("--collection_step_num_env_steps", type=int, default=10)
    parser.add_argument("--update_per_step", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_train_envs", type=int, default=10)
    parser.add_argument("--num_test_envs", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--frames_stack", type=int, default=4)
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--resume_id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb_project", type=str, default="atari.benchmark")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    parser.add_argument("--save_buffer_name", type=str, default=None)
    return parser.parse_args()


def test_rainbow(args: argparse.Namespace = get_args()) -> None:
    env, train_envs, test_envs = make_atari_env(
        args.task,
        args.seed,
        args.num_train_envs,
        args.num_test_envs,
        scale=args.scale_obs,
        frame_stack=args.frames_stack,
    )
    args.state_shape = env.observation_space.shape or env.observation_space.n  # type: ignore
    args.action_shape = env.action_space.shape or env.action_space.n  # type: ignore
    # should be N_FRAMES x H x W
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # define model
    c, h, w = args.state_shape
    net = RainbowNet(
        c=c,
        h=h,
        w=w,
        action_shape=args.action_shape,
        num_atoms=args.num_atoms,
        noisy_std=args.noisy_std,
        is_dueling=not args.no_dueling,
        is_noisy=not args.no_noisy,
    )

    # define policy and algorithm
    policy = C51Policy(
        model=net,
        action_space=env.action_space,
        num_atoms=args.num_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
        eps_training=args.eps_train,
        eps_inference=args.eps_test,
    )
    optim = AdamOptimizerFactory(lr=args.lr)
    algorithm: C51 = RainbowDQN(
        policy=policy,
        optim=optim,
        gamma=args.gamma,
        n_step_return_horizon=args.n_step,
        target_update_freq=args.target_update_freq,
    ).to(args.device)

    # load a previous policy
    if args.resume_path:
        algorithm.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    # replay buffer: `save_last_obs` and `stack_num` can be removed together
    # when you have enough RAM
    buffer: VectorReplayBuffer | PrioritizedVectorReplayBuffer
    if args.no_priority:
        buffer = VectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs),
            ignore_obs_next=True,
            save_only_last_obs=True,
            stack_num=args.frames_stack,
        )
    else:
        buffer = PrioritizedVectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs),
            ignore_obs_next=True,
            save_only_last_obs=True,
            stack_num=args.frames_stack,
            alpha=args.alpha,
            beta=args.beta,
            weight_norm=not args.no_weight_norm,
        )

    # collector
    train_collector = Collector[CollectStats](algorithm, train_envs, buffer, exploration_noise=True)
    test_collector = Collector[CollectStats](algorithm, test_envs, exploration_noise=True)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "rainbow"
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
        if env.spec.reward_threshold:  # type: ignore
            return mean_rewards >= env.spec.reward_threshold  # type: ignore
        if "Pong" in args.task:
            return mean_rewards >= 20
        return False

    def train_fn(epoch: int, env_step: int) -> None:
        # nature DQN setting, linear decay in the first 1M steps
        if env_step <= 1e6:
            eps = args.eps_train - env_step / 1e6 * (args.eps_train - args.eps_train_final)
        else:
            eps = args.eps_train_final
        policy.set_eps_training(eps)
        if env_step % 1000 == 0:
            logger.write("train/env_step", env_step, {"train/eps": eps})
        if not args.no_priority:
            if env_step <= args.beta_anneal_step:
                beta = args.beta - env_step / args.beta_anneal_step * (args.beta - args.beta_final)
            else:
                beta = args.beta_final
            buffer.set_beta(beta)
            if env_step % 1000 == 0:
                logger.write("train/env_step", env_step, {"train/beta": beta})

    def watch() -> None:
        print("Setup test envs ...")
        test_envs.seed(args.seed)
        if args.save_buffer_name:
            print(f"Generate buffer with size {args.buffer_size}")
            buffer = PrioritizedVectorReplayBuffer(
                args.buffer_size,
                buffer_num=len(test_envs),
                ignore_obs_next=True,
                save_only_last_obs=True,
                stack_num=args.frames_stack,
                alpha=args.alpha,
                beta=args.beta,
            )
            collector = Collector[CollectStats](
                algorithm, test_envs, buffer, exploration_noise=True
            )
            result = collector.collect(n_step=args.buffer_size, reset_before_collect=True)
            print(f"Save buffer into {args.save_buffer_name}")
            # Unfortunately, pickle will cause oom with 1M buffer size
            buffer.save_hdf5(args.save_buffer_name)
        else:
            print("Testing agent ...")
            test_collector.reset()
            result = test_collector.collect(n_episode=args.num_test_envs, render=args.render)
        result.pprint_asdict()

    if args.watch:
        watch()
        sys.exit(0)

    # test train_collector and start filling replay buffer
    train_collector.reset()
    train_collector.collect(n_step=args.batch_size * args.num_train_envs)

    # train
    result = algorithm.run_training(
        OffPolicyTrainerParams(
            train_collector=train_collector,
            test_collector=test_collector,
            max_epochs=args.epoch,
            epoch_num_steps=args.epoch_num_steps,
            collection_step_num_env_steps=args.collection_step_num_env_steps,
            test_step_num_episodes=args.num_test_envs,
            batch_size=args.batch_size,
            train_fn=train_fn,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            update_step_num_gradient_steps_per_sample=args.update_per_step,
            test_in_train=False,
        )
    )

    pprint.pprint(result)
    watch()


if __name__ == "__main__":
    test_rainbow(get_args())
