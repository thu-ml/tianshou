#!/usr/bin/env python3

import argparse
import datetime
import os
import pprint

import numpy as np
import torch
from mujoco_env import make_mujoco_env

from tianshou.algorithm import REDQ
from tianshou.algorithm.algorithm_base import Algorithm
from tianshou.algorithm.modelfree.redq import REDQPolicy
from tianshou.algorithm.modelfree.sac import AutoAlpha
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Collector, CollectStats, ReplayBuffer, VectorReplayBuffer
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.utils.net.common import EnsembleLinear, MLPActor
from tianshou.utils.net.continuous import ContinuousActorProbabilistic, ContinuousCritic


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Ant-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer_size", type=int, default=1000000)
    parser.add_argument("--hidden_sizes", type=int, nargs="*", default=[256, 256])
    parser.add_argument("--ensemble_size", type=int, default=10)
    parser.add_argument("--subset_size", type=int, default=2)
    parser.add_argument("--actor_lr", type=float, default=1e-3)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto_alpha", default=False, action="store_true")
    parser.add_argument("--alpha_lr", type=float, default=3e-4)
    parser.add_argument("--start_timesteps", type=int, default=10000)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--epoch_num_steps", type=int, default=5000)
    parser.add_argument("--collection_step_num_env_steps", type=int, default=1)
    parser.add_argument("--update_per_step", type=int, default=20)
    parser.add_argument("--n_step", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--target_mode", type=str, choices=("min", "mean"), default="min")
    parser.add_argument("--num_train_envs", type=int, default=1)
    parser.add_argument("--num_test_envs", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--resume_id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb_project", type=str, default="mujoco.benchmark")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    return parser.parse_args()


def main(args: argparse.Namespace = get_args()) -> None:
    env, train_envs, test_envs = make_mujoco_env(
        args.task,
        args.seed,
        args.num_train_envs,
        args.test_num,
        obs_norm=False,
    )
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # model
    net_a = MLPActor(state_shape=args.state_shape, hidden_sizes=args.hidden_sizes)
    actor = ContinuousActorProbabilistic(
        preprocess_net=net_a,
        action_shape=args.action_shape,
        unbounded=True,
        conditioned_sigma=True,
    ).to(args.device)
    actor_optim = AdamOptimizerFactory(lr=args.actor_lr)

    def linear(x: int, y: int) -> EnsembleLinear:
        return EnsembleLinear(args.ensemble_size, x, y)

    net_c = MLPActor(
        state_shape=args.state_shape,
        action_shape=args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        linear_layer=linear,
    )
    critics = ContinuousCritic(
        preprocess_net=net_c,
        linear_layer=linear,
        flatten_input=False,
    ).to(args.device)
    critics_optim = AdamOptimizerFactory(lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = 0.0
        alpha_optim = AdamOptimizerFactory(lr=args.alpha_lr)
        args.alpha = AutoAlpha(target_entropy, log_alpha, alpha_optim).to(args.device)

    policy = REDQPolicy(
        actor=actor,
        action_space=env.action_space,
    )
    algorithm: REDQ = REDQ(
        policy=policy,
        policy_optim=actor_optim,
        critic=critics,
        critic_optim=critics_optim,
        ensemble_size=args.ensemble_size,
        subset_size=args.subset_size,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        n_step_return_horizon=args.n_step,
        actor_delay=args.update_per_step,
        target_mode=args.target_mode,
    )

    # load a previous policy
    if args.resume_path:
        algorithm.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    # collector
    buffer: VectorReplayBuffer | ReplayBuffer
    if args.num_train_envs > 1:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(args.buffer_size)
    train_collector = Collector[CollectStats](algorithm, train_envs, buffer, exploration_noise=True)
    test_collector = Collector[CollectStats](algorithm, test_envs)
    train_collector.reset()
    train_collector.collect(n_step=args.start_timesteps, random=True)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "redq"
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

    if not args.watch:
        # train
        result = algorithm.run_training(
            OffPolicyTrainerParams(
                train_collector=train_collector,
                test_collector=test_collector,
                max_epochs=args.epoch,
                epoch_num_steps=args.epoch_num_steps,
                collection_step_num_env_steps=args.collection_step_num_env_steps,
                test_step_num_episodes=args.test_num,
                batch_size=args.batch_size,
                save_best_fn=save_best_fn,
                logger=logger,
                update_step_num_gradient_steps_per_sample=args.update_per_step,
                test_in_train=False,
            )
        )
        pprint.pprint(result)

    # Let's watch its performance!
    test_envs.seed(args.seed)
    test_collector.reset()
    collector_stats = test_collector.collect(n_episode=args.test_num, render=args.render)
    print(collector_stats)


if __name__ == "__main__":
    main()
