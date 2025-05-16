#!/usr/bin/env python3

import argparse
import datetime
import os
import pprint

import numpy as np
import torch
from mujoco_env import make_mujoco_env
from torch import nn
from torch.distributions import Distribution, Independent, Normal

from tianshou.algorithm import Reinforce
from tianshou.algorithm.algorithm_base import Algorithm
from tianshou.algorithm.modelfree.reinforce import ActorPolicyProbabilistic
from tianshou.algorithm.optim import AdamOptimizerFactory, LRSchedulerFactoryLinear
from tianshou.data import Collector, CollectStats, ReplayBuffer, VectorReplayBuffer
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.trainer import OnPolicyTrainerParams
from tianshou.utils.net.common import MLPActor
from tianshou.utils.net.continuous import ContinuousActorProbabilistic


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Ant-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=4096)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=30000)
    parser.add_argument("--step-per-collect", type=int, default=2048)
    parser.add_argument("--repeat-per-collect", type=int, default=1)
    # batch-size >> step-per-collect means calculating all data in one singe forward.
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=10)
    # reinforce special
    parser.add_argument("--rew-norm", type=int, default=True)
    # "clip" option also works well.
    parser.add_argument("--action-bound-method", type=str, default="tanh")
    parser.add_argument("--lr-decay", type=int, default=True)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="mujoco.benchmark")
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
        args.training_num,
        args.test_num,
        obs_norm=True,
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
    net_a = MLPActor(
        state_shape=args.state_shape,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Tanh,
    )
    actor = ContinuousActorProbabilistic(
        preprocess_net=net_a,
        action_shape=args.action_shape,
        unbounded=True,
    ).to(args.device)
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

    optim = AdamOptimizerFactory(lr=args.lr)
    if args.lr_decay:
        optim.with_lr_scheduler_factory(
            LRSchedulerFactoryLinear(
                max_epochs=args.epoch,
                epoch_num_steps=args.step_per_epoch,
                collection_step_num_env_steps=args.step_per_collect,
            )
        )

    def dist(loc_scale: tuple[torch.Tensor, torch.Tensor]) -> Distribution:
        loc, scale = loc_scale
        return Independent(Normal(loc, scale), 1)

    policy = ActorPolicyProbabilistic(
        actor=actor,
        dist_fn=dist,
        action_space=env.action_space,
        action_scaling=True,
        action_bound_method=args.action_bound_method,
    )
    algorithm: Reinforce = Reinforce(
        policy=policy,
        optim=optim,
        gamma=args.gamma,
        return_standardization=args.rew_norm,
    )

    # load a previous policy
    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location=args.device)
        algorithm.load_state_dict(ckpt["model"])
        train_envs.set_obs_rms(ckpt["obs_rms"])
        test_envs.set_obs_rms(ckpt["obs_rms"])
        print("Loaded agent from: ", args.resume_path)

    # collector
    buffer: VectorReplayBuffer | ReplayBuffer
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(args.buffer_size)
    train_collector = Collector[CollectStats](algorithm, train_envs, buffer, exploration_noise=True)
    test_collector = Collector[CollectStats](algorithm, test_envs)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "reinforce"
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
        state = {"model": policy.state_dict(), "obs_rms": train_envs.get_obs_rms()}
        torch.save(state, os.path.join(log_path, "policy.pth"))

    if not args.watch:
        # train
        result = algorithm.run_training(
            OnPolicyTrainerParams(
                train_collector=train_collector,
                test_collector=test_collector,
                max_epochs=args.epoch,
                epoch_num_steps=args.step_per_epoch,
                update_step_num_repetitions=args.repeat_per_collect,
                test_step_num_episodes=args.test_num,
                batch_size=args.batch_size,
                collection_step_num_env_steps=args.step_per_collect,
                save_best_fn=save_best_fn,
                logger=logger,
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
