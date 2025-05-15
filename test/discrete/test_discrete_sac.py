import argparse
import os
from test.determinism_test import AlgorithmDeterminismTest

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, CollectStats, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.algorithm import DiscreteSAC
from tianshou.algorithm.base import Algorithm
from tianshou.algorithm.modelfree.discrete_sac import (
    DiscreteSACPolicy,
)
from tianshou.algorithm.modelfree.sac import AutoAlpha
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.trainer.base import OffPolicyTrainerParams
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import DiscreteActor, DiscreteCritic
from tianshou.utils.space_info import SpaceInfo


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="CartPole-v1")
    parser.add_argument("--reward-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--auto-alpha", action="store_true", default=False)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--step-per-epoch", type=int, default=10000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_known_args()[0]


def test_discrete_sac(
    args: argparse.Namespace = get_args(),
    enable_assertions: bool = True,
) -> None:
    env = gym.make(args.task)
    assert isinstance(env.action_space, gym.spaces.Discrete)

    space_info = SpaceInfo.from_env(env)
    args.state_shape = space_info.observation_info.obs_shape
    args.action_shape = space_info.action_info.action_shape

    if args.reward_threshold is None:
        default_reward_threshold = {"CartPole-v1": 170}  # lower the goal
        args.reward_threshold = default_reward_threshold.get(
            args.task,
            env.spec.reward_threshold if env.spec else None,
        )

    train_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    obs_dim = space_info.observation_info.obs_dim
    action_dim = space_info.action_info.action_dim
    net = Net(state_shape=args.state_shape, hidden_sizes=args.hidden_sizes)
    actor = DiscreteActor(
        preprocess_net=net, action_shape=args.action_shape, softmax_output=False
    ).to(args.device)
    actor_optim = AdamOptimizerFactory(lr=args.actor_lr)
    net_c1 = Net(state_shape=args.state_shape, hidden_sizes=args.hidden_sizes)
    critic1 = DiscreteCritic(preprocess_net=net_c1, last_size=action_dim).to(args.device)
    critic1_optim = AdamOptimizerFactory(lr=args.critic_lr)
    net_c2 = Net(state_shape=obs_dim, hidden_sizes=args.hidden_sizes)
    critic2 = DiscreteCritic(preprocess_net=net_c2, last_size=action_dim).to(args.device)
    critic2_optim = AdamOptimizerFactory(lr=args.critic_lr)

    # better not to use auto alpha in CartPole
    if args.auto_alpha:
        target_entropy = 0.98 * np.log(action_dim)
        log_alpha = 0.0
        alpha_optim = AdamOptimizerFactory(lr=args.alpha_lr)
        args.alpha = AutoAlpha(target_entropy, log_alpha, alpha_optim).to(args.device)

    policy = DiscreteSACPolicy(
        actor=actor,
        action_space=env.action_space,
    )
    algorithm = DiscreteSAC(
        policy=policy,
        policy_optim=actor_optim,
        critic=critic1,
        critic_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        estimation_step=args.n_step,
    )
    # collector
    train_collector = Collector[CollectStats](
        algorithm,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
    )
    test_collector = Collector[CollectStats](algorithm, test_envs)
    # train_collector.collect(n_step=args.buffer_size)
    # log
    log_path = os.path.join(args.logdir, args.task, "discrete_sac")
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy: Algorithm) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards: float) -> bool:
        return mean_rewards >= args.reward_threshold

    # train
    result = algorithm.run_training(
        OffPolicyTrainerParams(
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            step_per_collect=args.step_per_collect,
            episode_per_test=args.test_num,
            batch_size=args.batch_size,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            update_per_step=args.update_per_step,
            test_in_train=False,
        )
    )

    if enable_assertions:
        assert stop_fn(result.best_reward)


def test_discrete_sac_determinism() -> None:
    main_fn = lambda args: test_discrete_sac(args, enable_assertions=False)
    AlgorithmDeterminismTest("discrete_sac", main_fn, get_args()).run()
