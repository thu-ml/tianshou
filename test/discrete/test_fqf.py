import argparse
import os
from test.determinism_test import AlgorithmDeterminismTest

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.algorithm import FQF
from tianshou.algorithm.algorithm_base import Algorithm
from tianshou.algorithm.modelfree.fqf import FQFPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory, RMSpropOptimizerFactory
from tianshou.data import (
    Collector,
    CollectStats,
    PrioritizedVectorReplayBuffer,
    ReplayBuffer,
    VectorReplayBuffer,
)
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import FractionProposalNetwork, FullQuantileFunction
from tianshou.utils.space_info import SpaceInfo
from tianshou.utils.torch_utils import policy_within_training_step


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="CartPole-v1")
    parser.add_argument("--reward-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eps-test", type=float, default=0.05)
    parser.add_argument("--eps-train", type=float, default=0.1)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--fraction-lr", type=float, default=2.5e-9)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--num-fractions", type=int, default=32)
    parser.add_argument("--num-cosines", type=int, default=64)
    parser.add_argument("--ent-coef", type=float, default=10.0)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=320)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--step-per-epoch", type=int, default=10000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[64, 64, 64])
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument("--prioritized-replay", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_known_args()[0]


def test_fqf(args: argparse.Namespace = get_args(), enable_assertions: bool = True) -> None:
    env = gym.make(args.task)
    space_info = SpaceInfo.from_env(env)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    args.state_shape = space_info.observation_info.obs_shape
    args.action_shape = space_info.action_info.action_shape
    if args.reward_threshold is None:
        default_reward_threshold = {"CartPole-v1": 195}
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
    feature_net = Net(
        state_shape=args.state_shape,
        action_shape=args.hidden_sizes[-1],
        hidden_sizes=args.hidden_sizes[:-1],
        softmax=False,
    )
    net = FullQuantileFunction(
        preprocess_net=feature_net,
        action_shape=args.action_shape,
        hidden_sizes=args.hidden_sizes,
        num_cosines=args.num_cosines,
    )
    optim = AdamOptimizerFactory(lr=args.lr)
    fraction_net = FractionProposalNetwork(args.num_fractions, net.input_dim)
    fraction_optim = RMSpropOptimizerFactory(lr=args.fraction_lr)
    policy = FQFPolicy(
        model=net,
        fraction_model=fraction_net,
        action_space=env.action_space,
        eps_training=args.eps_train,
        eps_inference=args.eps_test,
    )
    algorithm: FQF = FQF(
        policy=policy,
        optim=optim,
        fraction_optim=fraction_optim,
        gamma=args.gamma,
        num_fractions=args.num_fractions,
        ent_coef=args.ent_coef,
        estimation_step=args.n_step,
        target_update_freq=args.target_update_freq,
    ).to(args.device)

    # buffer
    buf: ReplayBuffer
    if args.prioritized_replay:
        buf = PrioritizedVectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs),
            alpha=args.alpha,
            beta=args.beta,
        )
    else:
        buf = VectorReplayBuffer(args.buffer_size, buffer_num=len(train_envs))

    # collectors
    train_collector = Collector[CollectStats](algorithm, train_envs, buf, exploration_noise=True)
    test_collector = Collector[CollectStats](algorithm, test_envs, exploration_noise=True)

    # initial data collection
    with policy_within_training_step(policy):
        train_collector.reset()
        train_collector.collect(n_step=args.batch_size * args.training_num)

    # logger
    log_path = os.path.join(args.logdir, args.task, "fqf")
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy: Algorithm) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards: float) -> bool:
        return mean_rewards >= args.reward_threshold

    def train_fn(epoch: int, env_step: int) -> None:
        # eps annnealing, just a demo
        if env_step <= 10000:
            policy.set_eps_training(args.eps_train)
        elif env_step <= 50000:
            eps = args.eps_train - (env_step - 10000) / 40000 * (0.9 * args.eps_train)
            policy.set_eps_training(eps)
        else:
            policy.set_eps_training(0.1 * args.eps_train)

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
            train_fn=train_fn,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            update_per_step=args.update_per_step,
            test_in_train=True,
        )
    )

    if enable_assertions:
        assert stop_fn(result.best_reward)


def test_pfqf(args: argparse.Namespace = get_args()) -> None:
    args.prioritized_replay = True
    args.gamma = 0.95
    test_fqf(args)


def test_fqf_determinism() -> None:
    main_fn = lambda args: test_fqf(args, enable_assertions=False)
    AlgorithmDeterminismTest("discrete_fqf", main_fn, get_args()).run()
