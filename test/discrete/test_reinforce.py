import argparse
import os
from test.determinism_test import AlgorithmDeterminismTest

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from torch.utils.tensorboard import SummaryWriter

from tianshou.algorithm import Reinforce
from tianshou.algorithm.algorithm_base import Algorithm
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Collector, CollectStats, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OnPolicyTrainerParams
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.space_info import SpaceInfo


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="CartPole-v1")
    parser.add_argument("--reward_threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--buffer_size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--epoch_num_steps", type=int, default=40000)
    parser.add_argument("--collection_step_num_episodes", type=int, default=8)
    parser.add_argument("--update_step_num_repetitions", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_sizes", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--num_train_envs", type=int, default=8)
    parser.add_argument("--num_test_envs", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument("--return_scaling", type=int, default=1)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_known_args()[0]


def test_reinforce(args: argparse.Namespace = get_args(), enable_assertions: bool = True) -> None:
    env = gym.make(args.task)
    space_info = SpaceInfo.from_env(env)
    args.state_shape = space_info.observation_info.obs_shape
    args.action_shape = space_info.action_info.action_shape
    if args.reward_threshold is None:
        default_reward_threshold = {"CartPole-v1": 195}
        args.reward_threshold = default_reward_threshold.get(
            args.task,
            env.spec.reward_threshold if env.spec else None,
        )
    train_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.num_train_envs)])
    test_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.num_test_envs)])

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # model
    net = Net(
        state_shape=args.state_shape,
        action_shape=args.action_shape,
        hidden_sizes=args.hidden_sizes,
        softmax=True,
    ).to(args.device)
    optim = AdamOptimizerFactory(lr=args.lr)
    dist_fn = torch.distributions.Categorical
    policy = ProbabilisticActorPolicy(
        actor=net,
        dist_fn=dist_fn,
        action_space=env.action_space,
        action_scaling=isinstance(env.action_space, Box),
    )
    algorithm: Reinforce = Reinforce(
        policy=policy,
        optim=optim,
        gamma=args.gamma,
        return_standardization=args.return_scaling,
    )
    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)

    # collector
    train_collector = Collector[CollectStats](
        algorithm,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
    )
    test_collector = Collector[CollectStats](algorithm, test_envs)

    # log
    log_path = os.path.join(args.logdir, args.task, "pg")
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(algorithm: Algorithm) -> None:
        torch.save(algorithm.policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards: float) -> bool:
        return mean_rewards >= args.reward_threshold

    # train
    training_config = OnPolicyTrainerParams(
        train_collector=train_collector,
        test_collector=test_collector,
        max_epochs=args.epoch,
        epoch_num_steps=args.epoch_num_steps,
        update_step_num_repetitions=args.update_step_num_repetitions,
        test_step_num_episodes=args.num_test_envs,
        batch_size=args.batch_size,
        collection_step_num_episodes=args.collection_step_num_episodes,
        collection_step_num_env_steps=None,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        test_in_train=True,
    )
    result = algorithm.run_training(training_config)

    if enable_assertions:
        assert stop_fn(result.best_reward)


def test_reinforce_determinism() -> None:
    main_fn = lambda args: test_reinforce(args, enable_assertions=False)
    AlgorithmDeterminismTest("discrete_reinforce", main_fn, get_args()).run()
