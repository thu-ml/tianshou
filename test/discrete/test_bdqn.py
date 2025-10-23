import argparse

import gymnasium as gym
import numpy as np
import torch

from test.determinism_test import AlgorithmDeterminismTest
from tianshou.algorithm import BDQN
from tianshou.algorithm.modelfree.bdqn import BDQNPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Collector, CollectStats, VectorReplayBuffer
from tianshou.env import ContinuousToDiscrete, DummyVectorEnv
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.utils.net.common import BranchingNet
from tianshou.utils.torch_utils import policy_within_training_step


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # task
    parser.add_argument("--task", type=str, default="Pendulum-v1")
    parser.add_argument("--reward_threshold", type=float, default=None)
    # network architecture
    parser.add_argument("--common_hidden_sizes", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--action_hidden_sizes", type=int, nargs="*", default=[64])
    parser.add_argument("--value_hidden_sizes", type=int, nargs="*", default=[64])
    parser.add_argument("--action_per_branch", type=int, default=40)
    # training hyperparameters
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--eps_test", type=float, default=0.01)
    parser.add_argument("--eps_train", type=float, default=0.76)
    parser.add_argument("--eps_decay", type=float, default=1e-4)
    parser.add_argument("--buffer_size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--target_update_freq", type=int, default=200)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--epoch_num_steps", type=int, default=80000)
    parser.add_argument("--collection_step_num_env_steps", type=int, default=10)
    parser.add_argument("--update_per_step", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_train_envs", type=int, default=10)
    parser.add_argument("--num_test_envs", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_known_args()[0]


def test_bdq(args: argparse.Namespace = get_args(), enable_assertions: bool = True) -> None:
    env = gym.make(args.task)
    env = ContinuousToDiscrete(env, args.action_per_branch)

    if isinstance(env.observation_space, gym.spaces.Box):
        args.state_shape = env.observation_space.shape
    elif isinstance(env.observation_space, gym.spaces.Discrete):
        args.state_shape = int(env.observation_space.n)
    assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
    args.num_branches = env.action_space.shape[0]

    if args.reward_threshold is None:
        default_reward_threshold = {"Pendulum-v0": -250, "Pendulum-v1": -250}
        args.reward_threshold = default_reward_threshold.get(
            args.task,
            env.spec.reward_threshold if env.spec else None,
        )

    print("Observations shape:", args.state_shape)
    print("Num branches:", args.num_branches)
    print("Actions per branch:", args.action_per_branch)

    train_envs = DummyVectorEnv(
        [
            lambda: ContinuousToDiscrete(gym.make(args.task), args.action_per_branch)
            for _ in range(args.num_train_envs)
        ],
    )
    test_envs = DummyVectorEnv(
        [
            lambda: ContinuousToDiscrete(gym.make(args.task), args.action_per_branch)
            for _ in range(args.num_test_envs)
        ],
    )

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net = BranchingNet(
        state_shape=args.state_shape,
        num_branches=args.num_branches,
        action_per_branch=args.action_per_branch,
        common_hidden_sizes=args.common_hidden_sizes,
        value_hidden_sizes=args.value_hidden_sizes,
        action_hidden_sizes=args.action_hidden_sizes,
    ).to(args.device)
    optim = AdamOptimizerFactory(lr=args.lr)
    policy = BDQNPolicy(
        model=net,
        action_space=env.action_space,  # type: ignore[arg-type]  # TODO: should `BranchingDQNPolicy` support also `MultiDiscrete` action spaces?
        eps_training=args.eps_train,
        eps_inference=args.eps_test,
    )
    algorithm: BDQN = BDQN(
        policy=policy,
        optim=optim,
        gamma=args.gamma,
        target_update_freq=args.target_update_freq,
    )
    # collector
    train_collector = Collector[CollectStats](
        algorithm,
        train_envs,
        VectorReplayBuffer(args.buffer_size, args.num_train_envs),
        exploration_noise=True,
    )
    test_collector = Collector[CollectStats](algorithm, test_envs, exploration_noise=False)

    # initial data collection
    with policy_within_training_step(policy):
        train_collector.reset()
        train_collector.collect(n_step=args.batch_size * args.num_train_envs)

    def train_fn(epoch: int, env_step: int) -> None:  # exp decay
        eps = max(args.eps_train * (1 - args.eps_decay) ** env_step, args.eps_test)
        policy.set_eps_training(eps)

    def stop_fn(mean_rewards: float) -> bool:
        return mean_rewards >= args.reward_threshold

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
            update_step_num_gradient_steps_per_sample=args.update_per_step,
            train_fn=train_fn,
            stop_fn=stop_fn,
            test_in_train=True,
        )
    )

    if enable_assertions:
        assert stop_fn(result.best_reward)


def test_bdq_determinism() -> None:
    main_fn = lambda args: test_bdq(args, enable_assertions=False)
    AlgorithmDeterminismTest("discrete_bdq", main_fn, get_args()).run()
