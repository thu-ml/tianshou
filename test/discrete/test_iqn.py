import argparse
import os

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from test.determinism_test import AlgorithmDeterminismTest
from tianshou.algorithm import IQN
from tianshou.algorithm.algorithm_base import Algorithm
from tianshou.algorithm.modelfree.iqn import IQNPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
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
from tianshou.utils.net.discrete import ImplicitQuantileNetwork
from tianshou.utils.space_info import SpaceInfo
from tianshou.utils.torch_utils import policy_within_training_step


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="CartPole-v1")
    parser.add_argument("--reward_threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eps_test", type=float, default=0.05)
    parser.add_argument("--eps_train", type=float, default=0.1)
    parser.add_argument("--buffer_size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--sample_size", type=int, default=32)
    parser.add_argument("--online_sample_size", type=int, default=8)
    parser.add_argument("--target_sample_size", type=int, default=8)
    parser.add_argument("--num_cosines", type=int, default=64)
    parser.add_argument("--n_step", type=int, default=3)
    parser.add_argument("--target_update_freq", type=int, default=320)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--epoch_num_steps", type=int, default=10000)
    parser.add_argument("--collection_step_num_env_steps", type=int, default=10)
    parser.add_argument("--update_per_step", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_sizes", type=int, nargs="*", default=[64, 64, 64])
    parser.add_argument("--num_training_envs", type=int, default=10)
    parser.add_argument("--num_test_envs", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument("--prioritized_replay", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_known_args()[0]


def test_iqn(args: argparse.Namespace = get_args(), enable_assertions: bool = True) -> None:
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
    training_envs = DummyVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.num_training_envs)]
    )
    test_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.num_test_envs)])

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    training_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # model
    feature_net = Net(
        state_shape=args.state_shape,
        action_shape=args.hidden_sizes[-1],
        hidden_sizes=args.hidden_sizes[:-1],
        softmax=False,
    )
    net = ImplicitQuantileNetwork(
        preprocess_net=feature_net,
        action_shape=args.action_shape,
        num_cosines=args.num_cosines,
    )
    optim = AdamOptimizerFactory(lr=args.lr)
    policy = IQNPolicy(
        model=net,
        action_space=env.action_space,
        sample_size=args.sample_size,
        online_sample_size=args.online_sample_size,
        target_sample_size=args.target_sample_size,
        eps_training=args.eps_train,
        eps_inference=args.eps_test,
    )
    algorithm: IQN = IQN(
        policy=policy,
        optim=optim,
        gamma=args.gamma,
        n_step_return_horizon=args.n_step,
        target_update_freq=args.target_update_freq,
    ).to(args.device)

    # buffer
    buf: ReplayBuffer
    if args.prioritized_replay:
        buf = PrioritizedVectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(training_envs),
            alpha=args.alpha,
            beta=args.beta,
        )
    else:
        buf = VectorReplayBuffer(args.buffer_size, buffer_num=len(training_envs))

    # collectors
    training_collector = Collector[CollectStats](
        algorithm, training_envs, buf, exploration_noise=True
    )
    test_collector = Collector[CollectStats](algorithm, test_envs, exploration_noise=True)

    # initial data collection
    with policy_within_training_step(policy):
        training_collector.reset()
        training_collector.collect(n_step=args.batch_size * args.num_training_envs)

    # logger
    log_path = os.path.join(args.logdir, args.task, "iqn")
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
            training_collector=training_collector,
            test_collector=test_collector,
            max_epochs=args.epoch,
            epoch_num_steps=args.epoch_num_steps,
            collection_step_num_env_steps=args.collection_step_num_env_steps,
            test_step_num_episodes=args.num_test_envs,
            batch_size=args.batch_size,
            training_fn=train_fn,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            update_step_num_gradient_steps_per_sample=args.update_per_step,
            test_in_training=True,
        )
    )

    if enable_assertions:
        assert stop_fn(result.best_reward)


def test_piqn(args: argparse.Namespace = get_args()) -> None:
    args.prioritized_replay = True
    args.gamma = 0.95
    test_iqn(args)


def test_iqn_determinism() -> None:
    main_fn = lambda args: test_iqn(args, enable_assertions=False)
    AlgorithmDeterminismTest("discrete_iqn", main_fn, get_args()).run()
