import argparse
import os
from test.determinism_test import AlgorithmDeterminismTest

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.algorithm import DQN
from tianshou.algorithm.algorithm_base import Algorithm
from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Collector, CollectStats, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Recurrent
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
    parser.add_argument("--stack-num", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=320)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--epoch_num_steps", type=int, default=20000)
    parser.add_argument("--update-per-step", type=float, default=1 / 16)
    parser.add_argument("--collection_step_num_env_steps", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--layer-num", type=int, default=2)
    parser.add_argument("--num_train_envs", type=int, default=16)
    parser.add_argument("--num_test_envs", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_known_args()[0]


def test_drqn(args: argparse.Namespace = get_args(), enable_assertions: bool = True) -> None:
    env = gym.make(args.task)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    space_info = SpaceInfo.from_env(env)
    args.state_shape = space_info.observation_info.obs_shape
    args.action_shape = space_info.action_info.action_shape
    if args.reward_threshold is None:
        default_reward_threshold = {"CartPole-v1": 195}
        args.reward_threshold = default_reward_threshold.get(
            args.task,
            env.spec.reward_threshold if env.spec else None,
        )
    # train_envs = gym.make(args.task)
    # you can also use tianshou.env.SubprocVectorEnv
    train_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.num_train_envs)])
    # test_envs = gym.make(args.task)
    test_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net = Recurrent(
        layer_num=args.layer_num, state_shape=args.state_shape, action_shape=args.action_shape
    ).to(
        args.device,
    )
    optim = AdamOptimizerFactory(lr=args.lr)
    policy = DiscreteQLearningPolicy(
        model=net,
        action_space=env.action_space,
        eps_training=args.eps_train,
        eps_inference=args.eps_test,
    )
    algorithm: DQN = DQN(
        policy=policy,
        optim=optim,
        gamma=args.gamma,
        n_step_return_horizon=args.n_step,
        target_update_freq=args.target_update_freq,
    )

    # collector
    buffer = VectorReplayBuffer(
        args.buffer_size,
        buffer_num=len(train_envs),
        stack_num=args.stack_num,
        ignore_obs_next=True,
    )
    train_collector = Collector[CollectStats](algorithm, train_envs, buffer, exploration_noise=True)
    # the stack_num is for RNN training: sample framestack obs
    test_collector = Collector[CollectStats](algorithm, test_envs, exploration_noise=True)

    # initial data collection
    with policy_within_training_step(policy):
        train_collector.reset()
        train_collector.collect(n_step=args.batch_size * args.num_train_envs)

    # log
    log_path = os.path.join(args.logdir, args.task, "drqn")
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
            max_epochs=args.epoch,
            epoch_num_steps=args.epoch_num_steps,
            collection_step_num_env_steps=args.collection_step_num_env_steps,
            test_step_num_episodes=args.test_num,
            batch_size=args.batch_size,
            update_step_num_gradient_steps_per_sample=args.update_per_step,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            test_in_train=True,
        )
    )

    if enable_assertions:
        assert stop_fn(result.best_reward)


def test_drqn_determinism() -> None:
    main_fn = lambda args: test_drqn(args, enable_assertions=False)
    AlgorithmDeterminismTest("discrete_drqn", main_fn, get_args()).run()
