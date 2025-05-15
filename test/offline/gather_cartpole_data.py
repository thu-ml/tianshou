import argparse
import os
import pickle

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import (
    Collector,
    CollectStats,
    PrioritizedVectorReplayBuffer,
    VectorReplayBuffer,
)
from tianshou.env import DummyVectorEnv
from tianshou.algorithm import QRDQN
from tianshou.algorithm.algorithm_base import Algorithm
from tianshou.algorithm.modelfree.qrdqn import QRDQNPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.space_info import SpaceInfo


def expert_file_name() -> str:
    return os.path.join(os.path.dirname(__file__), "expert_QRDQN_CartPole-v1.pkl")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="CartPole-v1")
    parser.add_argument("--reward-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eps-test", type=float, default=0.05)
    parser.add_argument("--eps-train", type=float, default=0.1)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--num-quantiles", type=int, default=200)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=320)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--step-per-epoch", type=int, default=10000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[128, 128, 128, 128])
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument("--prioritized-replay", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--save-buffer-name", type=str, default=expert_file_name())
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_known_args()[0]


def gather_data() -> VectorReplayBuffer | PrioritizedVectorReplayBuffer:
    args = get_args()
    env = gym.make(args.task)
    assert isinstance(env.action_space, gym.spaces.Discrete)

    space_info = SpaceInfo.from_env(env)
    args.state_shape = space_info.observation_info.obs_shape
    args.action_shape = space_info.action_info.action_shape

    if args.reward_threshold is None:
        default_reward_threshold = {"CartPole-v1": 190}
        args.reward_threshold = default_reward_threshold.get(
            args.task,
            env.spec.reward_threshold if env.spec else None,
        )
    # train_envs = gym.make(args.task)
    # you can also use tianshou.env.SubprocVectorEnv
    train_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.training_num)])
    # test_envs = gym.make(args.task)
    test_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.test_num)])
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
        softmax=False,
        num_atoms=args.num_quantiles,
    )
    optim = AdamOptimizerFactory(lr=args.lr)
    policy = QRDQNPolicy(
        model=net,
        action_space=env.action_space,
        eps_training=args.eps_train,
        eps_inference=args.eps_test,
    )
    algorithm: QRDQN = QRDQN(
        policy=policy,
        optim=optim,
        gamma=args.gamma,
        num_quantiles=args.num_quantiles,
        estimation_step=args.n_step,
        target_update_freq=args.target_update_freq,
    ).to(args.device)
    # buffer
    buf: VectorReplayBuffer | PrioritizedVectorReplayBuffer
    if args.prioritized_replay:
        buf = PrioritizedVectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs),
            alpha=args.alpha,
            beta=args.beta,
        )
    else:
        buf = VectorReplayBuffer(args.buffer_size, buffer_num=len(train_envs))
    # collector
    train_collector = Collector[CollectStats](algorithm, train_envs, buf, exploration_noise=True)
    train_collector.reset()
    test_collector = Collector[CollectStats](algorithm, test_envs, exploration_noise=True)
    test_collector.reset()
    train_collector.collect(n_step=args.batch_size * args.training_num)
    # log
    log_path = os.path.join(args.logdir, args.task, "qrdqn")
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
    assert stop_fn(result.best_reward)

    # save buffer in pickle format, for imitation learning unittest
    buf = VectorReplayBuffer(args.buffer_size, buffer_num=len(test_envs))
    policy.set_eps_inference(0.2)
    collector = Collector[CollectStats](algorithm, test_envs, buf, exploration_noise=True)
    collector.reset()
    collector_stats = collector.collect(n_step=args.buffer_size)
    if args.save_buffer_name.endswith(".hdf5"):
        buf.save_hdf5(args.save_buffer_name)
    else:
        with open(args.save_buffer_name, "wb") as f:
            pickle.dump(buf, f)
    print(collector_stats)
    return buf
