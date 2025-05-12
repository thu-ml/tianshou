import argparse
import os
import pickle
from test.determinism_test import AlgorithmDeterminismTest
from test.offline.gather_cartpole_data import expert_file_name, gather_data

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
from tianshou.policy import BasePolicy, DiscreteCQLPolicy
from tianshou.trainer import OfflineTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.space_info import SpaceInfo


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="CartPole-v1")
    parser.add_argument("--reward-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--eps-test", type=float, default=0.001)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--num-quantiles", type=int, default=200)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=500)
    parser.add_argument("--min-q-weight", type=float, default=10.0)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[64])
    parser.add_argument("--test-num", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument("--load-buffer-name", type=str, default=expert_file_name())
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_known_args()[0]


def test_discrete_cql(
    args: argparse.Namespace = get_args(),
    enable_assertions: bool = True,
) -> None:
    # envs
    env = gym.make(args.task)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    space_info = SpaceInfo.from_env(env)
    args.state_shape = space_info.observation_info.obs_shape
    args.action_shape = space_info.action_info.action_shape
    if args.reward_threshold is None:
        default_reward_threshold = {"CartPole-v1": 170}
        args.reward_threshold = default_reward_threshold.get(
            args.task,
            env.spec.reward_threshold if env.spec else None,
        )
    test_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net = Net(
        state_shape=args.state_shape,
        action_shape=args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
        softmax=False,
        num_atoms=args.num_quantiles,
    )
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    policy: DiscreteCQLPolicy = DiscreteCQLPolicy(
        model=net,
        optim=optim,
        action_space=env.action_space,
        discount_factor=args.gamma,
        num_quantiles=args.num_quantiles,
        estimation_step=args.n_step,
        target_update_freq=args.target_update_freq,
        min_q_weight=args.min_q_weight,
    ).to(args.device)
    # buffer
    buffer: VectorReplayBuffer | PrioritizedVectorReplayBuffer
    if os.path.exists(args.load_buffer_name) and os.path.isfile(args.load_buffer_name):
        if args.load_buffer_name.endswith(".hdf5"):
            buffer = VectorReplayBuffer.load_hdf5(args.load_buffer_name)
        else:
            with open(args.load_buffer_name, "rb") as f:
                buffer = pickle.load(f)
    else:
        buffer = gather_data()

    # collector
    test_collector = Collector[CollectStats](policy, test_envs, exploration_noise=True)

    log_path = os.path.join(args.logdir, args.task, "discrete_cql")
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy: BasePolicy) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards: float) -> bool:
        return mean_rewards >= args.reward_threshold

    result = OfflineTrainer(
        policy=policy,
        buffer=buffer,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
    ).run()

    if enable_assertions:
        assert stop_fn(result.best_reward)


def test_discrete_cql_determinism() -> None:
    main_fn = lambda args: test_discrete_cql(args, enable_assertions=False)
    AlgorithmDeterminismTest("offline_discrete_cql", main_fn, get_args(), is_offline=True).run()
