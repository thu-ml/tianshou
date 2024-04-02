import argparse
import datetime
import os
import pprint

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import ContinuousToDiscrete, SubprocVectorEnv
from tianshou.policy import BranchingDQNPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import BranchingNet


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # task
    parser.add_argument("--task", type=str, default="BipedalWalker-v3")
    # network architecture
    parser.add_argument("--common-hidden-sizes", type=int, nargs="*", default=[512, 256])
    parser.add_argument("--action-hidden-sizes", type=int, nargs="*", default=[128])
    parser.add_argument("--value-hidden-sizes", type=int, nargs="*", default=[128])
    parser.add_argument("--action-per-branch", type=int, default=25)
    # training hyperparameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eps-test", type=float, default=0.0)
    parser.add_argument("--eps-train", type=float, default=0.73)
    parser.add_argument("--eps-decay", type=float, default=5e-6)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-update-freq", type=int, default=1000)
    parser.add_argument("--epoch", type=int, default=25)
    parser.add_argument("--step-per-epoch", type=int, default=80000)
    parser.add_argument("--step-per-collect", type=int, default=16)
    parser.add_argument("--update-per-step", type=float, default=0.0625)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--training-num", type=int, default=20)
    parser.add_argument("--test-num", type=int, default=10)
    # other
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def test_bdq(args: argparse.Namespace = get_args()) -> None:
    env = gym.make(args.task)
    env = ContinuousToDiscrete(env, args.action_per_branch)

    assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
    assert isinstance(
        env.observation_space,
        gym.spaces.Box,
    )  # BipedalWalker-v3 has `Box` observation space by design
    args.state_shape = env.observation_space.shape
    args.action_shape = env.action_space.shape
    args.num_branches = args.action_shape[0]

    print("Observations shape:", args.state_shape)
    print("Num branches:", args.num_branches)
    print("Actions per branch:", args.action_per_branch)

    # train_envs = ContinuousToDiscrete(gym.make(args.task), args.action_per_branch)
    # you can also use tianshou.env.SubprocVectorEnv
    train_envs = SubprocVectorEnv(
        [
            lambda: ContinuousToDiscrete(gym.make(args.task), args.action_per_branch)
            for _ in range(args.training_num)
        ],
    )
    # test_envs = ContinuousToDiscrete(gym.make(args.task), args.action_per_branch)
    test_envs = SubprocVectorEnv(
        [
            lambda: ContinuousToDiscrete(gym.make(args.task), args.action_per_branch)
            for _ in range(args.test_num)
        ],
    )
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net = BranchingNet(
        args.state_shape,
        args.num_branches,
        args.action_per_branch,
        args.common_hidden_sizes,
        args.value_hidden_sizes,
        args.action_hidden_sizes,
        device=args.device,
    ).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    policy: BranchingDQNPolicy = BranchingDQNPolicy(
        model=net,
        optim=optim,
        discount_factor=args.gamma,
        action_space=env.action_space,  # type: ignore[arg-type]  # TODO: should `BranchingDQNPolicy` support also `MultiDiscrete` action spaces?
        target_update_freq=args.target_update_freq,
    )
    # collector
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=False)
    # policy.set_eps(1)
    train_collector.collect(n_step=args.batch_size * args.training_num)
    # log
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(args.logdir, "bdq", args.task, current_time)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy: BasePolicy) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards: float) -> bool:
        if env.spec and env.spec.reward_threshold:
            return mean_rewards >= env.spec.reward_threshold
        return False

    def train_fn(epoch: int, env_step: int) -> None:  # exp decay
        eps = max(args.eps_train * (1 - args.eps_decay) ** env_step, args.eps_test)
        policy.set_eps(eps)

    def test_fn(epoch: int, env_step: int | None) -> None:
        policy.set_eps(args.eps_test)

    # trainer
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        update_per_step=args.update_per_step,
        stop_fn=stop_fn,
        train_fn=train_fn,
        test_fn=test_fn,
        save_best_fn=save_best_fn,
        logger=logger,
    ).run()

    assert stop_fn(result.best_reward)
    if __name__ == "__main__":
        pprint.pprint(result)
        # Let's watch its performance!
        policy.eval()
        policy.set_eps(args.eps_test)
        test_envs.seed(args.seed)
        test_collector.reset()
        collector_stats = test_collector.collect(n_episode=args.test_num, render=args.render)
        print(collector_stats)


if __name__ == "__main__":
    test_bdq(get_args())
