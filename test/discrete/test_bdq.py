import argparse
import pprint

import gymnasium as gym
import numpy as np
import torch

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import ContinuousToDiscrete, DummyVectorEnv
from tianshou.policy import BranchingDQNPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import BranchingNet


def get_args():
    parser = argparse.ArgumentParser()
    # task
    parser.add_argument("--task", type=str, default="Pendulum-v1")
    parser.add_argument("--reward-threshold", type=float, default=None)
    # network architecture
    parser.add_argument("--common-hidden-sizes", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--action-hidden-sizes", type=int, nargs="*", default=[64])
    parser.add_argument("--value-hidden-sizes", type=int, nargs="*", default=[64])
    parser.add_argument("--action-per-branch", type=int, default=40)
    # training hyperparameters
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--eps-test", type=float, default=0.01)
    parser.add_argument("--eps-train", type=float, default=0.76)
    parser.add_argument("--eps-decay", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--target-update-freq", type=int, default=200)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--step-per-epoch", type=int, default=80000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_known_args()[0]


def test_bdq(args=get_args()):
    env = gym.make(args.task)
    env = ContinuousToDiscrete(env, args.action_per_branch)

    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.num_branches = env.action_space.shape[0]

    if args.reward_threshold is None:
        default_reward_threshold = {"Pendulum-v0": -250, "Pendulum-v1": -250}
        args.reward_threshold = default_reward_threshold.get(args.task, env.spec.reward_threshold)

    print("Observations shape:", args.state_shape)
    print("Num branches:", args.num_branches)
    print("Actions per branch:", args.action_per_branch)

    train_envs = DummyVectorEnv(
        [
            lambda: ContinuousToDiscrete(gym.make(args.task), args.action_per_branch)
            for _ in range(args.training_num)
        ],
    )
    test_envs = DummyVectorEnv(
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
    policy = BranchingDQNPolicy(net, optim, args.gamma, target_update_freq=args.target_update_freq)
    # collector
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, args.training_num),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=False)
    # policy.set_eps(1)
    train_collector.collect(n_step=args.batch_size * args.training_num)

    def train_fn(epoch, env_step):  # exp decay
        eps = max(args.eps_train * (1 - args.eps_decay) ** env_step, args.eps_test)
        policy.set_eps(eps)

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    def stop_fn(mean_rewards):
        return mean_rewards >= args.reward_threshold

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
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
    ).run()

    # assert stop_fn(result["best_reward"])
    if __name__ == "__main__":
        pprint.pprint(result)
        # Let's watch its performance!
        policy.eval()
        policy.set_eps(args.eps_test)
        test_envs.seed(args.seed)
        test_collector.reset()
        collector_result = test_collector.collect(n_episode=args.test_num, render=args.render)
        rews, lens = collector_result["rews"], collector_result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


if __name__ == "__main__":
    test_bdq(get_args())
