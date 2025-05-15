import argparse
import os
import pprint

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, CollectStats, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.algorithm import DQN
from tianshou.algorithm.base import Algorithm
from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.space_info import SpaceInfo


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # the parameters are found by Optuna
    parser.add_argument("--task", type=str, default="LunarLander-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eps-test", type=float, default=0.01)
    parser.add_argument("--eps-train", type=float, default=0.73)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.013)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=4)
    parser.add_argument("--target-update-freq", type=int, default=500)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--step-per-epoch", type=int, default=80000)
    parser.add_argument("--step-per-collect", type=int, default=16)
    parser.add_argument("--update-per-step", type=float, default=0.0625)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[128, 128])
    parser.add_argument("--dueling-q-hidden-sizes", type=int, nargs="*", default=[128, 128])
    parser.add_argument("--dueling-v-hidden-sizes", type=int, nargs="*", default=[128, 128])
    parser.add_argument("--training-num", type=int, default=16)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def test_dqn(args: argparse.Namespace = get_args()) -> None:
    env = gym.make(args.task)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    space_info = SpaceInfo.from_env(env)
    args.state_shape = space_info.observation_info.obs_shape
    args.action_shape = space_info.action_info.action_shape
    args.max_action = space_info.action_info.max_action
    # train_envs = gym.make(args.task)
    # you can also use tianshou.env.SubprocVectorEnv
    train_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.training_num)])
    # test_envs = gym.make(args.task)
    test_envs = SubprocVectorEnv([lambda: gym.make(args.task) for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    Q_param = {"hidden_sizes": args.dueling_q_hidden_sizes}
    V_param = {"hidden_sizes": args.dueling_v_hidden_sizes}
    net = Net(
        state_shape=args.state_shape,
        action_shape=args.action_shape,
        hidden_sizes=args.hidden_sizes,
        dueling_param=(Q_param, V_param),
    ).to(args.device)
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
        estimation_step=args.n_step,
        target_update_freq=args.target_update_freq,
    )
    # collector
    train_collector = Collector[CollectStats](
        algorithm,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector[CollectStats](algorithm, test_envs, exploration_noise=True)
    train_collector.reset()
    train_collector.collect(n_step=args.batch_size * args.training_num)
    # log
    log_path = os.path.join(args.logdir, args.task, "dqn")
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy: Algorithm) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards: float) -> bool:
        if env.spec:
            if not env.spec.reward_threshold:
                return False
            else:
                return mean_rewards >= env.spec.reward_threshold
        return False

    def train_fn(epoch: int, env_step: int) -> None:  # exp decay
        eps = max(args.eps_train * (1 - 5e-6) ** env_step, args.eps_test)
        policy.set_eps_training(eps)

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
            update_per_step=args.update_per_step,
            stop_fn=stop_fn,
            train_fn=train_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            test_in_train=True,
        )
    )

    assert stop_fn(result.best_reward)
    if __name__ == "__main__":
        pprint.pprint(result)
        # Let's watch its performance!
        test_envs.seed(args.seed)
        test_collector.reset()
        collector_stats = test_collector.collect(n_episode=args.test_num, render=args.render)
        print(collector_stats)


if __name__ == "__main__":
    test_dqn(get_args())
