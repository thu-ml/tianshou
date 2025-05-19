import argparse
import os
import pprint

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.algorithm import SAC
from tianshou.algorithm.algorithm_base import Algorithm
from tianshou.algorithm.modelfree.sac import AutoAlpha, SACPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Collector, CollectStats, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.exploration import OUNoise
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ContinuousActorProbabilistic, ContinuousCritic
from tianshou.utils.space_info import SpaceInfo


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="MountainCarContinuous-v0")
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--buffer_size", type=int, default=50000)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--alpha_lr", type=float, default=3e-4)
    parser.add_argument("--noise_std", type=float, default=1.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--auto_alpha", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--epoch_num_steps", type=int, default=12000)
    parser.add_argument("--collection_step_num_env_steps", type=int, default=5)
    parser.add_argument("--update_per_step", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_sizes", type=int, nargs="*", default=[128, 128])
    parser.add_argument("--num_train_envs", type=int, default=5)
    parser.add_argument("--num_test_envs", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def test_sac(args: argparse.Namespace = get_args()) -> None:
    env = gym.make(args.task)
    space_info = SpaceInfo.from_env(env)
    args.state_shape = space_info.observation_info.obs_shape
    args.action_shape = space_info.action_info.action_shape
    args.max_action = space_info.action_info.max_action
    # train_envs = gym.make(args.task)
    train_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.num_train_envs)])
    # test_envs = gym.make(args.task)
    test_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.num_test_envs)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net = Net(state_shape=args.state_shape, hidden_sizes=args.hidden_sizes)
    actor = ContinuousActorProbabilistic(
        preprocess_net=net, action_shape=args.action_shape, unbounded=True
    ).to(args.device)
    actor_optim = AdamOptimizerFactory(lr=args.actor_lr)
    net_c1 = Net(
        state_shape=args.state_shape,
        action_shape=args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
    )
    critic1 = ContinuousCritic(preprocess_net=net_c1).to(args.device)
    critic1_optim = AdamOptimizerFactory(lr=args.critic_lr)
    net_c2 = Net(
        state_shape=args.state_shape,
        action_shape=args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
    )
    critic2 = ContinuousCritic(preprocess_net=net_c2).to(args.device)
    critic2_optim = AdamOptimizerFactory(lr=args.critic_lr)

    action_dim = space_info.action_info.action_dim
    if args.auto_alpha:
        target_entropy = -action_dim
        log_alpha = 0.0
        alpha_optim = AdamOptimizerFactory(lr=args.alpha_lr)
        args.alpha = AutoAlpha(target_entropy, log_alpha, alpha_optim).to(args.device)

    policy = SACPolicy(
        actor=actor,
        exploration_noise=OUNoise(0.0, args.noise_std),
        action_space=env.action_space,
    )
    algorithm: SAC = SAC(
        policy=policy,
        policy_optim=actor_optim,
        critic=critic1,
        critic_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
    )
    # collector
    train_collector = Collector[CollectStats](
        algorithm,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector[CollectStats](algorithm, test_envs)
    # train_collector.collect(n_step=args.buffer_size)
    # log
    log_path = os.path.join(args.logdir, args.task, "sac")
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
            stop_fn=stop_fn,
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
        collector_stats = test_collector.collect(n_episode=args.num_test_envs, render=args.render)
        print(collector_stats)


if __name__ == "__main__":
    test_sac()
