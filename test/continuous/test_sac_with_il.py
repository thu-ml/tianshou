import argparse
import os

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from test.determinism_test import AlgorithmDeterminismTest
from tianshou.algorithm import SAC, OffPolicyImitationLearning
from tianshou.algorithm.algorithm_base import Algorithm
from tianshou.algorithm.imitation.imitation_base import ImitationPolicy
from tianshou.algorithm.modelfree.sac import AutoAlpha, SACPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Collector, CollectStats, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import (
    ContinuousActorDeterministic,
    ContinuousActorProbabilistic,
    ContinuousCritic,
)
from tianshou.utils.space_info import SpaceInfo

try:
    import envpool
except ImportError:
    envpool = None


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Pendulum-v1")
    parser.add_argument("--reward_threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--buffer_size", type=int, default=20000)
    parser.add_argument("--actor_lr", type=float, default=1e-3)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--il_lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto_alpha", type=int, default=1)
    parser.add_argument("--alpha_lr", type=float, default=3e-4)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--epoch_num_steps", type=int, default=24000)
    parser.add_argument("--il_step_per_epoch", type=int, default=500)
    parser.add_argument("--collection_step_num_env_steps", type=int, default=10)
    parser.add_argument("--update_per_step", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_sizes", type=int, nargs="*", default=[128, 128])
    parser.add_argument("--imitation_hidden_sizes", type=int, nargs="*", default=[128, 128])
    parser.add_argument("--num_training_envs", type=int, default=10)
    parser.add_argument("--num_test_envs", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument("--n_step", type=int, default=3)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_known_args()[0]


def test_sac_with_il(
    args: argparse.Namespace = get_args(),
    enable_assertions: bool = True,
    skip_il: bool = False,
) -> None:
    # if you want to use python vector env, please refer to other test scripts
    # training_envs = env = envpool.make_gymnasium(args.task, num_envs=args.num_training_envs, seed=args.seed)
    # test_envs = envpool.make_gymnasium(args.task, num_envs=args.num_test_envs, seed=args.seed)
    env = gym.make(args.task)
    training_envs = DummyVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.num_training_envs)]
    )
    test_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.num_test_envs)])
    space_info = SpaceInfo.from_env(env)
    args.state_shape = space_info.observation_info.obs_shape
    args.action_shape = space_info.action_info.action_shape
    args.max_action = space_info.action_info.max_action
    if args.reward_threshold is None:
        default_reward_threshold = {"Pendulum-v0": -250, "Pendulum-v1": -250}
        args.reward_threshold = default_reward_threshold.get(
            args.task,
            env.spec.reward_threshold if env.spec else None,
        )

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    training_envs.seed(args.seed)
    test_envs.seed(args.seed + args.num_training_envs)

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
        args.alpha = AutoAlpha(target_entropy, log_alpha, alpha_optim)
    policy = SACPolicy(
        actor=actor,
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
        n_step_return_horizon=args.n_step,
    )
    # collector
    train_collector = Collector[CollectStats](
        algorithm,
        training_envs,
        VectorReplayBuffer(args.buffer_size, len(training_envs)),
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
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            test_in_train=True,
        )
    )

    if enable_assertions:
        assert stop_fn(result.best_reward)

    if skip_il:
        return

    # here we define an imitation collector with a trivial policy
    if args.task.startswith("Pendulum"):
        args.reward_threshold -= 50  # lower the goal
    il_net = Net(
        state_shape=args.state_shape,
        hidden_sizes=args.imitation_hidden_sizes,
    )
    il_actor = ContinuousActorDeterministic(
        preprocess_net=il_net,
        action_shape=args.action_shape,
        max_action=args.max_action,
    ).to(args.device)
    optim = AdamOptimizerFactory(lr=args.il_lr)
    il_policy = ImitationPolicy(
        actor=il_actor,
        action_space=env.action_space,
        action_scaling=True,
        action_bound_method="clip",
    )
    il_algorithm: OffPolicyImitationLearning = OffPolicyImitationLearning(
        policy=il_policy,
        optim=optim,
    )
    il_test_env = gym.make(args.task)
    il_test_env.reset(seed=args.seed + args.num_training_envs + args.num_test_envs)
    il_test_collector = Collector[CollectStats](
        il_algorithm,
        # envpool.make_gymnasium(args.task, num_envs=args.num_test_envs, seed=args.seed),
        il_test_env,
    )
    train_collector.reset()
    result = il_algorithm.run_training(
        OffPolicyTrainerParams(
            train_collector=train_collector,
            test_collector=il_test_collector,
            max_epochs=args.epoch,
            epoch_num_steps=args.epoch_num_steps,
            collection_step_num_env_steps=args.collection_step_num_env_steps,
            test_step_num_episodes=args.num_test_envs,
            batch_size=args.batch_size,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            test_in_train=True,
        )
    )

    if enable_assertions:
        assert stop_fn(result.best_reward)


def test_sac_determinism() -> None:
    main_fn = lambda args: test_sac_with_il(args, enable_assertions=False, skip_il=True)
    AlgorithmDeterminismTest("continuous_sac", main_fn, get_args()).run()
