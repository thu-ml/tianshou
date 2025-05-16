import argparse
import os
from test.determinism_test import AlgorithmDeterminismTest

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Distribution, Independent, Normal
from torch.utils.tensorboard import SummaryWriter

from tianshou.algorithm import NPG
from tianshou.algorithm.algorithm_base import Algorithm
from tianshou.algorithm.modelfree.reinforce import ActorPolicyProbabilistic
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Collector, CollectStats, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OnPolicyTrainerParams
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import MLPActor
from tianshou.utils.net.continuous import ContinuousActorProbabilistic, ContinuousCritic
from tianshou.utils.space_info import SpaceInfo


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Pendulum-v1")
    parser.add_argument("--reward-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--epoch_num_steps", type=int, default=50000)
    parser.add_argument("--collection_step_num_env_steps", type=int, default=2048)
    parser.add_argument(
        "--update_step_num_repetitions", type=int, default=2
    )  # theoretically it should be 1
    parser.add_argument("--batch_size", type=int, default=99999)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--num_train_envs", type=int, default=16)
    parser.add_argument("--num_test_envs", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    # npg special
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--return_scaling", type=int, default=1)
    parser.add_argument("--advantage_normalization", type=int, default=1)
    parser.add_argument("--optim-critic-iters", type=int, default=5)
    parser.add_argument("--trust_region_size", type=float, default=0.5)
    return parser.parse_known_args()[0]


def test_npg(args: argparse.Namespace = get_args(), enable_assertions: bool = True) -> None:
    env = gym.make(args.task)

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
    train_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.num_train_envs)])
    test_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.test_num)])

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # model
    net = MLPActor(
        state_shape=args.state_shape,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Tanh,
    )
    actor = ContinuousActorProbabilistic(
        preprocess_net=net, action_shape=args.action_shape, unbounded=True
    ).to(args.device)
    critic = ContinuousCritic(
        preprocess_net=MLPActor(
            state_shape=args.state_shape,
            hidden_sizes=args.hidden_sizes,
            activation=nn.Tanh,
        ),
    ).to(args.device)

    # orthogonal initialization
    for m in list(actor.modules()) + list(critic.modules()):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    # replace DiagGuassian with Independent(Normal) which is equivalent
    # pass *logits to be consistent with policy.forward
    def dist(loc_scale: tuple[torch.Tensor, torch.Tensor]) -> Distribution:
        loc, scale = loc_scale
        return Independent(Normal(loc, scale), 1)

    policy = ActorPolicyProbabilistic(
        actor=actor,
        dist_fn=dist,
        action_space=env.action_space,
        deterministic_eval=True,
    )
    algorithm: NPG = NPG(
        policy=policy,
        critic=critic,
        optim=AdamOptimizerFactory(lr=args.lr),
        gamma=args.gamma,
        return_scaling=args.return_scaling,
        advantage_normalization=args.advantage_normalization,
        gae_lambda=args.gae_lambda,
        optim_critic_iters=args.optim_critic_iters,
        trust_region_size=args.trust_region_size,
    )
    # collector
    train_collector = Collector[CollectStats](
        algorithm,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
    )
    test_collector = Collector[CollectStats](algorithm, test_envs)
    # log
    log_path = os.path.join(args.logdir, args.task, "npg")
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy: Algorithm) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards: float) -> bool:
        return mean_rewards >= args.reward_threshold

    # trainer
    result = algorithm.run_training(
        OnPolicyTrainerParams(
            train_collector=train_collector,
            test_collector=test_collector,
            max_epochs=args.epoch,
            epoch_num_steps=args.epoch_num_steps,
            update_step_num_repetitions=args.update_step_num_repetitions,
            test_step_num_episodes=args.test_num,
            batch_size=args.batch_size,
            collection_step_num_env_steps=args.collection_step_num_env_steps,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            test_in_train=True,
        )
    )

    if enable_assertions:
        assert stop_fn(result.best_reward)


def test_npg_determinism() -> None:
    main_fn = lambda args: test_npg(args, enable_assertions=False)
    AlgorithmDeterminismTest("continuous_npg", main_fn, get_args()).run()
