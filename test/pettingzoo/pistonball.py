import argparse
import os
import warnings

import gymnasium as gym
import numpy as np
import torch
from pettingzoo.butterfly import pistonball_v6
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, CollectStats, InfoStats, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import DQN, Algorithm, MultiAgentOffPolicyAlgorithm
from tianshou.policy.base import OffPolicyAlgorithm
from tianshou.policy.modelfree.dqn import DiscreteQLearningPolicy
from tianshou.policy.optim import AdamOptimizerFactory
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import MLPActor


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--eps-test", type=float, default=0.05)
    parser.add_argument("--eps-train", type=float, default=0.1)
    parser.add_argument("--buffer-size", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.9,
        help="a smaller gamma favors earlier win",
    )
    parser.add_argument(
        "--n-pistons",
        type=int,
        default=3,
        help="Number of pistons(agents) in the env",
    )
    parser.add_argument("--n-step", type=int, default=100)
    parser.add_argument("--target-update-freq", type=int, default=320)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--step-per-epoch", type=int, default=500)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)

    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="no training, watch the play of pre-trained models",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]


def get_env(args: argparse.Namespace = get_args()) -> PettingZooEnv:
    return PettingZooEnv(pistonball_v6.env(continuous=False, n_pistons=args.n_pistons))


def get_agents(
    args: argparse.Namespace = get_args(),
    agents: list[OffPolicyAlgorithm] | None = None,
    optims: list[torch.optim.Optimizer] | None = None,
) -> tuple[Algorithm, list[torch.optim.Optimizer] | None, list]:
    env = get_env()
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gym.spaces.Dict)
        else env.observation_space
    )
    args.state_shape = observation_space.shape or int(observation_space.n)
    args.action_shape = env.action_space.shape or int(env.action_space.n)

    if agents is not None:
        algorithms = agents
    else:
        algorithms = []
        optims = []
        for _ in range(args.n_pistons):
            # model
            net = MLPActor(
                state_shape=args.state_shape,
                action_shape=args.action_shape,
                hidden_sizes=args.hidden_sizes,
            ).to(args.device)
            optim = AdamOptimizerFactory(lr=args.lr)
            policy = DiscreteQLearningPolicy(
                model=net,
                action_space=env.action_space,
                eps_training=args.eps_train,
                eps_inference=args.eps_test,
            )
            agent: DQN = DQN(
                policy=policy,
                optim=optim,
                gamma=args.gamma,
                estimation_step=args.n_step,
                target_update_freq=args.target_update_freq,
            )
            algorithms.append(agent)
            optims.append(optim)

    ma_algorithm = MultiAgentOffPolicyAlgorithm(algorithms=algorithms, env=env)
    return ma_algorithm, optims, env.agents


def train_agent(
    args: argparse.Namespace = get_args(),
    agents: list[OffPolicyAlgorithm] | None = None,
    optims: list[torch.optim.Optimizer] | None = None,
) -> tuple[InfoStats, Algorithm]:
    train_envs = DummyVectorEnv([get_env for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([get_env for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    marl_algorithm, optim, agents = get_agents(args, agents=agents, optims=optims)

    # collector
    train_collector = Collector[CollectStats](
        marl_algorithm,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector[CollectStats](marl_algorithm, test_envs, exploration_noise=True)
    train_collector.reset()
    train_collector.collect(n_step=args.batch_size * args.training_num)
    # log
    log_path = os.path.join(args.logdir, "pistonball", "dqn")
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    def save_best_fn(policy: Algorithm) -> None:
        pass

    def stop_fn(mean_rewards: float) -> bool:
        return False

    def reward_metric(rews: np.ndarray) -> np.ndarray:
        return rews[:, 0]

    # trainer
    result = marl_algorithm.run_training(
        OffPolicyTrainerParams(
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            step_per_collect=args.step_per_collect,
            episode_per_test=args.test_num,
            batch_size=args.batch_size,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            update_per_step=args.update_per_step,
            logger=logger,
            test_in_train=False,
            reward_metric=reward_metric,
        )
    )
    return result, marl_algorithm


def watch(args: argparse.Namespace = get_args(), policy: Algorithm | None = None) -> None:
    env = DummyVectorEnv([get_env])
    if not policy:
        warnings.warn(
            "watching random agents, as loading pre-trained policies is currently not supported",
        )
        policy, _, _ = get_agents(args)
    collector = Collector[CollectStats](policy, env, exploration_noise=True)
    result = collector.collect(n_episode=1, render=args.render)
    result.pprint_asdict()
