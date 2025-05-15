import argparse
import os
from copy import deepcopy
from functools import partial

import gymnasium
import numpy as np
import torch
from pettingzoo.classic import tictactoe_v3
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, CollectStats, VectorReplayBuffer
from tianshou.data.stats import InfoStats
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.algorithm import (
    DQN,
    Algorithm,
    MARLRandomDiscreteMaskedOffPolicyAlgorithm,
    MultiAgentOffPolicyAlgorithm,
)
from tianshou.algorithm.base import OffPolicyAlgorithm
from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory, OptimizerFactory
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net


def get_env(render_mode: str | None = None) -> PettingZooEnv:
    return PettingZooEnv(tictactoe_v3.env(render_mode=render_mode))


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--eps-test", type=float, default=0.05)
    parser.add_argument("--eps-train", type=float, default=0.1)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.9,
        help="a smaller gamma favors earlier win",
    )
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=320)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[128, 128, 128, 128])
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.1)
    parser.add_argument(
        "--win-rate",
        type=float,
        default=0.6,
        help="the expected winning rate: Optimal policy can get 0.7",
    )
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="no training, watch the play of pre-trained models",
    )
    parser.add_argument(
        "--agent-id",
        type=int,
        default=2,
        help="the learned agent plays as the agent_id-th player. Choices are 1 and 2.",
    )
    parser.add_argument(
        "--resume-path",
        type=str,
        default="",
        help="the path of agent pth file for resuming from a pre-trained agent",
    )
    parser.add_argument(
        "--opponent-path",
        type=str,
        default="",
        help="the path of opponent agent pth file for resuming from a pre-trained agent",
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


def get_agents(
    args: argparse.Namespace = get_args(),
    agent_learn: OffPolicyAlgorithm | None = None,
    agent_opponent: OffPolicyAlgorithm | None = None,
    optim: OptimizerFactory | None = None,
) -> tuple[MultiAgentOffPolicyAlgorithm, torch.optim.Optimizer | None, list]:
    env = get_env()
    observation_space = (
        env.observation_space.spaces["observation"]
        if isinstance(env.observation_space, gymnasium.spaces.Dict)
        else env.observation_space
    )
    args.state_shape = observation_space.shape or int(observation_space.n)
    args.action_shape = env.action_space.shape or int(env.action_space.n)
    if agent_learn is None:
        # model
        net = Net(
            state_shape=args.state_shape,
            action_shape=args.action_shape,
            hidden_sizes=args.hidden_sizes,
        ).to(args.device)
        if optim is None:
            optim = AdamOptimizerFactory(lr=args.lr)
        algorithm = DiscreteQLearningPolicy(
            model=net,
            action_space=env.action_space,
            eps_training=args.eps_train,
            eps_inference=args.eps_test,
        )
        agent_learn = DQN(
            policy=algorithm,
            optim=optim,
            estimation_step=args.n_step,
            gamma=args.gamma,
            target_update_freq=args.target_update_freq,
        )
        if args.resume_path:
            agent_learn.load_state_dict(torch.load(args.resume_path))

    if agent_opponent is None:
        if args.opponent_path:
            agent_opponent = deepcopy(agent_learn)
            agent_opponent.load_state_dict(torch.load(args.opponent_path))
        else:
            agent_opponent = MARLRandomDiscreteMaskedOffPolicyAlgorithm(
                action_space=env.action_space
            )

    if args.agent_id == 1:
        agents = [agent_learn, agent_opponent]
    else:
        agents = [agent_opponent, agent_learn]
    ma_algorithm = MultiAgentOffPolicyAlgorithm(algorithms=agents, env=env)
    return ma_algorithm, optim, env.agents


def train_agent(
    args: argparse.Namespace = get_args(),
    agent_learn: OffPolicyAlgorithm | None = None,
    agent_opponent: OffPolicyAlgorithm | None = None,
    optim: OptimizerFactory | None = None,
) -> tuple[InfoStats, OffPolicyAlgorithm]:
    train_envs = DummyVectorEnv([get_env for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([get_env for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    marl_algorithm, optim, agents = get_agents(
        args,
        agent_learn=agent_learn,
        agent_opponent=agent_opponent,
        optim=optim,
    )

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
    log_path = os.path.join(args.logdir, "tic_tac_toe", "dqn")
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    player_agent_id = agents[args.agent_id - 1]

    def save_best_fn(policy: Algorithm) -> None:
        if hasattr(args, "model_save_path"):
            model_save_path = args.model_save_path
        else:
            model_save_path = os.path.join(args.logdir, "tic_tac_toe", "dqn", "policy.pth")
        torch.save(policy.get_algorithm(player_agent_id).state_dict(), model_save_path)

    def stop_fn(mean_rewards: float) -> bool:
        return mean_rewards >= args.win_rate

    def reward_metric(rews: np.ndarray) -> np.ndarray:
        return rews[:, args.agent_id - 1]

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

    return result, marl_algorithm.get_algorithm(player_agent_id)


def watch(
    args: argparse.Namespace = get_args(),
    agent_learn: OffPolicyAlgorithm | None = None,
    agent_opponent: OffPolicyAlgorithm | None = None,
) -> None:
    env = DummyVectorEnv([partial(get_env, render_mode="human")])
    policy, optim, agents = get_agents(args, agent_learn=agent_learn, agent_opponent=agent_opponent)
    collector = Collector[CollectStats](policy, env, exploration_noise=True)
    result = collector.collect(n_episode=1, render=args.render, reset_before_collect=True)
    result.pprint_asdict()
