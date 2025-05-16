import argparse
import os
import warnings
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from pettingzoo.butterfly import pistonball_v6
from torch import nn
from torch.distributions import Distribution, Independent, Normal
from torch.utils.tensorboard import SummaryWriter

from tianshou.algorithm import PPO, Algorithm
from tianshou.algorithm.algorithm_base import OnPolicyAlgorithm
from tianshou.algorithm.modelfree.reinforce import ActorPolicyProbabilistic
from tianshou.algorithm.multiagent.marl import MultiAgentOnPolicyAlgorithm
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Collector, CollectStats, VectorReplayBuffer
from tianshou.data.stats import InfoStats
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.trainer import OnPolicyTrainerParams
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ModuleWithVectorOutput
from tianshou.utils.net.continuous import ContinuousActorProbabilistic, ContinuousCritic


class DQNet(ModuleWithVectorOutput):
    """Reference: Human-level control through deep reinforcement learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        c: int,
        h: int,
        w: int,
        device: str | int | torch.device = "cpu",
    ) -> None:
        net = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            output_dim = np.prod(net(torch.zeros(1, c, h, w)).shape[1:])
        super().__init__(int(output_dim))
        self.device = device
        self.c = c
        self.h = h
        self.w = w
        self.net = net

    def forward(
        self,
        x: np.ndarray | torch.Tensor,
        state: Any | None = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, Any]:
        r"""Mapping: x -> Q(x, \*)."""
        if info is None:
            info = {}
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        return self.net(x.reshape(-1, self.c, self.w, self.h)), state


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
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--epoch_num_steps", type=int, default=500)
    parser.add_argument("--collection_step_num_env_steps", type=int, default=10)
    parser.add_argument("--collection_step_num_episodes", type=int, default=16)
    parser.add_argument("--update_step_num_repetitions", type=int, default=2)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--num_train_envs", type=int, default=10)
    parser.add_argument("--num_test_envs", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")

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
    # ppo special
    parser.add_argument("--vf-coef", type=float, default=0.25)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--eps-clip", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--return_scaling", type=int, default=1)
    parser.add_argument("--dual-clip", type=float, default=None)
    parser.add_argument("--value-clip", type=int, default=1)
    parser.add_argument("--advantage_normalization", type=int, default=1)
    parser.add_argument("--recompute-adv", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-interval", type=int, default=4)
    parser.add_argument("--render", type=float, default=0.0)

    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]


def get_env(args: argparse.Namespace = get_args()) -> PettingZooEnv:
    return PettingZooEnv(pistonball_v6.env(continuous=True, n_pistons=args.n_pistons))


def get_agents(
    args: argparse.Namespace = get_args(),
    agents: list[OnPolicyAlgorithm] | None = None,
    optims: list[torch.optim.Optimizer] | None = None,
) -> tuple[Algorithm, list[torch.optim.Optimizer] | None, list]:
    env = get_env()
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gym.spaces.Dict)
        else env.observation_space
    )
    args.state_shape = observation_space.shape or observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]

    if agents is not None:
        algorithms = agents
    else:
        algorithms = []
        optims = []
        for _ in range(args.n_pistons):
            # model
            net = DQNet(
                observation_space.shape[2],
                observation_space.shape[1],
                observation_space.shape[0],
                device=args.device,
            ).to(args.device)

            actor = ContinuousActorProbabilistic(
                preprocess_net=net,
                action_shape=args.action_shape,
                max_action=args.max_action,
            ).to(args.device)
            net2 = DQNet(
                observation_space.shape[2],
                observation_space.shape[1],
                observation_space.shape[0],
                device=args.device,
            ).to(args.device)
            critic = ContinuousCritic(preprocess_net=net2).to(args.device)
            for m in set(actor.modules()).union(critic.modules()):
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.orthogonal_(m.weight)
                    torch.nn.init.zeros_(m.bias)
            optim = AdamOptimizerFactory(lr=args.lr)

            def dist(loc_scale: tuple[torch.Tensor, torch.Tensor]) -> Distribution:
                loc, scale = loc_scale
                return Independent(Normal(loc, scale), 1)

            policy = ActorPolicyProbabilistic(
                actor=actor,
                dist_fn=dist,
                action_space=env.action_space,
                action_scaling=True,
                action_bound_method="clip",
            )
            algorithm: PPO = PPO(
                policy=policy,
                critic=critic,
                optim=optim,
                gamma=args.gamma,
                max_grad_norm=args.max_grad_norm,
                eps_clip=args.eps_clip,
                vf_coef=args.vf_coef,
                ent_coef=args.ent_coef,
                return_scaling=args.return_scaling,
                advantage_normalization=args.advantage_normalization,
                recompute_advantage=args.recompute_adv,
                # dual_clip=args.dual_clip,
                # dual clip cause monotonically increasing log_std :)
                value_clip=args.value_clip,
                gae_lambda=args.gae_lambda,
            )

            algorithms.append(algorithm)
            optims.append(optim)

    ma_algorithm = MultiAgentOnPolicyAlgorithm(
        algorithms=algorithms,
        env=env,
    )
    return ma_algorithm, optims, env.agents


def train_agent(
    args: argparse.Namespace = get_args(),
    agents: list[OnPolicyAlgorithm] | None = None,
    optims: list[torch.optim.Optimizer] | None = None,
) -> tuple[InfoStats, Algorithm]:
    train_envs = DummyVectorEnv([get_env for _ in range(args.num_train_envs)])
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
        exploration_noise=False,  # True
    )
    test_collector = Collector[CollectStats](marl_algorithm, test_envs)
    # train_collector.collect(n_step=args.batch_size * args.num_train_envs, reset_before_collect=True)
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

    # train
    result = marl_algorithm.run_training(
        OnPolicyTrainerParams(
            train_collector=train_collector,
            test_collector=test_collector,
            max_epochs=args.epoch,
            epoch_num_steps=args.epoch_num_steps,
            update_step_num_repetitions=args.update_step_num_repetitions,
            test_step_num_episodes=args.test_num,
            batch_size=args.batch_size,
            collection_step_num_episodes=args.collection_step_num_episodes,
            collection_step_num_env_steps=None,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            resume_from_log=args.resume,
            test_in_train=True,
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
    collector = Collector[CollectStats](policy, env)
    collector_result = collector.collect(n_episode=1, render=args.render)
    collector_result.pprint_asdict()
