import argparse
import datetime
import os
import pickle
from test.determinism_test import AlgorithmDeterminismTest
from test.offline.gather_pendulum_data import expert_file_name, gather_data

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, CollectStats, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import BasePolicy, CQLPolicy
from tianshou.policy.imitation.cql import CQLTrainingStats
from tianshou.trainer import OfflineTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils.space_info import SpaceInfo


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Pendulum-v1")
    parser.add_argument("--reward-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", default=True, action="store_true")
    parser.add_argument("--alpha-lr", type=float, default=1e-3)
    parser.add_argument("--cql-alpha-lr", type=float, default=1e-3)
    parser.add_argument("--start-timesteps", type=int, default=10000)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--step-per-epoch", type=int, default=500)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cql-weight", type=float, default=1.0)
    parser.add_argument("--with-lagrange", type=bool, default=True)
    parser.add_argument("--lagrange-threshold", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--eval-freq", type=int, default=1)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=1 / 35)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    parser.add_argument("--load-buffer-name", type=str, default=expert_file_name())
    return parser.parse_known_args()[0]


def test_cql(args: argparse.Namespace = get_args(), enable_assertions: bool = True) -> None:
    if os.path.exists(args.load_buffer_name) and os.path.isfile(args.load_buffer_name):
        if args.load_buffer_name.endswith(".hdf5"):
            buffer = VectorReplayBuffer.load_hdf5(args.load_buffer_name)
        else:
            with open(args.load_buffer_name, "rb") as f:
                buffer = pickle.load(f)
    else:
        buffer = gather_data()
    env = gym.make(args.task)
    assert isinstance(env.action_space, gym.spaces.Box)

    space_info = SpaceInfo.from_env(env)

    args.state_shape = space_info.observation_info.obs_shape
    args.action_shape = space_info.action_info.action_shape
    args.min_action = space_info.action_info.min_action
    args.max_action = space_info.action_info.max_action
    args.state_dim = space_info.observation_info.obs_dim
    args.action_dim = space_info.action_info.action_dim

    if args.reward_threshold is None:
        # too low?
        default_reward_threshold = {"Pendulum-v0": -1200, "Pendulum-v1": -1200}
        args.reward_threshold = default_reward_threshold.get(
            args.task,
            env.spec.reward_threshold if env.spec else None,
        )

    # test_envs = gym.make(args.task)
    test_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    test_envs.seed(args.seed)

    # model
    # actor network
    net_a = Net(
        state_shape=args.state_shape,
        action_shape=args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
    )
    actor = ActorProb(
        net_a,
        action_shape=args.action_shape,
        device=args.device,
        unbounded=True,
        conditioned_sigma=True,
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    # critic network
    net_c = Net(
        state_shape=args.state_shape,
        action_shape=args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    critic = Critic(net_c, device=args.device).to(args.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = -np.prod(args.action_shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    policy: CQLPolicy[CQLTrainingStats] = CQLPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic=critic,
        critic_optim=critic_optim,
        # CQL seems to perform better without action scaling
        # TODO: investigate why
        action_scaling=False,
        action_space=env.action_space,
        cql_alpha_lr=args.cql_alpha_lr,
        cql_weight=args.cql_weight,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        temperature=args.temperature,
        with_lagrange=args.with_lagrange,
        lagrange_threshold=args.lagrange_threshold,
        min_action=args.min_action,
        max_action=args.max_action,
        device=args.device,
    )

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    # collector
    # buffer has been gathered
    # train_collector = Collector[CollectStats](policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector[CollectStats](policy, test_envs)
    # log
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'seed_{args.seed}_{t0}-{args.task.replace("-", "_")}_cql'
    log_path = os.path.join(args.logdir, args.task, "cql", log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    def save_best_fn(policy: BasePolicy) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards: float) -> bool:
        return mean_rewards >= args.reward_threshold

    # trainer
    trainer = OfflineTrainer(
        policy=policy,
        buffer=buffer,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        save_best_fn=save_best_fn,
        stop_fn=stop_fn,
        logger=logger,
    )
    stats = trainer.run()

    if enable_assertions:
        assert stop_fn(stats.best_reward)


def test_cql_determinism() -> None:
    main_fn = lambda args: test_cql(args, enable_assertions=False)
    AlgorithmDeterminismTest("offline_cql", main_fn, get_args(), is_offline=True).run()
