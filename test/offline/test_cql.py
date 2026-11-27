import argparse
import datetime
import math
import os
import pickle

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch.utils.tensorboard import SummaryWriter

from test.determinism_test import AlgorithmDeterminismTest
from test.offline.gather_pendulum_data import expert_file_name, gather_data
from tianshou.algorithm import CQL, Algorithm
from tianshou.algorithm.imitation.cql import _conservative_loss, _sample_uniform_actions
from tianshou.algorithm.modelfree.sac import AutoAlpha, SACPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Collector, CollectStats, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OfflineTrainerParams
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ContinuousActorProbabilistic, ContinuousCritic
from tianshou.utils.space_info import SpaceInfo


@pytest.mark.parametrize(("low", "high", "dim"), [(-1.0, 1.0, 3), (-3.0, 1.0, 1), (-0.4, 0.4, 17)])
def test_cql_uniform_actions(low: float, high: float, dim: int) -> None:
    act = torch.empty((8, dim), dtype=torch.float64)
    samples, log_prob = _sample_uniform_actions(act, 10, low, high)
    assert samples.shape == (80, dim)
    assert samples.dtype == act.dtype
    assert (samples >= low).all() and (samples <= high).all()
    assert (samples.amax(dim=0) > samples.amin(dim=0)).all()
    assert log_prob == pytest.approx(-math.log((high - low) ** dim))


@pytest.mark.parametrize("calibrated", [False, True])
def test_cql_penalty_and_gradients_per_state(calibrated: bool) -> None:
    # Two draws per state, with one column for each proposal.
    weights = torch.tensor([[1.0, 1, 1], [9, 9, 9], [4, 8, 16], [1, 2, 4]])
    scores = weights.log().requires_grad_()
    data_q = torch.tensor([1.0, 3.0], requires_grad=True)
    returns = torch.tensor([2.0, 3.0]).log() if calibrated else None
    loss = _conservative_loss(data_q, scores.split(1, dim=1), 1.0, returns)
    score_grad, data_grad = torch.autograd.grad(loss, (scores, data_q))

    if calibrated:
        # Values below the return contribute to the partition but receive zero gradient.
        totals = torch.tensor([33.0, 38.0])
        weights[0] = 0
        weights[3, :2] = 0
    else:
        totals = torch.tensor([30.0, 35.0])
    torch.testing.assert_close(loss, totals.log().mean() - 2)
    torch.testing.assert_close(data_grad, torch.tensor([-0.5, -0.5]))
    torch.testing.assert_close(score_grad[:2], weights[:2] / (2 * totals[0]))
    torch.testing.assert_close(score_grad[2:], weights[2:] / (2 * totals[1]))


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Pendulum-v1")
    parser.add_argument("--reward_threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hidden_sizes", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--actor_lr", type=float, default=1e-3)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto_alpha", default=True, action="store_true")
    parser.add_argument("--alpha_lr", type=float, default=1e-3)
    parser.add_argument("--cql_alpha_lr", type=float, default=1e-3)
    parser.add_argument("--start_timesteps", type=int, default=10000)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--epoch_num_steps", type=int, default=500)
    parser.add_argument("--n_step", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cql_weight", type=float, default=5.0)
    parser.add_argument("--with_lagrange", type=bool, default=True)
    parser.add_argument("--lagrange_threshold", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--num_test_envs", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=1 / 35)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    parser.add_argument("--load_buffer_name", type=str, default=expert_file_name())
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
    test_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.num_test_envs)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    test_envs.seed(args.seed)

    net_a = Net(state_shape=args.state_shape, hidden_sizes=args.hidden_sizes)
    actor = ContinuousActorProbabilistic(
        preprocess_net=net_a,
        action_shape=args.action_shape,
        unbounded=True,
        conditioned_sigma=True,
    ).to(args.device)
    actor_optim = AdamOptimizerFactory(lr=args.actor_lr)

    # critic network
    net_c = Net(
        state_shape=args.state_shape,
        action_shape=args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
    )
    critic = ContinuousCritic(preprocess_net=net_c).to(args.device)
    critic_optim = AdamOptimizerFactory(lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = float(-np.prod(args.action_shape))
        log_alpha = 0.0
        alpha_optim = AdamOptimizerFactory(lr=args.alpha_lr)
        args.alpha = AutoAlpha(target_entropy, log_alpha, alpha_optim)

    # The collector stores SAC actions before scaling them for Pendulum.
    policy = SACPolicy(actor=actor, action_space=env.action_space)
    algorithm = CQL(
        policy=policy,
        policy_optim=actor_optim,
        critic=critic,
        critic_optim=critic_optim,
        cql_alpha_lr=args.cql_alpha_lr,
        cql_weight=args.cql_weight,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        temperature=args.temperature,
        with_lagrange=args.with_lagrange,
        lagrange_threshold=args.lagrange_threshold,
    )

    # load a previous policy
    if args.resume_path:
        algorithm.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    # collector
    # buffer has been gathered
    # training_collector = Collector[CollectStats](policy, training_envs, buffer, exploration_noise=True)
    test_collector = Collector[CollectStats](algorithm, test_envs)
    # log
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f"seed_{args.seed}_{t0}-{args.task.replace('-', '_')}_cql"
    log_path = os.path.join(args.logdir, args.task, "cql", log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    def save_best_fn(policy: Algorithm) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards: float) -> bool:
        return mean_rewards >= args.reward_threshold

    # trainer
    result = algorithm.run_training(
        OfflineTrainerParams(
            buffer=buffer,
            test_collector=test_collector,
            max_epochs=args.epoch,
            epoch_num_steps=args.epoch_num_steps,
            test_step_num_episodes=args.num_test_envs,
            batch_size=args.batch_size,
            save_best_fn=save_best_fn,
            stop_fn=stop_fn,
            logger=logger,
        )
    )

    if enable_assertions:
        assert stop_fn(result.best_reward)


def test_cql_determinism() -> None:
    main_fn = lambda args: test_cql(args, enable_assertions=False)
    AlgorithmDeterminismTest("offline_cql", main_fn, get_args(), is_offline=True).run()
