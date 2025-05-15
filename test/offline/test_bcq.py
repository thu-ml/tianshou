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
from tianshou.algorithm import BCQ, Algorithm
from tianshou.algorithm.imitation.bcq import BCQPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.trainer.base import OfflineTrainerParams
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import MLP, Net
from tianshou.utils.net.continuous import VAE, ContinuousCritic, Perturbation
from tianshou.utils.space_info import SpaceInfo


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Pendulum-v1")
    parser.add_argument("--reward-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[64])
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--step-per-epoch", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=1 / 35)

    parser.add_argument("--vae-hidden-sizes", type=int, nargs="*", default=[32, 32])
    # default to 2 * action_dim
    parser.add_argument("--latent_dim", type=int, default=None)
    parser.add_argument("--gamma", default=0.99)
    parser.add_argument("--tau", default=0.005)
    # Weighting for Clipped Double Q-learning in BCQ
    parser.add_argument("--lmbda", default=0.75)
    # Max perturbation hyper-parameter for BCQ
    parser.add_argument("--phi", default=0.05)
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
    parser.add_argument("--show-progress", action="store_true")
    return parser.parse_known_args()[0]


def test_bcq(args: argparse.Namespace = get_args(), enable_assertions: bool = True) -> None:
    if os.path.exists(args.load_buffer_name) and os.path.isfile(args.load_buffer_name):
        if args.load_buffer_name.endswith(".hdf5"):
            buffer = VectorReplayBuffer.load_hdf5(args.load_buffer_name)
        else:
            with open(args.load_buffer_name, "rb") as f:
                buffer = pickle.load(f)
    else:
        buffer = gather_data()
    env = gym.make(args.task)

    space_info = SpaceInfo.from_env(env)

    args.state_shape = space_info.observation_info.obs_shape
    args.action_shape = space_info.action_info.action_shape
    args.max_action = space_info.action_info.max_action
    args.state_dim = space_info.observation_info.obs_dim
    args.action_dim = space_info.action_info.action_dim

    if args.reward_threshold is None:
        # too low?
        default_reward_threshold = {"Pendulum-v0": -1100, "Pendulum-v1": -1100}
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

    # perturbation network
    net_a = MLP(
        input_dim=args.state_dim + args.action_dim,
        output_dim=args.action_dim,
        hidden_sizes=args.hidden_sizes,
    )
    actor = Perturbation(preprocess_net=net_a, max_action=args.max_action, phi=args.phi).to(
        args.device,
    )
    actor_optim = AdamOptimizerFactory(lr=args.actor_lr)

    net_c = Net(
        state_shape=args.state_shape,
        action_shape=args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
    )
    critic = ContinuousCritic(preprocess_net=net_c).to(args.device)
    critic_optim = AdamOptimizerFactory(lr=args.critic_lr)

    # vae
    # output_dim = 0, so the last Module in the encoder is ReLU
    vae_encoder = MLP(
        input_dim=args.state_dim + args.action_dim,
        hidden_sizes=args.vae_hidden_sizes,
    )
    if not args.latent_dim:
        args.latent_dim = args.action_dim * 2
    vae_decoder = MLP(
        input_dim=args.state_dim + args.latent_dim,
        output_dim=args.action_dim,
        hidden_sizes=args.vae_hidden_sizes,
    )
    vae = VAE(
        encoder=vae_encoder,
        decoder=vae_decoder,
        hidden_dim=args.vae_hidden_sizes[-1],
        latent_dim=args.latent_dim,
        max_action=args.max_action,
    ).to(args.device)
    vae_optim = AdamOptimizerFactory()

    policy = BCQPolicy(
        actor_perturbation=actor,
        critic=critic,
        vae=vae,
        action_space=env.action_space,
    )
    algorithm = BCQ(
        policy=policy,
        actor_perturbation_optim=actor_optim,
        critic_optim=critic_optim,
        vae_optim=vae_optim,
        gamma=args.gamma,
        tau=args.tau,
        lmbda=args.lmbda,
    ).to(args.device)

    # load a previous policy
    if args.resume_path:
        algorithm.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    # collector
    # buffer has been gathered
    # train_collector = Collector[CollectStats](policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector[CollectStats](algorithm, test_envs)

    # logger
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'seed_{args.seed}_{t0}-{args.task.replace("-", "_")}_bcq'
    log_path = os.path.join(args.logdir, args.task, "bcq", log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    def save_best_fn(policy: Algorithm) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards: float) -> bool:
        return mean_rewards >= args.reward_threshold

    def watch() -> None:
        algorithm.load_state_dict(
            torch.load(os.path.join(log_path, "policy.pth"), map_location=torch.device("cpu")),
        )
        collector = Collector[CollectStats](algorithm, env)
        collector.collect(n_episode=1, render=1 / 35)

    # train
    result = algorithm.run_training(
        OfflineTrainerParams(
            buffer=buffer,
            test_collector=test_collector,
            max_epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            episode_per_test=args.test_num,
            batch_size=args.batch_size,
            save_best_fn=save_best_fn,
            stop_fn=stop_fn,
            logger=logger,
            show_progress=args.show_progress,
        )
    )

    if enable_assertions:
        assert stop_fn(result.best_reward)


def test_bcq_determinism() -> None:
    main_fn = lambda args: test_bcq(args, enable_assertions=False)
    AlgorithmDeterminismTest("offline_bcq", main_fn, get_args(), is_offline=True).run()
