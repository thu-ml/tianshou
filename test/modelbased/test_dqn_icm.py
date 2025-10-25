import argparse
import os

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.algorithm import DQN, Algorithm, ICMOffPolicyWrapper
from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import (
    Collector,
    CollectStats,
    PrioritizedVectorReplayBuffer,
    VectorReplayBuffer,
)
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import MLP, Net
from tianshou.utils.net.discrete import IntrinsicCuriosityModule
from tianshou.utils.space_info import SpaceInfo


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="CartPole-v1")
    parser.add_argument("--reward_threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--eps_test", type=float, default=0.05)
    parser.add_argument("--eps_train", type=float, default=0.1)
    parser.add_argument("--buffer_size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--n_step", type=int, default=3)
    parser.add_argument("--target_update_freq", type=int, default=320)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--epoch_num_steps", type=int, default=10000)
    parser.add_argument("--collection_step_num_env_steps", type=int, default=10)
    parser.add_argument("--update_per_step", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_sizes", type=int, nargs="*", default=[128, 128, 128, 128])
    parser.add_argument("--num_training_envs", type=int, default=10)
    parser.add_argument("--num_test_envs", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument("--prioritized_replay", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--lr_scale",
        type=float,
        default=1.0,
        help="use intrinsic curiosity module with this lr scale",
    )
    parser.add_argument(
        "--reward_scale",
        type=float,
        default=0.01,
        help="scaling factor for intrinsic curiosity reward",
    )
    parser.add_argument(
        "--forward_loss_weight",
        type=float,
        default=0.2,
        help="weight for the forward model loss in ICM",
    )
    return parser.parse_known_args()[0]


def test_dqn_icm(args: argparse.Namespace = get_args()) -> None:
    env = gym.make(args.task)
    assert isinstance(env.action_space, gym.spaces.Discrete)

    space_info = SpaceInfo.from_env(env)
    args.state_shape = space_info.observation_info.obs_shape
    args.action_shape = space_info.action_info.action_shape

    if args.reward_threshold is None:
        default_reward_threshold = {"CartPole-v1": 195}
        args.reward_threshold = default_reward_threshold.get(
            args.task,
            env.spec.reward_threshold if env.spec else None,
        )
    training_envs = DummyVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.num_training_envs)]
    )
    test_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.num_test_envs)])

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    training_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # Q_param = V_param = {"hidden_sizes": [128]}
    # model
    net = Net(
        state_shape=args.state_shape,
        action_shape=args.action_shape,
        hidden_sizes=args.hidden_sizes,
        # dueling=(Q_param, V_param),
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
        n_step_return_horizon=args.n_step,
        target_update_freq=args.target_update_freq,
    )

    # ICM wrapper
    feature_dim = args.hidden_sizes[-1]
    obs_dim = space_info.observation_info.obs_dim
    feature_net = MLP(
        input_dim=obs_dim,
        output_dim=feature_dim,
        hidden_sizes=args.hidden_sizes[:-1],
    )
    action_dim = space_info.action_info.action_dim
    icm_net = IntrinsicCuriosityModule(
        feature_net=feature_net,
        feature_dim=feature_dim,
        action_dim=action_dim,
        hidden_sizes=args.hidden_sizes[-1:],
    ).to(args.device)
    icm_optim = AdamOptimizerFactory(lr=args.lr)
    icm_algorithm = ICMOffPolicyWrapper(
        wrapped_algorithm=algorithm,
        model=icm_net,
        optim=icm_optim,
        lr_scale=args.lr_scale,
        reward_scale=args.reward_scale,
        forward_loss_weight=args.forward_loss_weight,
    )

    # buffer
    buf: PrioritizedVectorReplayBuffer | VectorReplayBuffer
    if args.prioritized_replay:
        buf = PrioritizedVectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(training_envs),
            alpha=args.alpha,
            beta=args.beta,
        )
    else:
        buf = VectorReplayBuffer(args.buffer_size, buffer_num=len(training_envs))

    # collector
    train_collector = Collector[CollectStats](
        icm_algorithm, training_envs, buf, exploration_noise=True
    )
    test_collector = Collector[CollectStats](icm_algorithm, test_envs, exploration_noise=True)
    train_collector.reset()
    train_collector.collect(n_step=args.batch_size * args.num_training_envs)

    # log
    log_path = str(os.path.join(args.logdir, args.task, "dqn_icm"))
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy: Algorithm) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards: float) -> bool:
        return mean_rewards >= args.reward_threshold

    def train_fn(epoch: int, env_step: int) -> None:
        # eps annnealing, just a demo
        if env_step <= 10000:
            policy.set_eps_training(args.eps_train)
        elif env_step <= 50000:
            eps = args.eps_train - (env_step - 10000) / 40000 * (0.9 * args.eps_train)
            policy.set_eps_training(eps)
        else:
            policy.set_eps_training(0.1 * args.eps_train)

    # train
    result = icm_algorithm.run_training(
        OffPolicyTrainerParams(
            train_collector=train_collector,
            test_collector=test_collector,
            max_epochs=args.epoch,
            epoch_num_steps=args.epoch_num_steps,
            collection_step_num_env_steps=args.collection_step_num_env_steps,
            test_step_num_episodes=args.num_test_envs,
            batch_size=args.batch_size,
            update_step_num_gradient_steps_per_sample=args.update_per_step,
            train_fn=train_fn,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            test_in_train=True,
        )
    )
    assert stop_fn(result.best_reward)
