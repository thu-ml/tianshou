#!/usr/bin/env python3

import argparse
import datetime
import os
import pprint
from typing import SupportsFloat, cast

import d4rl
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Distribution, Independent, Normal
from torch.utils.tensorboard import SummaryWriter

from tianshou.algorithm import GAIL
from tianshou.algorithm.algorithm_base import Algorithm
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory, LRSchedulerFactoryLinear
from tianshou.data import (
    Batch,
    Collector,
    CollectStats,
    ReplayBuffer,
    VectorReplayBuffer,
)
from tianshou.data.types import RolloutBatchProtocol
from tianshou.env import SubprocVectorEnv, VectorEnvNormObs
from tianshou.trainer import OnPolicyTrainerParams
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ContinuousActorProbabilistic, ContinuousCritic
from tianshou.utils.space_info import SpaceInfo


class NoRewardEnv(gym.RewardWrapper):
    """sets the reward to 0.

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def reward(self, reward: SupportsFloat) -> np.ndarray:
        """Set reward to 0."""
        return np.zeros_like(reward)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="HalfCheetah-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expert_data_task", type=str, default="halfcheetah-expert-v2")
    parser.add_argument("--buffer_size", type=int, default=4096)
    parser.add_argument("--hidden_sizes", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--disc_lr", type=float, default=2.5e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--epoch_num_steps", type=int, default=30000)
    parser.add_argument("--collection_step_num_env_steps", type=int, default=2048)
    parser.add_argument("--update_step_num_repetitions", type=int, default=10)
    parser.add_argument("--disc_update_num", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_train_envs", type=int, default=64)
    parser.add_argument("--num_test_envs", type=int, default=10)
    # ppo special
    parser.add_argument("--return_scaling", type=int, default=True)
    # In theory, `vf-coef` will not make any difference if using Adam optimizer.
    parser.add_argument("--vf_coef", type=float, default=0.25)
    parser.add_argument("--ent_coef", type=float, default=0.001)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--bound_action_method", type=str, default="clip")
    parser.add_argument("--lr_decay", type=int, default=True)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--eps_clip", type=float, default=0.2)
    parser.add_argument("--dual_clip", type=float, default=None)
    parser.add_argument("--value_clip", type=int, default=0)
    parser.add_argument("--advantage_normalization", type=int, default=0)
    parser.add_argument("--recompute_adv", type=int, default=1)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
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
    return parser.parse_args()


def test_gail(args: argparse.Namespace = get_args()) -> None:
    env = gym.make(args.task)
    space_info = SpaceInfo.from_env(env)
    args.state_shape = space_info.observation_info.obs_shape
    args.action_shape = space_info.action_info.action_shape
    args.max_action = space_info.action_info.max_action
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", args.min_action, args.max_action)
    # train_envs = gym.make(args.task)
    train_envs = SubprocVectorEnv(
        [lambda: NoRewardEnv(gym.make(args.task)) for _ in range(args.num_train_envs)],
    )
    train_envs = VectorEnvNormObs(train_envs)
    # test_envs = gym.make(args.task)
    test_envs = SubprocVectorEnv([lambda: gym.make(args.task) for _ in range(args.num_test_envs)])
    test_envs = VectorEnvNormObs(test_envs, update_obs_rms=False)
    test_envs.set_obs_rms(train_envs.get_obs_rms())

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net_a = Net(
        state_shape=args.state_shape,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Tanh,
    )
    actor = ContinuousActorProbabilistic(
        preprocess_net=net_a,
        action_shape=args.action_shape,
        unbounded=True,
    ).to(args.device)
    net_c = Net(
        state_shape=args.state_shape,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Tanh,
    )
    critic = ContinuousCritic(preprocess_net=net_c).to(args.device)
    torch.nn.init.constant_(actor.sigma_param, -0.5)
    for m in list(actor.modules()) + list(critic.modules()):
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
    # do last policy layer scaling, this will make initial actions have (close to)
    # 0 mean and std, and will help boost performances,
    # see https://arxiv.org/abs/2006.05990, Fig.24 for details
    for m in actor.mu.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)

    optim = AdamOptimizerFactory(lr=args.lr)
    # discriminator
    net_d = Net(
        state_shape=args.state_shape,
        action_shape=args.action_shape,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Tanh,
        concat=True,
    )
    disc_net = ContinuousCritic(preprocess_net=net_d).to(args.device)
    for m in disc_net.modules():
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
    disc_optim = AdamOptimizerFactory(lr=args.disc_lr)

    if args.lr_decay:
        optim.with_lr_scheduler_factory(
            LRSchedulerFactoryLinear(
                max_epochs=args.epoch,
                epoch_num_steps=args.epoch_num_steps,
                collection_step_num_env_steps=args.collection_step_num_env_steps,
            )
        )

    def dist(loc_scale: tuple[torch.Tensor, torch.Tensor]) -> Distribution:
        loc, scale = loc_scale
        return Independent(Normal(loc, scale), 1)

    # expert replay buffer
    dataset = d4rl.qlearning_dataset(gym.make(args.expert_data_task))
    dataset_size = dataset["rewards"].size

    print("dataset_size", dataset_size)
    expert_buffer = ReplayBuffer(dataset_size)

    for i in range(dataset_size):
        expert_buffer.add(
            cast(
                RolloutBatchProtocol,
                Batch(
                    obs=dataset["observations"][i],
                    act=dataset["actions"][i],
                    rew=dataset["rewards"][i],
                    done=dataset["terminals"][i],
                    obs_next=dataset["next_observations"][i],
                ),
            ),
        )
    print("dataset loaded")

    policy = ProbabilisticActorPolicy(
        actor=actor,
        dist_fn=dist,
        action_scaling=True,
        action_bound_method=args.bound_action_method,
        action_space=env.action_space,
    )
    algorithm: GAIL = GAIL(
        policy=policy,
        critic=critic,
        optim=optim,
        expert_buffer=expert_buffer,
        disc_net=disc_net,
        disc_optim=disc_optim,
        disc_update_num=args.disc_update_num,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        return_scaling=args.return_scaling,
        eps_clip=args.eps_clip,
        value_clip=args.value_clip,
        dual_clip=args.dual_clip,
        advantage_normalization=args.advantage_normalization,
        recompute_advantage=args.recompute_adv,
    )

    # load a previous policy
    if args.resume_path:
        algorithm.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    # collector
    buffer: ReplayBuffer
    if args.num_train_envs > 1:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(args.buffer_size)
    train_collector = Collector[CollectStats](algorithm, train_envs, buffer, exploration_noise=True)
    test_collector = Collector[CollectStats](algorithm, test_envs)
    # log
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f"seed_{args.seed}_{t0}-{args.task.replace('-', '_')}_gail"
    log_path = os.path.join(args.logdir, args.task, "gail", log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer, update_interval=100, train_interval=100)

    def save_best_fn(policy: Algorithm) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    if not args.watch:
        # train
        result = algorithm.run_training(
            OnPolicyTrainerParams(
                train_collector=train_collector,
                test_collector=test_collector,
                max_epochs=args.epoch,
                epoch_num_steps=args.epoch_num_steps,
                update_step_num_repetitions=args.update_step_num_repetitions,
                test_step_num_episodes=args.num_test_envs,
                batch_size=args.batch_size,
                collection_step_num_env_steps=args.collection_step_num_env_steps,
                save_best_fn=save_best_fn,
                logger=logger,
                test_in_train=False,
            )
        )
        pprint.pprint(result)

    # Let's watch its performance!
    test_envs.seed(args.seed)
    test_collector.reset()
    collector_stats = test_collector.collect(n_episode=args.num_test_envs, render=args.render)
    print(collector_stats)


if __name__ == "__main__":
    test_gail()
