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
from tianshou.algorithm.modelfree.reinforce import ActorPolicyProbabilistic
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
from tianshou.utils.net.common import MLPActor
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
    parser.add_argument("--expert-data-task", type=str, default="halfcheetah-expert-v2")
    parser.add_argument("--buffer-size", type=int, default=4096)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--disc-lr", type=float, default=2.5e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=30000)
    parser.add_argument("--step-per-collect", type=int, default=2048)
    parser.add_argument("--repeat-per-collect", type=int, default=10)
    parser.add_argument("--disc-update-num", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--training-num", type=int, default=64)
    parser.add_argument("--test-num", type=int, default=10)
    # ppo special
    parser.add_argument("--rew-norm", type=int, default=True)
    # In theory, `vf-coef` will not make any difference if using Adam optimizer.
    parser.add_argument("--vf-coef", type=float, default=0.25)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--bound-action-method", type=str, default="clip")
    parser.add_argument("--lr-decay", type=int, default=True)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--eps-clip", type=float, default=0.2)
    parser.add_argument("--dual-clip", type=float, default=None)
    parser.add_argument("--value-clip", type=int, default=0)
    parser.add_argument("--norm-adv", type=int, default=0)
    parser.add_argument("--recompute-adv", type=int, default=1)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
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
        [lambda: NoRewardEnv(gym.make(args.task)) for _ in range(args.training_num)],
    )
    train_envs = VectorEnvNormObs(train_envs)
    # test_envs = gym.make(args.task)
    test_envs = SubprocVectorEnv([lambda: gym.make(args.task) for _ in range(args.test_num)])
    test_envs = VectorEnvNormObs(test_envs, update_obs_rms=False)
    test_envs.set_obs_rms(train_envs.get_obs_rms())

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net_a = MLPActor(
        state_shape=args.state_shape,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Tanh,
    )
    actor = ContinuousActorProbabilistic(
        preprocess_net=net_a,
        action_shape=args.action_shape,
        unbounded=True,
    ).to(args.device)
    net_c = MLPActor(
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
    net_d = MLPActor(
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
                epoch_num_steps=args.step_per_epoch,
                collection_step_num_env_steps=args.step_per_collect,
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

    policy = ActorPolicyProbabilistic(
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
        return_scaling=args.rew_norm,
        eps_clip=args.eps_clip,
        value_clip=args.value_clip,
        dual_clip=args.dual_clip,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
    )

    # load a previous policy
    if args.resume_path:
        algorithm.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    # collector
    buffer: ReplayBuffer
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(args.buffer_size)
    train_collector = Collector[CollectStats](algorithm, train_envs, buffer, exploration_noise=True)
    test_collector = Collector[CollectStats](algorithm, test_envs)
    # log
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'seed_{args.seed}_{t0}-{args.task.replace("-", "_")}_gail'
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
                epoch_num_steps=args.step_per_epoch,
                update_step_num_repetitions=args.repeat_per_collect,
                test_step_num_episodes=args.test_num,
                batch_size=args.batch_size,
                collection_step_num_env_steps=args.step_per_collect,
                save_best_fn=save_best_fn,
                logger=logger,
                test_in_train=False,
            )
        )
        pprint.pprint(result)

    # Let's watch its performance!
    test_envs.seed(args.seed)
    test_collector.reset()
    collector_stats = test_collector.collect(n_episode=args.test_num, render=args.render)
    print(collector_stats)


if __name__ == "__main__":
    test_gail()
