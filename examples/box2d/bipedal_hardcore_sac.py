import argparse
import os
import pprint
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.core import WrapperActType, WrapperObsType
from torch.utils.tensorboard import SummaryWriter

from tianshou.algorithm import SAC
from tianshou.algorithm.algorithm_base import Algorithm
from tianshou.algorithm.modelfree.sac import AutoAlpha, SACPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Collector, CollectStats, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ContinuousActorProbabilistic, ContinuousCritic
from tianshou.utils.space_info import SpaceInfo


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="BipedalWalkerHardcore-v3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer_size", type=int, default=1000000)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--auto_alpha", type=int, default=1)
    parser.add_argument("--alpha_lr", type=float, default=3e-4)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--epoch_num_steps", type=int, default=100000)
    parser.add_argument("--collection_step_num_env_steps", type=int, default=10)
    parser.add_argument("--update_per_step", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_sizes", type=int, nargs="*", default=[128, 128])
    parser.add_argument("--num_train_envs", type=int, default=10)
    parser.add_argument("--num_test_envs", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument("--n_step", type=int, default=4)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--resume_path", type=str, default=None)
    return parser.parse_args()


class Wrapper(gym.Wrapper):
    """Env wrapper for reward scale, action repeat and removing done penalty."""

    def __init__(
        self,
        env: gym.Env,
        action_repeat: int = 3,
        reward_scale: int = 5,
        rm_done: bool = True,
    ) -> None:
        super().__init__(env)
        self.action_repeat = action_repeat
        self.reward_scale = reward_scale
        self.rm_done = rm_done

    def step(
        self,
        action: WrapperActType,
    ) -> tuple[WrapperObsType, float, bool, bool, dict[str, Any]]:
        rew_sum = 0.0
        for _ in range(self.action_repeat):
            obs, rew, terminated, truncated, info = self.env.step(action)
            done = terminated | truncated
            # remove done reward penalty
            if not done or not self.rm_done:
                rew_sum = rew_sum + float(rew)
            if done:
                break
        # scale reward
        return obs, self.reward_scale * rew_sum, terminated, truncated, info


def test_sac_bipedal(args: argparse.Namespace = get_args()) -> None:
    env = Wrapper(gym.make(args.task))
    space_info = SpaceInfo.from_env(env)
    args.state_shape = space_info.observation_info.obs_shape
    args.action_shape = space_info.action_info.action_shape
    args.max_action = space_info.action_info.max_action
    train_envs = SubprocVectorEnv(
        [lambda: Wrapper(gym.make(args.task)) for _ in range(args.num_train_envs)],
    )
    # test_envs = gym.make(args.task)
    test_envs = SubprocVectorEnv(
        [
            lambda: Wrapper(gym.make(args.task), reward_scale=1, rm_done=False)
            for _ in range(args.test_num)
        ],
    )

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # model
    net_a = Net(state_shape=args.state_shape, hidden_sizes=args.hidden_sizes)
    actor = ContinuousActorProbabilistic(
        preprocess_net=net_a,
        action_shape=args.action_shape,
        unbounded=True,
    ).to(args.device)
    actor_optim = AdamOptimizerFactory(lr=args.actor_lr)

    net_c1 = Net(
        state_shape=args.state_shape,
        action_shape=args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
    )
    critic1 = ContinuousCritic(preprocess_net=net_c1).to(args.device)
    critic1_optim = AdamOptimizerFactory(lr=args.critic_lr)

    net_c2 = Net(
        state_shape=args.state_shape,
        action_shape=args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
    )
    critic2 = ContinuousCritic(preprocess_net=net_c2).to(args.device)
    critic2_optim = AdamOptimizerFactory(lr=args.critic_lr)

    action_dim = space_info.action_info.action_dim
    if args.auto_alpha:
        target_entropy = -action_dim
        log_alpha = 0.0
        alpha_optim = AdamOptimizerFactory(lr=args.alpha_lr)
        args.alpha = AutoAlpha(target_entropy, log_alpha, alpha_optim).to(args.device)

    policy = SACPolicy(
        actor=actor,
        action_space=env.action_space,
    )
    algorithm: SAC = SAC(
        policy=policy,
        policy_optim=actor_optim,
        critic=critic1,
        critic_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        n_step_return_horizon=args.n_step,
    )
    # load a previous policy
    if args.resume_path:
        algorithm.load_state_dict(torch.load(args.resume_path))
        print("Loaded agent from: ", args.resume_path)

    # collector
    train_collector = Collector[CollectStats](
        algorithm,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector[CollectStats](algorithm, test_envs)
    # train_collector.collect(n_step=args.buffer_size)
    # log
    log_path = os.path.join(args.logdir, args.task, "sac")
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy: Algorithm) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards: float) -> bool:
        if env.spec:
            if not env.spec.reward_threshold:
                return False
            else:
                return mean_rewards >= env.spec.reward_threshold
        return False

    # train
    result = algorithm.run_training(
        OffPolicyTrainerParams(
            train_collector=train_collector,
            test_collector=test_collector,
            max_epochs=args.epoch,
            epoch_num_steps=args.epoch_num_steps,
            collection_step_num_env_steps=args.collection_step_num_env_steps,
            test_step_num_episodes=args.test_num,
            batch_size=args.batch_size,
            update_step_num_gradient_steps_per_sample=args.update_per_step,
            test_in_train=False,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
        )
    )

    if __name__ == "__main__":
        pprint.pprint(result)
        # Let's watch its performance!
        test_envs.seed(args.seed)
        test_collector.reset()
        collector_stats = test_collector.collect(n_episode=args.test_num, render=args.render)
        print(collector_stats)


if __name__ == "__main__":
    test_sac_bipedal()
