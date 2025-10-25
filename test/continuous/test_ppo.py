import argparse
import os

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Distribution, Independent, Normal
from torch.utils.tensorboard import SummaryWriter

from test.determinism_test import AlgorithmDeterminismTest
from tianshou.algorithm import PPO
from tianshou.algorithm.algorithm_base import Algorithm
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Collector, CollectStats, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OnPolicyTrainerParams
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ContinuousActorProbabilistic, ContinuousCritic
from tianshou.utils.space_info import SpaceInfo


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Pendulum-v1")
    parser.add_argument("--reward_threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--buffer_size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--epoch_num_steps", type=int, default=150000)
    parser.add_argument("--collection_step_num_episodes", type=int, default=16)
    parser.add_argument("--update_step_num_repetitions", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_sizes", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--num_training_envs", type=int, default=16)
    parser.add_argument("--num_test_envs", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    # ppo special
    parser.add_argument("--vf_coef", type=float, default=0.25)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--eps_clip", type=float, default=0.2)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--return_scaling", type=int, default=1)
    parser.add_argument("--dual_clip", type=float, default=None)
    parser.add_argument("--value_clip", type=int, default=1)
    parser.add_argument("--advantage_normalization", type=int, default=1)
    parser.add_argument("--recompute_adv", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_interval", type=int, default=4)
    return parser.parse_known_args()[0]


def test_ppo(args: argparse.Namespace = get_args(), enable_assertions: bool = True) -> None:
    env = gym.make(args.task)

    space_info = SpaceInfo.from_env(env)
    args.state_shape = space_info.observation_info.obs_shape
    args.action_shape = space_info.action_info.action_shape
    args.max_action = space_info.action_info.max_action

    if args.reward_threshold is None:
        default_reward_threshold = {"Pendulum-v0": -250, "Pendulum-v1": -250}
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

    # model
    net = Net(state_shape=args.state_shape, hidden_sizes=args.hidden_sizes)
    actor = ContinuousActorProbabilistic(
        preprocess_net=net, action_shape=args.action_shape, unbounded=True
    ).to(args.device)
    critic = ContinuousCritic(
        preprocess_net=Net(state_shape=args.state_shape, hidden_sizes=args.hidden_sizes),
    ).to(args.device)
    actor_critic = ActorCritic(actor, critic)
    # orthogonal initialization
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    optim = AdamOptimizerFactory(lr=args.lr)

    # replace DiagGuassian with Independent(Normal) which is equivalent
    # pass *logits to be consistent with policy.forward
    def dist(loc_scale: tuple[torch.Tensor, torch.Tensor]) -> Distribution:
        loc, scale = loc_scale
        return Independent(Normal(loc, scale), 1)

    policy = ProbabilisticActorPolicy(
        actor=actor,
        dist_fn=dist,
        action_space=env.action_space,
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
        dual_clip=args.dual_clip,
        value_clip=args.value_clip,
        gae_lambda=args.gae_lambda,
    )
    # collector
    train_collector = Collector[CollectStats](
        algorithm,
        training_envs,
        VectorReplayBuffer(args.buffer_size, len(training_envs)),
    )
    test_collector = Collector[CollectStats](algorithm, test_envs)
    # log
    log_path = os.path.join(args.logdir, args.task, "ppo")
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer, save_interval=args.save_interval)

    def save_best_fn(policy: Algorithm) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards: float) -> bool:
        return mean_rewards >= args.reward_threshold

    def save_checkpoint_fn(epoch: int, env_step: int, gradient_step: int) -> str:
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        # Example: saving by epoch num
        # ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
        torch.save(
            algorithm.state_dict(),
            ckpt_path,
        )
        return ckpt_path

    if args.resume:
        # load from existing checkpoint
        print(f"Loading agent under {log_path}")
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=args.device)
            algorithm.load_state_dict(checkpoint)
            print("Successfully restore policy and optim.")
        else:
            print("Fail to restore policy and optim.")

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
            collection_step_num_episodes=args.collection_step_num_episodes,
            collection_step_num_env_steps=None,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            resume_from_log=args.resume,
            save_checkpoint_fn=save_checkpoint_fn,
            test_in_train=True,
        )
    )

    if enable_assertions:
        assert stop_fn(result.best_reward)


def test_ppo_resume(args: argparse.Namespace = get_args()) -> None:
    args.resume = True
    test_ppo(args)


def test_ppo_determinism() -> None:
    main_fn = lambda args: test_ppo(args, enable_assertions=False)
    AlgorithmDeterminismTest("continuous_ppo", main_fn, get_args()).run()
