import argparse
import os

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from torch.utils.tensorboard import SummaryWriter

from tianshou.algorithm import PPO
from tianshou.algorithm.algorithm_base import Algorithm
from tianshou.algorithm.modelbased.icm import ICMOnPolicyWrapper
from tianshou.algorithm.modelfree.reinforce import ActorPolicyProbabilistic
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Collector, CollectStats, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OnPolicyTrainerParams
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import MLP, ActorCritic, MLPActor
from tianshou.utils.net.discrete import (
    DiscreteActor,
    DiscreteCritic,
    IntrinsicCuriosityModule,
)
from tianshou.utils.space_info import SpaceInfo


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="CartPole-v1")
    parser.add_argument("--reward-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--epoch_num_steps", type=int, default=50000)
    parser.add_argument("--collection_step_num_env_steps", type=int, default=2000)
    parser.add_argument("--update_step_num_repetitions", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--num_train_envs", type=int, default=20)
    parser.add_argument("--num_test_envs", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    # ppo special
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--eps-clip", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--return_scaling", type=int, default=0)
    parser.add_argument("--advantage_normalization", type=int, default=0)
    parser.add_argument("--recompute-adv", type=int, default=0)
    parser.add_argument("--dual-clip", type=float, default=None)
    parser.add_argument("--value-clip", type=int, default=0)
    parser.add_argument(
        "--lr-scale",
        type=float,
        default=1.0,
        help="use intrinsic curiosity module with this lr scale",
    )
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=0.01,
        help="scaling factor for intrinsic curiosity reward",
    )
    parser.add_argument(
        "--forward-loss-weight",
        type=float,
        default=0.2,
        help="weight for the forward model loss in ICM",
    )
    return parser.parse_known_args()[0]


def test_ppo(args: argparse.Namespace = get_args()) -> None:
    env = gym.make(args.task)

    space_info = SpaceInfo.from_env(env)
    args.state_shape = space_info.observation_info.obs_shape
    args.action_shape = space_info.action_info.action_shape

    if args.reward_threshold is None:
        default_reward_threshold = {"CartPole-v1": 195}
        args.reward_threshold = default_reward_threshold.get(
            args.task,
            env.spec.reward_threshold if env.spec else None,
        )

    train_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.num_train_envs)])
    test_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.test_num)])

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # model
    net = MLPActor(state_shape=args.state_shape, hidden_sizes=args.hidden_sizes)
    actor = DiscreteActor(preprocess_net=net, action_shape=args.action_shape).to(args.device)
    critic = DiscreteCritic(preprocess_net=net).to(args.device)
    actor_critic = ActorCritic(actor, critic)

    # orthogonal initialization
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    # base algorithm: PPO
    optim = AdamOptimizerFactory(lr=args.lr)
    dist = torch.distributions.Categorical
    policy = ActorPolicyProbabilistic(
        actor=actor,
        dist_fn=dist,
        action_scaling=isinstance(env.action_space, Box),
        action_space=env.action_space,
        deterministic_eval=True,
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
        gae_lambda=args.gae_lambda,
        return_scaling=args.return_scaling,
        dual_clip=args.dual_clip,
        value_clip=args.value_clip,
        advantage_normalization=args.advantage_normalization,
        recompute_advantage=args.recompute_adv,
    )

    # ICM wrapper
    feature_dim = args.hidden_sizes[-1]
    feature_net = MLP(
        input_dim=space_info.observation_info.obs_dim,
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
    icm_algorithm = ICMOnPolicyWrapper(
        wrapped_algorithm=algorithm,
        model=icm_net,
        optim=icm_optim,
        lr_scale=args.lr_scale,
        reward_scale=args.reward_scale,
        forward_loss_weight=args.forward_loss_weight,
    )

    # collector
    train_collector = Collector[CollectStats](
        icm_algorithm,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
    )
    test_collector = Collector[CollectStats](icm_algorithm, test_envs)

    # log
    log_path = os.path.join(args.logdir, args.task, "ppo_icm")
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(alg: Algorithm) -> None:
        torch.save(alg.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards: float) -> bool:
        return mean_rewards >= args.reward_threshold

    # train
    result = icm_algorithm.run_training(
        OnPolicyTrainerParams(
            train_collector=train_collector,
            test_collector=test_collector,
            max_epochs=args.epoch,
            epoch_num_steps=args.epoch_num_steps,
            update_step_num_repetitions=args.update_step_num_repetitions,
            test_step_num_episodes=args.test_num,
            batch_size=args.batch_size,
            collection_step_num_env_steps=args.collection_step_num_env_steps,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            test_in_train=True,
        )
    )
    assert stop_fn(result.best_reward)
