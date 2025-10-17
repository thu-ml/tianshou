"""
A low-level counterpart of `mujoco_ppo_hl_multi.py`. The directory structure of the persisted logs
mirrors that of the high-level example.

Rollout of multiple experiments with different seeds is done manually in a for loop,
the results are aggregated and evaluated with rliable.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from mujoco_env import make_mujoco_env
from sensai.util import logging
from sensai.util.logging import datetime_tag
from torch import nn
from torch.distributions import Distribution, Independent, Normal

from tianshou.algorithm import PPO
from tianshou.algorithm.algorithm_base import Algorithm
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory, LRSchedulerFactoryLinear
from tianshou.data import (
    Collector,
    CollectStats,
    ReplayBuffer,
    VectorReplayBuffer,
)
from tianshou.evaluation.rliable_evaluation_hl import RLiableExperimentResult
from tianshou.highlevel.experiment import Experiment
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.trainer import OnPolicyTrainerParams
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ContinuousActorProbabilistic, ContinuousCritic

DATETIME_TAG = datetime_tag()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Ant-v4")
    parser.add_argument("--num_experiments", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer_size", type=int, default=4096)
    parser.add_argument("--hidden_sizes", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--epoch_num_steps", type=int, default=2000)
    parser.add_argument("--collection_step_num_env_steps", type=int, default=2048)
    parser.add_argument("--update_step_num_repetitions", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_train_envs", type=int, default=8)
    parser.add_argument("--num_test_envs", type=int, default=10)
    # ppo special
    parser.add_argument("--return_scaling", type=int, default=True)
    # In theory, `vf-coef` will not make any difference if using Adam optimizer.
    parser.add_argument("--vf_coef", type=float, default=0.25)
    parser.add_argument("--ent_coef", type=float, default=0.0)
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
    parser.add_argument("--resume_id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb_project", type=str, default="mujoco.benchmark")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    return parser.parse_args()


def get_persistence_dir(args: argparse.Namespace) -> str:
    algo_name = "ppo"
    log_subdir = os.path.join(
        args.task, f"{algo_name}_{DATETIME_TAG}", Experiment.seeding_info_str_static(args.seed)
    )
    return os.path.join(args.logdir, log_subdir)


def main(args: argparse.Namespace = get_args()) -> None:
    print("Creating envs")
    env, train_envs, test_envs = make_mujoco_env(
        args.task,
        args.seed,
        args.num_train_envs,
        args.num_test_envs,
        obs_norm=True,
    )
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
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
    actor_critic = ActorCritic(actor, critic)

    torch.nn.init.constant_(actor.sigma_param, -0.5)
    for m in actor_critic.modules():
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

    policy = ProbabilisticActorPolicy(
        actor=actor,
        dist_fn=dist,
        action_scaling=True,
        action_bound_method=args.bound_action_method,
        action_space=env.action_space,
    )
    algorithm: PPO = PPO(
        policy=policy,
        critic=critic,
        optim=optim,
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
        ckpt = torch.load(args.resume_path, map_location=args.device)
        algorithm.load_state_dict(ckpt["model"])
        train_envs.set_obs_rms(ckpt["obs_rms"])
        test_envs.set_obs_rms(ckpt["obs_rms"])
        print("Loaded agent from: ", args.resume_path)

    # collector
    buffer: VectorReplayBuffer | ReplayBuffer
    if args.num_train_envs > 1:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(args.buffer_size)
    train_collector = Collector[CollectStats](algorithm, train_envs, buffer, exploration_noise=True)
    test_collector = Collector[CollectStats](algorithm, test_envs)

    # log
    persistence_dir = get_persistence_dir(args)
    experiment_subpath = Path(persistence_dir).relative_to(args.logdir)

    # logger
    logger_factory = LoggerFactoryDefault()
    if args.logger == "wandb":
        logger_factory.logger_type = "wandb"
        logger_factory.wandb_project = args.wandb_project
    else:
        logger_factory.logger_type = "tensorboard"

    logger = logger_factory.create_logger(
        log_dir=persistence_dir,
        experiment_name=str(experiment_subpath),
        run_id=args.resume_id,
        config_dict=vars(args),
    )

    def save_best_fn(policy: Algorithm) -> None:
        state = {"model": policy.state_dict(), "obs_rms": train_envs.get_obs_rms()}
        torch.save(state, os.path.join(persistence_dir, "policy.pth"))

    print("Running the training")
    algorithm.run_training(
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


if __name__ == "__main__":
    args = get_args()
    main_seed = args.seed

    # Manual rollout of multiple experiments differing only by seed.
    # If desired, this can be parallelized, e.g., using joblib.
    # Often one doesn't gain much from parallelization on a single machine though, as each experiment is already
    # using multiple cores (parallelized rollouts)
    for i in range(args.num_experiments):
        print(f"Running experiment {i + 1}/{args.num_experiments} with seed {main_seed + i}")
        args.seed = main_seed + i
        logging.run_main(lambda: main(args=args), level=logging.INFO)

    # Evaluate the results with rliable
    persistence_dir_all_seeds = str(Path(get_persistence_dir(args)).parent)
    rliable_result = RLiableExperimentResult.load_from_disk(persistence_dir_all_seeds)
    rliable_result.eval_results(save_plots=True)
