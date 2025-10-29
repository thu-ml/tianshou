import argparse
import os

import numpy as np
import pytest
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.algorithm import PSRL
from tianshou.algorithm.modelbased.psrl import PSRLPolicy
from tianshou.data import Collector, CollectStats, VectorReplayBuffer
from tianshou.trainer import OnPolicyTrainerParams
from tianshou.utils import LazyLogger, TensorboardLogger, WandbLogger

try:
    import envpool
except ImportError:
    envpool = None


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="NChain-v0")
    parser.add_argument("--reward_threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--buffer_size", type=int, default=50000)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--epoch_num_steps", type=int, default=1000)
    parser.add_argument("--collection_step_num_episodes", type=int, default=1)
    parser.add_argument("--num_training_envs", type=int, default=1)
    parser.add_argument("--num_test_envs", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument("--rew_mean_prior", type=float, default=0.0)
    parser.add_argument("--rew_std_prior", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps", type=float, default=0.01)
    parser.add_argument("--add_done_loop", action="store_true", default=False)
    parser.add_argument(
        "--logger",
        type=str,
        default="none",  # TODO: Change to "wandb" once wandb supports Gym >=0.26.0
        choices=["wandb", "tensorboard", "none"],
    )
    return parser.parse_known_args()[0]


@pytest.mark.skipif(
    envpool is None,
    reason="EnvPool is not installed. If on linux, please install it (e.g. as poetry extra)",
)
def test_psrl(args: argparse.Namespace = get_args()) -> None:
    training_envs = env = envpool.make_gymnasium(
        args.task, num_envs=args.num_training_envs, seed=args.seed
    )
    test_envs = envpool.make_gymnasium(args.task, num_envs=args.num_test_envs, seed=args.seed)
    if args.reward_threshold is None:
        default_reward_threshold = {"NChain-v0": 3400}
        args.reward_threshold = default_reward_threshold.get(args.task, env.spec.reward_threshold)
    print("reward threshold:", args.reward_threshold)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # model
    n_action = args.action_shape
    n_state = args.state_shape
    trans_count_prior = np.ones((n_state, n_action, n_state))
    rew_mean_prior = np.full((n_state, n_action), args.rew_mean_prior)
    rew_std_prior = np.full((n_state, n_action), args.rew_std_prior)
    policy = PSRLPolicy(
        trans_count_prior=trans_count_prior,
        rew_mean_prior=rew_mean_prior,
        rew_std_prior=rew_std_prior,
        action_space=env.action_space,
        discount_factor=args.gamma,
        epsilon=args.eps,
    )
    algorithm: PSRL = PSRL(
        policy=policy,
        add_done_loop=args.add_done_loop,
    )

    # collector
    training_collector = Collector[CollectStats](
        algorithm,
        training_envs,
        VectorReplayBuffer(args.buffer_size, len(training_envs)),
        exploration_noise=True,
    )
    training_collector.reset()
    test_collector = Collector[CollectStats](algorithm, test_envs)
    test_collector.reset()

    # Logger
    log_path = os.path.join(args.logdir, args.task, "psrl")
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger: WandbLogger | TensorboardLogger | LazyLogger
    if args.logger == "wandb":
        logger = WandbLogger(save_interval=1, project="psrl", name="wandb_test", config=args)
        logger.load(writer)
    elif args.logger == "tensorboard":
        logger = TensorboardLogger(writer)
    else:
        logger = LazyLogger()

    def stop_fn(mean_rewards: float) -> bool:
        return mean_rewards >= args.reward_threshold

    training_collector.collect(n_step=args.buffer_size, random=True)

    # train
    result = algorithm.run_training(
        OnPolicyTrainerParams(
            training_collector=training_collector,
            test_collector=test_collector,
            max_epochs=args.epoch,
            epoch_num_steps=args.epoch_num_steps,
            update_step_num_repetitions=1,
            test_step_num_episodes=args.num_test_envs,
            batch_size=0,
            collection_step_num_episodes=args.collection_step_num_episodes,
            collection_step_num_env_steps=None,
            stop_fn=stop_fn,
            logger=logger,
            test_in_training=False,
        )
    )
    assert result.best_reward >= args.reward_threshold
