import argparse
import datetime
import os
import pprint
import sys

import numpy as np
import torch
from env import make_vizdoom_env
from torch.distributions import Categorical

from tianshou.algorithm import PPO
from tianshou.algorithm.algorithm_base import Algorithm
from tianshou.algorithm.modelbased.icm import ICMOnPolicyWrapper
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory, LRSchedulerFactoryLinear
from tianshou.data import Collector, CollectStats, VectorReplayBuffer
from tianshou.env.atari.atari_network import DQNet
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.trainer import OnPolicyTrainerParams
from tianshou.utils.net.discrete import (
    DiscreteActor,
    DiscreteCritic,
    IntrinsicCuriosityModule,
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="D1_basic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.00002)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--epoch_num_steps", type=int, default=100000)
    parser.add_argument("--collection_step_num_env_steps", type=int, default=1000)
    parser.add_argument("--update_step_num_repetitions", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_train_envs", type=int, default=10)
    parser.add_argument("--num_test_envs", type=int, default=10)
    parser.add_argument("--return_scaling", type=int, default=False)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--lr_decay", type=int, default=True)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--eps_clip", type=float, default=0.2)
    parser.add_argument("--dual_clip", type=float, default=None)
    parser.add_argument("--value_clip", type=int, default=0)
    parser.add_argument("--advantage_normalization", type=int, default=1)
    parser.add_argument("--recompute_adv", type=int, default=0)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--frames_stack", type=int, default=4)
    parser.add_argument("--skip_num", type=int, default=4)
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--resume_id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb_project", type=str, default="vizdoom.benchmark")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    parser.add_argument(
        "--save_lmp",
        default=False,
        action="store_true",
        help="save lmp file for replay whole episode",
    )
    parser.add_argument("--save_buffer_name", type=str, default=None)
    parser.add_argument(
        "--icm_lr_scale",
        type=float,
        default=0.0,
        help="use intrinsic curiosity module with this lr scale",
    )
    parser.add_argument(
        "--icm_reward_scale",
        type=float,
        default=0.01,
        help="scaling factor for intrinsic curiosity reward",
    )
    parser.add_argument(
        "--icm_forward_loss_weight",
        type=float,
        default=0.2,
        help="weight for the forward model loss in ICM",
    )
    return parser.parse_args()


def test_ppo(args: argparse.Namespace = get_args()) -> None:
    # make environments
    env, train_envs, test_envs = make_vizdoom_env(
        args.task,
        args.skip_num,
        (args.frames_stack, 84, 84),
        args.save_lmp,
        args.seed,
        args.num_train_envs,
        args.num_test_envs,
    )
    args.state_shape = env.observation_space.shape
    args.action_shape = env.action_space.n
    # should be N_FRAMES x H x W
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # define model
    c, h, w = args.state_shape
    net = DQNet(
        c=c,
        h=h,
        w=w,
        action_shape=args.action_shape,
        features_only=True,
        output_dim_added_layer=args.hidden_size,
    )
    actor = DiscreteActor(preprocess_net=net, action_shape=args.action_shape, softmax_output=False)
    critic = DiscreteCritic(preprocess_net=net)
    optim = AdamOptimizerFactory(lr=args.lr)

    if args.lr_decay:
        optim.with_lr_scheduler_factory(
            LRSchedulerFactoryLinear(
                max_epochs=args.epoch,
                epoch_num_steps=args.epoch_num_steps,
                collection_step_num_env_steps=args.collection_step_num_env_steps,
            )
        )

    def dist(logits: torch.Tensor) -> Categorical:
        return Categorical(logits=logits)

    # define policy and algorithm
    policy = ProbabilisticActorPolicy(
        actor=actor,
        dist_fn=dist,
        action_scaling=False,
        action_space=env.action_space,
    )
    algorithm: PPO | ICMOnPolicyWrapper
    algorithm = PPO(
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
    ).to(args.device)
    if args.icm_lr_scale > 0:
        c, h, w = args.state_shape
        feature_net = DQNet(
            c=c,
            h=h,
            w=w,
            action_shape=args.action_shape,
            features_only=True,
            output_dim_added_layer=args.hidden_size,
        )
        action_dim = np.prod(args.action_shape)
        feature_dim = feature_net.output_dim
        icm_net = IntrinsicCuriosityModule(
            feature_net=feature_net.net,
            feature_dim=feature_dim,
            action_dim=action_dim,
        )
        icm_optim = AdamOptimizerFactory(lr=args.lr)
        algorithm = ICMOnPolicyWrapper(
            wrapped_algorithm=algorithm,
            model=icm_net,
            optim=icm_optim,
            lr_scale=args.icm_lr_scale,
            reward_scale=args.icm_reward_scale,
            forward_loss_weight=args.icm_forward_loss_weight,
        ).to(args.device)
    # load a previous policy
    if args.resume_path:
        algorithm.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)
    # replay buffer: `save_last_obs` and `stack_num` can be removed together
    # when you have enough RAM
    buffer = VectorReplayBuffer(
        args.buffer_size,
        buffer_num=len(train_envs),
        ignore_obs_next=True,
        save_only_last_obs=True,
        stack_num=args.frames_stack,
    )
    # collector
    train_collector = Collector[CollectStats](algorithm, train_envs, buffer, exploration_noise=True)
    test_collector = Collector[CollectStats](algorithm, test_envs, exploration_noise=True)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "ppo_icm" if args.icm_lr_scale > 0 else "ppo"
    log_name = os.path.join(args.task, args.algo_name, str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)

    # logger
    logger_factory = LoggerFactoryDefault()
    if args.logger == "wandb":
        logger_factory.logger_type = "wandb"
        logger_factory.wandb_project = args.wandb_project
    else:
        logger_factory.logger_type = "tensorboard"

    logger = logger_factory.create_logger(
        log_dir=log_path,
        experiment_name=log_name,
        run_id=args.resume_id,
        config_dict=vars(args),
    )

    def save_best_fn(policy: Algorithm) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards: float) -> bool:
        if env.spec.reward_threshold:
            return mean_rewards >= env.spec.reward_threshold
        return False

    # watch agent's performance
    def watch() -> None:
        print("Setup test envs ...")
        test_envs.seed(args.seed)
        if args.save_buffer_name:
            print(f"Generate buffer with size {args.buffer_size}")
            buffer = VectorReplayBuffer(
                args.buffer_size,
                buffer_num=len(test_envs),
                ignore_obs_next=True,
                save_only_last_obs=True,
                stack_num=args.frames_stack,
            )
            collector = Collector[CollectStats](
                algorithm, test_envs, buffer, exploration_noise=True
            )
            result = collector.collect(n_step=args.buffer_size, reset_before_collect=True)
            print(f"Save buffer into {args.save_buffer_name}")
            # Unfortunately, pickle will cause oom with 1M buffer size
            buffer.save_hdf5(args.save_buffer_name)
        else:
            print("Testing agent ...")
            test_collector.reset()
            result = test_collector.collect(n_episode=args.num_test_envs, render=args.render)
        result.pprint_asdict()

    if args.watch:
        watch()
        sys.exit(0)

    # test train_collector and start filling replay buffer
    train_collector.reset()
    train_collector.collect(n_step=args.batch_size * args.num_train_envs)

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
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            test_in_train=False,
        )
    )

    pprint.pprint(result)
    watch()


if __name__ == "__main__":
    test_ppo(get_args())
