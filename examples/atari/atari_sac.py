import argparse
import datetime
import os
import pprint
import sys

import numpy as np
import torch

from tianshou.data import Collector, CollectStats, VectorReplayBuffer
from tianshou.env.atari.atari_network import DQNet
from tianshou.env.atari.atari_wrapper import make_atari_env
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.algorithm import DiscreteSAC, ICMOffPolicyWrapper
from tianshou.algorithm.base import Algorithm
from tianshou.algorithm.modelfree.discrete_sac import DiscreteSACPolicy
from tianshou.algorithm.modelfree.sac import AutoAlpha
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.utils.net.discrete import (
    DiscreteActor,
    DiscreteCritic,
    IntrinsicCuriosityModule,
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="PongNoFrameskip-v4")
    parser.add_argument("--seed", type=int, default=4213)
    parser.add_argument("--scale-obs", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--actor-lr", type=float, default=1e-5)
    parser.add_argument("--critic-lr", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--auto-alpha", action="store_true", default=False)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=100000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--rew-norm", type=int, default=False)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--frames-stack", type=int, default=4)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="atari.benchmark")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    parser.add_argument("--save-buffer-name", type=str, default=None)
    parser.add_argument(
        "--icm-lr-scale",
        type=float,
        default=0.0,
        help="use intrinsic curiosity module with this lr scale",
    )
    parser.add_argument(
        "--icm-reward-scale",
        type=float,
        default=0.01,
        help="scaling factor for intrinsic curiosity reward",
    )
    parser.add_argument(
        "--icm-forward-loss-weight",
        type=float,
        default=0.2,
        help="weight for the forward model loss in ICM",
    )
    return parser.parse_args()


def test_discrete_sac(args: argparse.Namespace = get_args()) -> None:
    env, train_envs, test_envs = make_atari_env(
        args.task,
        args.seed,
        args.training_num,
        args.test_num,
        scale=args.scale_obs,
        frame_stack=args.frames_stack,
    )
    c, h, w = env.observation_space.shape  # type: ignore
    args.action_shape = env.action_space.n  # type: ignore

    # should be N_FRAMES x H x W
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # define model
    net = DQNet(
        c,
        h,
        w,
        action_shape=args.action_shape,
        features_only=True,
        output_dim_added_layer=args.hidden_size,
    )
    actor = DiscreteActor(preprocess_net=net, action_shape=args.action_shape, softmax_output=False)
    actor_optim = AdamOptimizerFactory(lr=args.actor_lr)
    critic1 = DiscreteCritic(preprocess_net=net, last_size=args.action_shape)
    critic1_optim = AdamOptimizerFactory(lr=args.critic_lr)
    critic2 = DiscreteCritic(preprocess_net=net, last_size=args.action_shape)
    critic2_optim = AdamOptimizerFactory(lr=args.critic_lr)

    # define policy and algorithm
    if args.auto_alpha:
        target_entropy = 0.98 * np.log(np.prod(args.action_shape))
        log_alpha = 0.0
        alpha_optim = AdamOptimizerFactory(lr=args.alpha_lr)
        args.alpha = AutoAlpha(target_entropy, log_alpha, alpha_optim)
    algorithm: DiscreteSAC | ICMOffPolicyWrapper
    policy = DiscreteSACPolicy(
        actor=actor,
        action_space=env.action_space,
    )
    algorithm = DiscreteSAC(
        policy=policy,
        policy_optim=actor_optim,
        critic=critic1,
        critic_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        estimation_step=args.n_step,
    ).to(args.device)
    if args.icm_lr_scale > 0:
        c, h, w = args.state_shape
        feature_net = DQNet(c=c, h=h, w=w, action_shape=args.action_shape, features_only=True)
        action_dim = np.prod(args.action_shape)
        feature_dim = feature_net.output_dim
        icm_net = IntrinsicCuriosityModule(
            feature_net=feature_net.net,
            feature_dim=feature_dim,
            action_dim=int(action_dim),
            hidden_sizes=[args.hidden_size],
        )
        icm_optim = AdamOptimizerFactory(lr=args.actor_lr)
        algorithm = ICMOffPolicyWrapper(
            wrapped_algorithm=algorithm,
            model=icm_net,
            optim=icm_optim,
            lr_scale=args.icm_lr_scale,
            reward_scale=args.icm_reward_scale,
            forward_loss_weight=args.icm_forward_loss_weight,
        ).to(args.device)

    # load a previous model
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
    args.algo_name = "discrete_sac_icm" if args.icm_lr_scale > 0 else "discrete_sac"
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
        if env.spec.reward_threshold:  # type: ignore
            return mean_rewards >= env.spec.reward_threshold  # type: ignore
        if "Pong" in args.task:
            return mean_rewards >= 20
        return False

    def save_checkpoint_fn(epoch: int, env_step: int, gradient_step: int) -> str:
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        torch.save({"model": algorithm.state_dict()}, ckpt_path)
        return ckpt_path

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
            result = test_collector.collect(n_episode=args.test_num, render=args.render)
        result.pprint_asdict()

    if args.watch:
        watch()
        sys.exit(0)

    # test train_collector and start filling replay buffer
    train_collector.reset()
    train_collector.collect(n_step=args.batch_size * args.training_num)

    # train
    result = algorithm.run_training(
        OffPolicyTrainerParams(
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            step_per_collect=args.step_per_collect,
            episode_per_test=args.test_num,
            batch_size=args.batch_size,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            update_per_step=args.update_per_step,
            test_in_train=False,
            resume_from_log=args.resume_id is not None,
            save_checkpoint_fn=save_checkpoint_fn,
        )
    )

    pprint.pprint(result)
    watch()


if __name__ == "__main__":
    test_discrete_sac(get_args())
