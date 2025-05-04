import argparse
import datetime
import os
import pprint
import sys
from collections.abc import Sequence
from typing import cast

import numpy as np
import torch

from tianshou.data import Collector, CollectStats, VectorReplayBuffer
from tianshou.env.atari.atari_network import (
    DQNet,
    ScaledObsInputModule,
    layer_init,
    scale_obs,
)
from tianshou.env.atari.atari_wrapper import make_atari_env
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.policy import PPO
from tianshou.policy.base import Algorithm
from tianshou.policy.modelbased.icm import ICMOnPolicyWrapper
from tianshou.policy.modelfree.pg import DiscreteActorPolicy
from tianshou.policy.optim import AdamOptimizerFactory, LRSchedulerFactoryLinear
from tianshou.trainer import OnPolicyTrainerParams
from tianshou.utils.net.discrete import (
    DiscreteActor,
    DiscreteCritic,
    IntrinsicCuriosityModule,
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="PongNoFrameskip-v4")
    parser.add_argument("--seed", type=int, default=4213)
    parser.add_argument("--scale-obs", type=int, default=1)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=100000)
    parser.add_argument("--step-per-collect", type=int, default=1000)
    parser.add_argument("--repeat-per-collect", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--rew-norm", type=int, default=False)
    parser.add_argument("--vf-coef", type=float, default=0.25)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--lr-decay", type=int, default=True)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--eps-clip", type=float, default=0.1)
    parser.add_argument("--dual-clip", type=float, default=None)
    parser.add_argument("--value-clip", type=int, default=1)
    parser.add_argument("--norm-adv", type=int, default=1)
    parser.add_argument("--recompute-adv", type=int, default=0)
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


def main(args: argparse.Namespace = get_args()) -> None:
    env, train_envs, test_envs = make_atari_env(
        args.task,
        args.seed,
        args.training_num,
        args.test_num,
        scale=0,
        frame_stack=args.frames_stack,
    )
    args.state_shape = cast(tuple[int, ...], env.observation_space.shape)
    args.action_shape = cast(Sequence[int] | int, env.action_space.shape or env.action_space.n)  # type: ignore
    # should be N_FRAMES x H x W
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # define model
    c, h, w = args.state_shape
    net: ScaledObsInputModule | DQNet
    net = DQNet(
        c=c,
        h=h,
        w=w,
        action_shape=args.action_shape,
        features_only=True,
        output_dim_added_layer=args.hidden_size,
        layer_init=layer_init,
    )
    if args.scale_obs:
        net = scale_obs(net)
    actor = DiscreteActor(preprocess_net=net, action_shape=args.action_shape, softmax_output=False)
    critic = DiscreteCritic(preprocess_net=net)
    optim = AdamOptimizerFactory(lr=args.lr, eps=1e-5)

    if args.lr_decay:
        optim.with_lr_scheduler_factory(
            LRSchedulerFactoryLinear(
                num_epochs=args.epoch,
                step_per_epoch=args.step_per_epoch,
                step_per_collect=args.step_per_collect,
            )
        )

    # define algorithm
    policy = DiscreteActorPolicy(
        actor=actor,
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
        reward_normalization=args.rew_norm,
        eps_clip=args.eps_clip,
        value_clip=args.value_clip,
        dual_clip=args.dual_clip,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
    ).to(args.device)
    if args.icm_lr_scale > 0:
        c, h, w = args.state_shape
        feature_net = DQNet(c=c, h=h, w=w, action_shape=args.action_shape, features_only=True)
        action_dim = int(np.prod(args.action_shape))
        feature_dim = feature_net.output_dim
        icm_net = IntrinsicCuriosityModule(
            feature_net=feature_net.net,
            feature_dim=feature_dim,
            action_dim=action_dim,
            hidden_sizes=[args.hidden_size],
        )
        icm_optim = AdamOptimizerFactory(lr=args.lr)
        algorithm = ICMOnPolicyWrapper(  # type: ignore[assignment]
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
        if env.spec.reward_threshold:  # type: ignore
            return mean_rewards >= env.spec.reward_threshold  # type: ignore
        if "Pong" in args.task:
            return mean_rewards >= 20
        return False

    def save_checkpoint_fn(epoch: int, env_step: int, gradient_step: int) -> str:
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
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
        OnPolicyTrainerParams(
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            repeat_per_collect=args.repeat_per_collect,
            episode_per_test=args.test_num,
            batch_size=args.batch_size,
            step_per_collect=args.step_per_collect,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            test_in_train=False,
            resume_from_log=args.resume_id is not None,
            save_checkpoint_fn=save_checkpoint_fn,
        )
    )

    pprint.pprint(result)
    watch()


if __name__ == "__main__":
    main(get_args())
