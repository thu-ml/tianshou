import argparse
import datetime
import os
import pprint

import numpy as np
import torch
from env import make_vizdoom_env
from network import DQN
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import ICMPolicy, PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.discrete import Actor, Critic, IntrinsicCuriosityModule


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="D1_basic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.00002)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--step-per-epoch", type=int, default=100000)
    parser.add_argument("--step-per-collect", type=int, default=1000)
    parser.add_argument("--repeat-per-collect", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=100)
    parser.add_argument("--rew-norm", type=int, default=False)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--lr-decay", type=int, default=True)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--eps-clip", type=float, default=0.2)
    parser.add_argument("--dual-clip", type=float, default=None)
    parser.add_argument("--value-clip", type=int, default=0)
    parser.add_argument("--norm-adv", type=int, default=1)
    parser.add_argument("--recompute-adv", type=int, default=0)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--frames-stack", type=int, default=4)
    parser.add_argument("--skip-num", type=int, default=4)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="vizdoom.benchmark")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    parser.add_argument(
        "--save-lmp",
        default=False,
        action="store_true",
        help="save lmp file for replay whole episode",
    )
    parser.add_argument("--save-buffer-name", type=str, default=None)
    parser.add_argument(
        "--icm-lr-scale",
        type=float,
        default=0.,
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


def test_ppo(args=get_args()):
    # make environments
    env, train_envs, test_envs = make_vizdoom_env(
        args.task, args.skip_num, (args.frames_stack, 84, 84), args.save_lmp,
        args.seed, args.training_num, args.test_num
    )
    args.state_shape = env.observation_space.shape
    args.action_shape = env.action_space.shape or env.action_space.n
    # should be N_FRAMES x H x W
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # define model
    net = DQN(
        *args.state_shape,
        args.action_shape,
        device=args.device,
        features_only=True,
        output_dim=args.hidden_size,
    )
    actor = Actor(net, args.action_shape, device=args.device, softmax_output=False)
    critic = Critic(net, device=args.device)
    optim = torch.optim.Adam(ActorCritic(actor, critic).parameters(), lr=args.lr)

    lr_scheduler = None
    if args.lr_decay:
        # decay learning rate to 0 linearly
        max_update_num = np.ceil(
            args.step_per_epoch / args.step_per_collect
        ) * args.epoch

        lr_scheduler = LambdaLR(
            optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num
        )

    # define policy
    def dist(p):
        return torch.distributions.Categorical(logits=p)

    policy = PPOPolicy(
        actor,
        critic,
        optim,
        dist,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        reward_normalization=args.rew_norm,
        action_scaling=False,
        lr_scheduler=lr_scheduler,
        action_space=env.action_space,
        eps_clip=args.eps_clip,
        value_clip=args.value_clip,
        dual_clip=args.dual_clip,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
    ).to(args.device)
    if args.icm_lr_scale > 0:
        feature_net = DQN(
            *args.state_shape,
            args.action_shape,
            device=args.device,
            features_only=True,
            output_dim=args.hidden_size,
        )
        action_dim = np.prod(args.action_shape)
        feature_dim = feature_net.output_dim
        icm_net = IntrinsicCuriosityModule(
            feature_net.net, feature_dim, action_dim, device=args.device
        )
        icm_optim = torch.optim.Adam(icm_net.parameters(), lr=args.lr)
        policy = ICMPolicy(
            policy, icm_net, icm_optim, args.icm_lr_scale, args.icm_reward_scale,
            args.icm_forward_loss_weight
        ).to(args.device)
    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
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
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "ppo_icm" if args.icm_lr_scale > 0 else "ppo"
    log_name = os.path.join(args.task, args.algo_name, str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)

    # logger
    if args.logger == "wandb":
        logger = WandbLogger(
            save_interval=1,
            name=log_name.replace(os.path.sep, "__"),
            run_id=args.resume_id,
            config=args,
            project=args.wandb_project,
        )
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    if args.logger == "tensorboard":
        logger = TensorboardLogger(writer)
    else:  # wandb
        logger.load(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards):
        if env.spec.reward_threshold:
            return mean_rewards >= env.spec.reward_threshold
        else:
            return False

    # watch agent's performance
    def watch():
        print("Setup test envs ...")
        policy.eval()
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
            collector = Collector(policy, test_envs, buffer, exploration_noise=True)
            result = collector.collect(n_step=args.buffer_size)
            print(f"Save buffer into {args.save_buffer_name}")
            # Unfortunately, pickle will cause oom with 1M buffer size
            buffer.save_hdf5(args.save_buffer_name)
        else:
            print("Testing agent ...")
            test_collector.reset()
            result = test_collector.collect(
                n_episode=args.test_num, render=args.render
            )
        rew = result["rews"].mean()
        lens = result["lens"].mean() * args.skip_num
        print(f'Mean reward (over {result["n/ep"]} episodes): {rew}')
        print(f'Mean length (over {result["n/ep"]} episodes): {lens}')

    if args.watch:
        watch()
        exit(0)

    # test train_collector and start filling replay buffer
    train_collector.collect(n_step=args.batch_size * args.training_num)
    # trainer
    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.repeat_per_collect,
        args.test_num,
        args.batch_size,
        step_per_collect=args.step_per_collect,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        test_in_train=False,
    )

    pprint.pprint(result)
    watch()


if __name__ == "__main__":
    test_ppo(get_args())
