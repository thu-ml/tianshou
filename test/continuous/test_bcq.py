import argparse
import datetime
import os
import pprint

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import BCQPolicy, SACPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import BasicLogger, TensorboardLogger
from tianshou.utils.net.common import MLP, Net
from tianshou.utils.net.continuous import ActorProb, Critic, VAE, Perturbation


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Pendulum-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer_size', type=int, default=20000)
    parser.add_argument('--sac_hidden_sizes', type=int, nargs='*', default=[128, 128])
    parser.add_argument('--hidden_sizes', type=int, nargs='*', default=[200, 150])
    parser.add_argument('--actor_lr', type=float, default=1e-3)
    parser.add_argument('--critic_lr', type=float, default=1e-3)
    parser.add_argument("--start_timesteps", type=int, default=50000)
    parser.add_argument('--epoch', type=int, default=7)
    parser.add_argument('--step_per_epoch', type=int, default=8000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--training_num', type=int, default=10)
    parser.add_argument('--test_num', type=int, default=10)
    parser.add_argument('--step_per_collect', type=int, default=10)
    parser.add_argument('--update_per_step', type=float, default=0.125)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)

    parser.add_argument("--vae_hidden_sizes", type=int, nargs='*', default=[375, 375])
    # default to 2 * action_dim
    parser.add_argument('--latent_dim', type=int)
    parser.add_argument("--gamma", default=0.99)
    parser.add_argument("--tau", default=0.005)
    # Weighting for Clipped Double Q-learning in BCQ
    parser.add_argument("--lmbda", default=0.75)
    # Max perturbation hyper-parameter for BCQ
    parser.add_argument("--phi", default=0.05)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument(
        '--watch',
        default=False,
        action='store_true',
        help='watch the play of pre-trained policy only'
    )
    # sac:
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--auto_alpha', type=int, default=1)
    parser.add_argument('--alpha_lr', type=float, default=3e-4)
    parser.add_argument('--rew_norm', action="store_true", default=False)
    parser.add_argument('--n_step', type=int, default=3)
    args = parser.parse_known_args()[0]
    return args


def gather_data():
    args = get_args()
    env = gym.make(args.task)
    if args.task == 'Pendulum-v0':
        env.spec.reward_threshold = -250
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    # you can also use tianshou.env.SubprocVectorEnv
    # train_envs = gym.make(args.task)
    train_envs = DummyVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.training_num)]
    )
    # test_envs = gym.make(args.task)
    test_envs = DummyVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.test_num)]
    )
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net = Net(args.state_shape, hidden_sizes=args.sac_hidden_sizes, device=args.device)
    actor = ActorProb(
        net,
        args.action_shape,
        max_action=args.max_action,
        device=args.device,
        unbounded=True
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c1 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.sac_hidden_sizes,
        concat=True,
        device=args.device
    )
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    net_c2 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.sac_hidden_sizes,
        concat=True,
        device=args.device
    )
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    policy = SACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        reward_normalization=args.rew_norm,
        estimation_step=args.n_step,
        action_space=env.action_space
    )
    # collector
    buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    # train_collector.collect(n_step=args.buffer_size)
    # log
    log_path = os.path.join(args.logdir, args.task, 'sac')
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(mean_rewards):
        return mean_rewards >= env.spec.reward_threshold

    # trainer
    offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        update_per_step=args.update_per_step,
        save_fn=save_fn,
        stop_fn=stop_fn,
        logger=logger
    )
    train_collector.reset()
    return train_collector


def test_bcq():
    train_collector = gather_data()
    args = get_args()
    env = gym.make(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]  # float
    if args.task == 'Pendulum-v0':
        env.spec.reward_threshold = -800

    args.state_dim = args.state_shape[0]
    args.action_dim = args.action_shape[0]
    # train_envs = gym.make(args.task)
    if args.training_num > 1:
        train_envs = SubprocVectorEnv(
            [lambda: gym.make(args.task) for _ in range(args.training_num)]
        )
    else:
        train_envs = gym.make(args.task)
    # test_envs = gym.make(args.task)
    test_envs = SubprocVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.test_num)]
    )
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # model
    # perturbation network
    net_a = MLP(
        input_dim=args.state_dim + args.action_dim,
        output_dim=args.action_dim,
        hidden_sizes=args.hidden_sizes,
        device=args.device
    )
    actor = Perturbation(
        net_a, max_action=args.max_action, device=args.device, phi=args.phi
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    net_c1 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device
    )
    net_c2 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device
    )
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    # vae
    # output_dim = 0, so the last Module in the encoder is ReLU
    vae_encoder = MLP(
        input_dim=args.state_dim + args.action_dim,
        hidden_sizes=args.vae_hidden_sizes,
        device=args.device
    )
    if not args.latent_dim:
        args.latent_dim = args.action_dim * 2
    vae_decoder = MLP(
        input_dim=args.state_dim + args.latent_dim,
        output_dim=args.action_dim,
        hidden_sizes=args.vae_hidden_sizes,
        device=args.device
    )
    vae = VAE(
        vae_encoder,
        vae_decoder,
        hidden_dim=args.vae_hidden_sizes[-1],
        latent_dim=args.latent_dim,
        max_action=args.max_action,
        device=args.device
    ).to(args.device)
    vae_optim = torch.optim.Adam(vae.parameters())

    policy = BCQPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        vae,
        vae_optim,
        device=args.device,
        gamma=args.gamma,
        tau=args.tau,
        lmbda=args.lmbda
    )

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    # collector
    # buffer has been gathered
    # train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    # train_collector.collect(n_step=args.start_timesteps, random=True)
    # log
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'seed_{args.seed}_{t0}-{args.task.replace("-", "_")}_bcq'
    log_path = os.path.join(args.logdir, args.task, 'bcq', log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = BasicLogger(writer)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(mean_rewards):
        return mean_rewards >= env.spec.reward_threshold

    def watch():
        policy.load_state_dict(
            torch.load(
                os.path.join(log_path, 'policy.pth'), map_location=torch.device('cpu')
            )
        )
        policy.eval()
        collector = Collector(policy, env)
        collector.collect(n_episode=1, render=1 / 35)

    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        save_fn=save_fn,
        stop_fn=stop_fn,
        logger=logger,
        update_per_step=args.update_per_step,
        test_in_train=False
    )
    assert stop_fn(result['best_reward'])

    # Let's watch its performance!
    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        env = gym.make(args.task)
        policy.eval()
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


if __name__ == '__main__':
    test_bcq()
