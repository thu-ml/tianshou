import os
import gym
import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.env import SubprocVectorEnv
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, ReplayBuffer
from tianshou.policy import SACPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="BipedalWalkerHardcore-v3")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=1000000)
    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--step-per-epoch', type=int, default=2400)
    parser.add_argument('--collect-per-step', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--layer-num', type=int, default=1)
    parser.add_argument('--training-num', type=int, default=8)
    parser.add_argument('--test-num', type=int, default=8)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--rew-norm', type=int, default=0)
    parser.add_argument('--ignore-done', type=int, default=0)
    parser.add_argument('--n-step', type=int, default=4)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


class EnvWrapper(object):
    """Env wrapper for reward scale, action repeat and action noise"""

    def __init__(self, task, action_repeat=3,
                 reward_scale=5, act_noise=0.3):
        self._env = gym.make(task)
        self.action_repeat = action_repeat
        self.reward_scale = reward_scale
        self.act_noise = act_noise

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        # add action noise
        action += self.act_noise * (-2 * np.random.random(4) + 1)
        r = 0.0
        for _ in range(self.action_repeat):
            obs_, reward_, done_, info_ = self._env.step(action)
            # remove done reward penalty
            if done_:
                break
            r = r + reward_
        # scale reward
        return obs_, self.reward_scale * r, done_, info_


def test_sac_bipedal(args=get_args()):
    torch.set_num_threads(1)  # we just need only one thread for NN

    env = EnvWrapper(args.task)

    def IsStop(reward):
        return reward >= env.spec.reward_threshold

    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]

    train_envs = SubprocVectorEnv(
        [lambda: EnvWrapper(args.task) for _ in range(args.training_num)])
    # test_envs = gym.make(args.task)
    test_envs = SubprocVectorEnv([lambda: EnvWrapper(args.task, reward_scale=1)
                                  for _ in range(args.test_num)])

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # model
    net_a = Net(args.layer_num, args.state_shape, device=args.device)
    actor = ActorProb(
        net_a, args.action_shape,
        args.max_action, args.device
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    net_c1 = Net(args.layer_num, args.state_shape,
                 args.action_shape, concat=True, device=args.device)
    critic1 = Critic(net_c1, args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)

    net_c2 = Net(args.layer_num, args.state_shape,
                 args.action_shape, concat=True, device=args.device)
    critic2 = Critic(net_c2, args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    policy = SACPolicy(
        actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim,
        args.tau, args.gamma, args.alpha,
        [env.action_space.low[0], env.action_space.high[0]],
        reward_normalization=args.rew_norm,
        ignore_done=args.ignore_done,
        estimation_step=args.n_step)

    # collector
    train_collector = Collector(
        policy, train_envs, ReplayBuffer(args.buffer_size))
    test_collector = Collector(policy, test_envs)
    # train_collector.collect(n_step=args.buffer_size)
    # log
    log_path = os.path.join(args.logdir, args.task, 'sac')
    writer = SummaryWriter(log_path)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    # trainer
    result = offpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.test_num,
        args.batch_size, stop_fn=IsStop, save_fn=save_fn, writer=writer)

    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        policy.eval()
        test_envs.seed(args.seed)
        test_collector.reset()
        result = test_collector.collect(n_episode=[1] * args.test_num,
                                        render=args.render)
        print(f'Final reward: {result["rew"]}, length: {result["len"]}')


if __name__ == '__main__':
    test_sac_bipedal()
