import free_mjc
import gym
import torch
import pprint
import argparse
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import datetime

from tianshou.policy import TD3Policy
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.exploration import GaussianNoise
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, ReplayBuffer
from tianshou.utils.net.continuous import Actor, Critic


from typing import Dict, List, Union, Optional, Callable
import time
import warnings
from numbers import Number
from typing import Dict, List, Union, Optional, Callable

from tianshou.policy import BasePolicy
from tianshou.exploration import BaseNoise
from tianshou.data.batch import _create_value
from tianshou.env import BaseVectorEnv, DummyVectorEnv
from tianshou.data import Batch, ReplayBuffer, ListReplayBuffer, to_numpy


class step_Collector:
    def __init__(self, policy, env, buffer, seed):
        self.buffer = buffer
        self.policy = policy
        self.env = env
        self.env.seed(seed)
        self.buffer.reset()
        self.data = Batch()
        self.data.done = True
        self.done = True
        self.rew_keep = 0
        self.len_keep = 0
        self.rew = 0
        self.len = 0
    def collect(self, n_step = 1, random = False):
        for i in range(n_step):
            if self.done == True:
                self.data.obs, self.done = np.expand_dims(self.env.reset(), axis=0), False
                self.rew_keep = self.rew
                self.len_keep = self.len
                self.rew = 0
                self.len = 0
            if not random:
                with torch.no_grad():
                    # from IPython import embed;embed()
                    self.data.update(self.policy(self.data))#h act
                    self.data.act = to_numpy(self.data.act)
            else:
                self.data.update(act = np.expand_dims(self.env.action_space.sample(), axis=0))
            obs_next, rew, self.done, info = self.env.step(to_numpy(self.data.act[0]))
            self.rew += rew
            self.len+=1
            self.data.update(obs_next=np.expand_dims(obs_next, axis=0), rew=rew,
                                done= self.done if self.len < self.env._max_episode_steps else False, info=info)
            # if not self.len < self.env._max_episode_steps:
            #     from IPython import embed;embed()
            self.buffer.add(**self.data)
            self.data.obs = self.data.obs_next

        return {
            "n/ep": 0,
            "n/st": 1,
            "v/st": 0,
            "v/ep": 0,
            "rew": self.rew_keep,
            "rew_std": 0,
            "len": self.len_keep,
        }
    def reset_stat(self):
        return

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Ant-v2')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=1000000)
    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=1.5e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--exploration-noise', type=float, default=0.1)
    parser.add_argument('--policy-noise', type=float, default=0.2)
    parser.add_argument('--noise-clip', type=float, default=0.5)
    parser.add_argument('--update-actor-freq', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=400)
    parser.add_argument('--step-per-epoch', type=int, default=5000)
    parser.add_argument('--collect-per-step', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--layer-num', type=int, default=1)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--log-interval', type=int, default=1000)
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--hidden-layer-size', type=int, default=256)
    parser.add_argument("--start-timesteps", type=int, default=25000)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def test_td3(args=get_args()):
    env = gym.make(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    # train_envs = gym.make(args.task)
    train_envs = DummyVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.training_num)])
    # test_envs = gym.make(args.task)
    test_envs = DummyVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net = Net(args.layer_num, args.state_shape, hidden_layer_size=args.hidden_layer_size, device=args.device)
    actor = Actor(
        net, args.action_shape,
        args.max_action, args.device,
        hidden_layer_size=args.hidden_layer_size
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net = Net(args.layer_num, args.state_shape,
              args.action_shape, concat=True, hidden_layer_size=args.hidden_layer_size, device=args.device)
    critic1 = Critic(net, args.device, hidden_layer_size=args.hidden_layer_size).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(net, args.device, hidden_layer_size=args.hidden_layer_size).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)
    policy = TD3Policy(
        actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim,
        action_range=[env.action_space.low[0], env.action_space.high[0]],
        tau=args.tau, gamma=args.gamma,
        exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        policy_noise=args.policy_noise,
        update_actor_freq=args.update_actor_freq,
        noise_clip=args.noise_clip,
        reward_normalization=False, ignore_done=False)
    # collector
    # train_collector = Collector(
    #     policy, train_envs, ReplayBuffer(args.buffer_size))

    train_collector = step_Collector(
    policy, gym.make(args.task), ReplayBuffer(args.buffer_size), args.seed)

    test_collector = Collector(policy, test_envs)
    #args.start_timesteps = int(args.start_timesteps)
    train_collector.collect(n_step=args.start_timesteps, random = True)
    # log
    log_path = os.path.join(args.logdir, args.task, 'td3', 'seed_' + str(args.seed) + '_' + datetime.datetime.now().strftime('%m%d-%H%M%S'))
    writer = SummaryWriter(log_path)

    def stop_fn(mean_rewards):
        return mean_rewards >= env.spec.reward_threshold

    # trainer
    result = offpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.test_num,
        args.batch_size, stop_fn=stop_fn, writer=writer, log_interval = args.log_interval)
    assert stop_fn(result['best_reward'])
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
    test_td3()
