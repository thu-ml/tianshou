import gym
import torch
import pprint
import argparse
import datetime
import numpy as np
from copy import deepcopy

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import PPOPolicy
from tianshou.env import SubprocVectorEnv
from tianshou.trainer import imitation_trainer
from tianshou.data import Collector, ReplayBuffer

if __name__ == '__main__':
    from .net import Net, Actor, Critic
else:  # pytest
    from test.discrete.net import Net, Actor, Critic


def get_args():
    parser = argparse.ArgumentParser()
    # spec
    parser.add_argument('--note', type=str, default=None)
    parser.add_argument('--peer', type=float, default=0.)
    parser.add_argument('--peer-decay-steps', type=int, default=0)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--reward-threshold', type=float, default=None)
    #
    parser.add_argument('--task', type=str, default='Acrobot-v1')
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--collect-per-step', type=int, default=8)
    parser.add_argument('--repeat-per-collect', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--layer-num', type=int, default=1)
    parser.add_argument('--training-num', type=int, default=8)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--device', type=str, default='cpu')
    # ppo special
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    args = parser.parse_args()
    args.note = args.note or \
                datetime.datetime.now().strftime('%y%m%d%H%M%S')
    return args


def test_ppo(args=get_args()):
    env = gym.make(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # train_envs = gym.make(args.task)
    train_envs = SubprocVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.training_num)])
    # test_envs = gym.make(args.task)
    test_envs = SubprocVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # log
    writer = SummaryWriter(
        f'{args.logdir}/imitation/{args.task}/imitation/{args.note}')
    # model
    net = Net(args.layer_num, args.state_shape, device=args.device)
    actor = Actor(net, args.action_shape).to(args.device)
    critic = Critic(net).to(args.device)
    optim = torch.optim.Adam(list(
        actor.parameters()) + list(critic.parameters()), lr=args.lr)
    dist = torch.distributions.Categorical
    policy = PPOPolicy(
        actor, critic, optim, dist, args.gamma,
        max_grad_norm=args.max_grad_norm,
        eps_clip=args.eps_clip,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        action_range=None)

    # Load expert model.
    assert args.load is not None, 'args.load should not be None'
    expert = deepcopy(policy)
    expert.load_state_dict(torch.load(
        f'{args.logdir}/{args.task}/ppo/{args.load}/policy.pth'))
    expert.eval()

    # collector
    expert_collector = Collector(
        expert, train_envs, ReplayBuffer(args.buffer_size))
    test_collector = Collector(policy, test_envs)

    def stop_fn(x):
        return x >= (args.reward_threshold or env.spec.reward_threshold)

    def learner(pol, batch, batch_size, repeat, peer=0.):
        losses, ent_losses = [], []
        for _ in range(repeat):
            for b in batch.split(batch_size):
                logits = pol(b).logits
                demo = torch.tensor(b.act, dtype=torch.long)
                loss = F.cross_entropy(logits, demo)
                if peer != 0:
                    peer_demo = demo[torch.randperm(len(demo))]
                    loss -= peer * F.cross_entropy(logits, peer_demo)
                pol.optim.zero_grad()
                loss.backward()
                pol.optim.step()
                losses.append(loss.detach().cpu().numpy())
        return {
            'loss': losses,
            'loss/ent': ent_losses,
            'peer': peer,
        }

    # trainer
    result = imitation_trainer(
        policy, learner, expert_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.repeat_per_collect,
        args.test_num, args.batch_size, stop_fn=stop_fn, writer=writer,
        task=args.task, peer=args.peer, peer_decay_steps=args.peer_decay_steps)
    assert stop_fn(result['best_reward'])
    expert_collector.close()
    test_collector.close()

    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        env = gym.make(args.task)
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        print(f'Final reward: {result["rew"]}, length: {result["len"]}')
        collector.close()


if __name__ == '__main__':
    test_ppo()
