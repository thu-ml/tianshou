import argparse
import os
import pickle
import pprint

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import DiscreteCQLPolicy
from tianshou.trainer import offline_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net

if __name__ == "__main__":
    from gather_cartpole_data import expert_file_name, gather_data
else:  # pytest
    from test.offline.gather_cartpole_data import expert_file_name, gather_data


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="CartPole-v0")
    parser.add_argument("--reward-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--eps-test", type=float, default=0.001)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument('--num-quantiles', type=int, default=200)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=500)
    parser.add_argument("--min-q-weight", type=float, default=10.)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--update-per-epoch", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[64])
    parser.add_argument("--test-num", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.)
    parser.add_argument("--load-buffer-name", type=str, default=expert_file_name())
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_known_args()[0]
    return args


def test_discrete_cql(args=get_args()):
    # envs
    env = gym.make(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    if args.reward_threshold is None:
        default_reward_threshold = {"CartPole-v0": 170}
        args.reward_threshold = default_reward_threshold.get(
            args.task, env.spec.reward_threshold
        )
    test_envs = DummyVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.test_num)]
    )
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
        softmax=False,
        num_atoms=args.num_quantiles
    )
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    policy = DiscreteCQLPolicy(
        net,
        optim,
        args.gamma,
        args.num_quantiles,
        args.n_step,
        args.target_update_freq,
        min_q_weight=args.min_q_weight
    ).to(args.device)
    # buffer
    if os.path.exists(args.load_buffer_name) and os.path.isfile(args.load_buffer_name):
        if args.load_buffer_name.endswith(".hdf5"):
            buffer = VectorReplayBuffer.load_hdf5(args.load_buffer_name)
        else:
            buffer = pickle.load(open(args.load_buffer_name, "rb"))
    else:
        buffer = gather_data()

    # collector
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    log_path = os.path.join(args.logdir, args.task, 'discrete_cql')
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(mean_rewards):
        return mean_rewards >= args.reward_threshold

    result = offline_trainer(
        policy,
        buffer,
        test_collector,
        args.epoch,
        args.update_per_epoch,
        args.test_num,
        args.batch_size,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger
    )

    assert stop_fn(result['best_reward'])

    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        env = gym.make(args.task)
        policy.eval()
        policy.set_eps(args.eps_test)
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


if __name__ == "__main__":
    test_discrete_cql(get_args())
