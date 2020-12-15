from tianshou.policy import BCQPolicy
import os
import gym
import torch
import pprint
import argparse
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tianshou.env import DummyVectorEnv
from tianshou.trainer import offline_trainer
from tianshou.data import Collector, ReplayBuffer, PrioritizedReplayBuffer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="CartPole-v0")
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=1)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--buffer-size", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=5)
    parser.add_argument("--test-frequency", type=int, default=5)
    parser.add_argument("--target-update-frequency", type=int, default=5)
    parser.add_argument("--episode-per-test", type=int, default=5)
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--imitation_logits_penalty", type=float, default=0.1)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_known_args()[0]
    return args


def test_bcq(args=get_args()):
    env = gym.make(args.task)
    state_shape = env.observation_space.shape or env.observation_space.n
    state_shape = state_shape[0]
    action_shape = env.action_space.shape or env.action_space.n
    model = BCQPolicy.BCQN(state_shape, action_shape, args.hidden_dim, args.hidden_dim)
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    policy = BCQPolicy(
        model,
        optim,
        args.tau,
        args.target_update_frequency,
        args.device,
        args.gamma,
        args.imitation_logits_penalty,
    )

    # Make up some dummy training data in replay buffer
    buffer = ReplayBuffer(size=args.buffer_size)
    for i in range(args.buffer_size):
        buffer.add(
            obs=torch.rand(state_shape),
            act=random.randint(0, action_shape - 1),
            rew=1,
            done=False,
            obs_next=torch.rand(state_shape),
            info={},
        )

    test_envs = DummyVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.test_num)]
    )

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    test_envs.seed(args.seed)

    test_collector = Collector(policy, test_envs)

    log_path = os.path.join(args.logdir, "writer")
    writer = SummaryWriter(log_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    res = offline_trainer(
        policy,
        buffer,
        test_collector,
        args.epoch,
        args.batch_size,
        args.episode_per_test,
        writer,
        args.test_frequency,
    )
    print("final best_reward", res["best_reward"])


if __name__ == "__main__":
    test_bcq(get_args())
