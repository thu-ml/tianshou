import argparse
import datetime
import os
import pickle
import pprint

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import DecisionTransformerConfig, DecisionTransformerModel

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import DTPolicy
from tianshou.trainer import OfflineTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import DecisionTransformer

if __name__ == "__main__":
    from gather_pendulum_data import expert_file_name, gather_data
else:  # pytest
    from test.offline.gather_pendulum_data import expert_file_name, gather_data


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Pendulum-v1')
    parser.add_argument('--reward-threshold', type=float, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--embed-size", type=int, default=128)
    parser.add_argument("--num-layer", type=int, default=3)
    parser.add_argument("--num-head", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-length", type=int, default=20)
    parser.add_argument("--target-return", type=float, default=50.0)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--step-per-epoch', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eval-freq", type=int, default=1)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=1 / 35)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument(
        '--watch',
        default=False,
        action='store_true',
        help='watch the play of pre-trained policy only',
    )
    parser.add_argument("--load-buffer-name", type=str, default=expert_file_name())
    args = parser.parse_known_args()[0]
    return args


def test_dt(args=get_args()):
    if os.path.exists(args.load_buffer_name) and os.path.isfile(args.load_buffer_name):
        if args.load_buffer_name.endswith(".hdf5"):
            buffer = VectorReplayBuffer.load_hdf5(args.load_buffer_name)
        else:
            buffer = pickle.load(open(args.load_buffer_name, "rb"))
    else:
        buffer = gather_data()
    env = gym.make(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]  # float
    if args.reward_threshold is None:
        # too low?
        default_reward_threshold = {"Pendulum-v0": -1200, "Pendulum-v1": -1200}
        args.reward_threshold = default_reward_threshold.get(
            args.task, env.spec.reward_threshold
        )

    args.state_dim = args.state_shape[0]
    args.action_dim = args.action_shape[0]
    # test_envs = gym.make(args.task)
    test_envs = DummyVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.test_num)]
    )
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    test_envs.seed(args.seed)

    # model
    # actor network
    config = DecisionTransformerConfig(
        batch_size=args.batch_size,
        act_dim=args.action_dim,
        state_dim=args.state_dim,
        n_layer=args.num_layer,
        n_head=args.num_head,
        n_embd=args.embed_size,
        max_length=args.max_length,
    )
    net = DecisionTransformerModel(config=config)
    actor = DecisionTransformer(
        net,
        args.state_shape,
        args.action_shape,
        max_length=args.max_length,
        target_return=args.target_return,
        device=args.device
    ).to(args.device)
    optim = torch.optim.Adam(actor.parameters(), lr=args.lr)

    policy = DTPolicy(
        actor,
        optim,
        action_space=env.action_space,
        device=args.device,
    )

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    # collector
    # buffer has been gathered
    # train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    # log
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'seed_{args.seed}_{t0}-{args.task.replace("-", "_")}_dt'
    log_path = os.path.join(args.logdir, args.task, 'dt', log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(mean_rewards):
        return mean_rewards >= args.reward_threshold

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
    trainer = OfflineTrainer(
        policy,
        buffer,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.test_num,
        args.batch_size,
        save_best_fn=save_best_fn,
        stop_fn=stop_fn,
        logger=logger,
    )

    for epoch, epoch_stat, info in trainer:
        print(f"Epoch: {epoch}")
        print(epoch_stat)
        print(info)

    assert stop_fn(info["best_reward"])

    # Let's watch its performance!
    if __name__ == "__main__":
        pprint.pprint(info)
        env = gym.make(args.task)
        policy.eval()
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


if __name__ == '__main__':
    test_dt()
