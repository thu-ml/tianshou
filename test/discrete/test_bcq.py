from tianshou.policy import BCQPolicy
from tianshou.policy import BCQN
import os
import gym
import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import random
from tianshou.policy import DQNPolicy
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
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
    parser.add_argument("--tao", type=float, default=0.8)
    parser.add_argument("--gamma", type=float, default=0.9)
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

    model = BCQN(state_shape, action_shape, args.hidden_dim, args.hidden_dim)
    optim = torch.optim.Adam(model.parameters(), lr=0.5)
    policy = BCQPolicy(
        model,
        optim,
        args.tao,
        args.target_update_frequency,
        args.device,
        args.gamma,
        args.imitation_logits_penalty,
    )

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
    # TODOzjl save policy torch.save(policy.state_dict(), os.path.join(best_policy_save_dir, 'policy.pth'))

    # # batch = buffer.sample(1)
    # print(buffer.obs)
    # print(buffer.rew)
    # buffer.rew = torch.Tensor([1,2])
    # print(buffer.rew)

    # buffer.obs = torch.Tensor([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])
    # buffer.act = torch.Tensor([1, 2])
    # buffer.rew = torch.Tensor([10, 20])
    # buffer.done = torch.Tensor([False, True])
    # buffer.obs_next = torch.Tensor([[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]])

    # print(buffer.obs)
    # print(buffer)
    # buffer.add(
    #     obs=torch.Tensor([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]),
    #     act=torch.Tensor([1, 2]),
    #     rew=torch.Tensor([1.0, 2.0]),
    #     done=torch.Tensor([False, True]),
    #     obs_next=torch.Tensor([[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]]),
    # )
    # print(buffer)
    # print(buffer)
    # batch = buffer.sample(2)[0]
    # print(batch)
    # batch.to_torch()
    # print(batch)
    # batch = policy.forward(batch)
    # print(batch)

    # loss = policy.learn(batch)
    # print(loss)
    ###########################
    # best_reward = -1
    # best_policy = policy

    # global_iter = 0

    # total_iter = len(buffer) // args.batch_size
    # for epoch in range(1, 1 + args.epoch):
    #     for iter in range(total_iter):
    #         global_iter += 1
    #         loss = policy.update(args.batch_size, buffer)
    #         # batch = buffer.sample(args.batch_size)[0]
    #         # batch.to_torch()
    #         if global_iter % log_frequency == 0:
    #             writer.add_scalar(
    #                 "train/loss", loss['loss'], global_step=global_iter)
    #             test_collector = Collector(policy, test_envs)

    #             test_result = test_episode(
    #                 policy, test_collector, None,
    #                 epoch, args.episode_per_test, writer, global_iter)
    #             # for k in result.keys():
    #             #     writer.add_scalar(
    #             #         "train/" + k, result[k], global_step=env_step)

    #             if best_reward < result["rew"]:
    #                 best_reward = result["rew"]
    #                 best_policy = policy
    #             # epoch, args.episode_per_test, writer, env_step)
