import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter

import tianshou as ts


def main():
    task = "CartPole-v1"
    lr, epoch, batch_size = 1e-3, 10, 64
    train_num, test_num = 10, 100
    gamma, n_step, target_freq = 0.9, 3, 320
    buffer_size = 20000
    eps_train, eps_test = 0.1, 0.05
    step_per_epoch, step_per_collect = 10000, 10
    logger = ts.utils.TensorboardLogger(SummaryWriter("log/dqn"))  # TensorBoard is supported!
    # For other loggers: https://tianshou.readthedocs.io/en/master/tutorials/logger.html

    # you can also try with SubprocVectorEnv
    train_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(train_num)])
    test_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(test_num)])

    from tianshou.utils.net.common import Net

    # you can define other net by following the API:
    # https://tianshou.readthedocs.io/en/master/tutorials/dqn.html#build-the-network
    env = gym.make(task, render_mode="human")
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[128, 128, 128])
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    policy = ts.policy.DQNPolicy(
        model=net,
        optim=optim,
        discount_factor=gamma,
        action_space=env.action_space,
        estimation_step=n_step,
        target_update_freq=target_freq,
    )
    train_collector = ts.data.Collector(
        policy,
        train_envs,
        ts.data.VectorReplayBuffer(buffer_size, train_num),
        exploration_noise=True,
    )
    test_collector = ts.data.Collector(
        policy,
        test_envs,
        exploration_noise=True,
    )  # because DQN uses epsilon-greedy method

    result = ts.trainer.OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=epoch,
        step_per_epoch=step_per_epoch,
        step_per_collect=step_per_collect,
        episode_per_test=test_num,
        batch_size=batch_size,
        update_per_step=1 / step_per_collect,
        train_fn=lambda epoch, env_step: policy.set_eps(eps_train),
        test_fn=lambda epoch, env_step: policy.set_eps(eps_test),
        stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold,
        logger=logger,
    ).run()
    print(f"Finished training in {result.timing.total_time} seconds")

    # watch performance
    policy.eval()
    policy.set_eps(eps_test)
    collector = ts.data.Collector(policy, env, exploration_noise=True)
    collector.collect(n_episode=100, render=1 / 35)


if __name__ == "__main__":
    main()
