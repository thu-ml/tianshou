import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

import tianshou as ts
from tianshou.data import CollectStats
from tianshou.policy.modelfree.dqn import DiscreteQLearningPolicy
from tianshou.policy.optim import AdamOptimizerFactory
from tianshou.trainer.base import OffPolicyTrainerParams
from tianshou.utils.space_info import SpaceInfo


def main() -> None:
    task = "CartPole-v1"
    lr, epoch, batch_size = 1e-3, 10, 64
    train_num, test_num = 10, 100
    gamma, n_step, target_freq = 0.9, 3, 320
    buffer_size = 20000
    eps_train, eps_test = 0.1, 0.05
    step_per_epoch, step_per_collect = 10000, 10

    logger = ts.utils.TensorboardLogger(SummaryWriter("log/dqn"))  # TensorBoard is supported!
    # For other loggers, see https://tianshou.readthedocs.io/en/master/tutorials/logger.html

    # You can also try SubprocVectorEnv, which will use parallelization
    train_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(train_num)])
    test_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(test_num)])

    from tianshou.utils.net.common import MLPActor

    # Note: You can easily define other networks.
    # See https://tianshou.readthedocs.io/en/master/01_tutorials/00_dqn.html#build-the-network
    env = gym.make(task, render_mode="human")
    assert isinstance(env.action_space, gym.spaces.Discrete)
    space_info = SpaceInfo.from_env(env)
    state_shape = space_info.observation_info.obs_shape
    action_shape = space_info.action_info.action_shape
    net = MLPActor(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[128, 128, 128])
    optim = AdamOptimizerFactory(lr=lr)

    policy = DiscreteQLearningPolicy(
        model=net, action_space=env.action_space, eps_training=eps_train, eps_inference=eps_test
    )
    algorithm: ts.policy.DQN = ts.policy.DQN(
        policy=policy,
        optim=optim,
        gamma=gamma,
        estimation_step=n_step,
        target_update_freq=target_freq,
    )
    train_collector = ts.data.Collector[CollectStats](
        algorithm,
        train_envs,
        ts.data.VectorReplayBuffer(buffer_size, train_num),
        exploration_noise=True,
    )
    test_collector = ts.data.Collector[CollectStats](
        algorithm,
        test_envs,
        exploration_noise=True,
    )  # because DQN uses epsilon-greedy method

    def stop_fn(mean_rewards: float) -> bool:
        if env.spec:
            if not env.spec.reward_threshold:
                return False
            else:
                return mean_rewards >= env.spec.reward_threshold
        return False

    result = algorithm.run_training(
        OffPolicyTrainerParams(
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=epoch,
            step_per_epoch=step_per_epoch,
            step_per_collect=step_per_collect,
            episode_per_test=test_num,
            batch_size=batch_size,
            update_per_step=1 / step_per_collect,
            stop_fn=stop_fn,
            logger=logger,
            test_in_train=True,
        )
    )
    print(f"Finished training in {result.timing.total_time} seconds")

    # watch performance
    collector = ts.data.Collector[CollectStats](algorithm, env, exploration_noise=True)
    collector.collect(n_episode=100, render=1 / 35)


if __name__ == "__main__":
    main()
