import gymnasium as gym

from tianshou.env import DummyVectorEnv, SubprocVectorEnv, ShmemVectorEnv


def test_gym_env_action_space():
    env = gym.make("Pendulum-v1")
    env.action_space.seed(0)
    action1 = env.action_space.sample()

    env.action_space.seed(0)
    action2 = env.action_space.sample()

    assert action1 == action2


def test_dummy_vec_env_action_space():
    num_envs = 4
    envs = DummyVectorEnv([lambda: gym.make("Pendulum-v1") for _ in range(num_envs)])
    envs.seed(0)
    action1 = [ac_space.sample() for ac_space in envs.action_space]

    envs.seed(0)
    action2 = [ac_space.sample() for ac_space in envs.action_space]

    assert action1 == action2


def test_subproc_vec_env_action_space():
    num_envs = 4
    envs = SubprocVectorEnv([lambda: gym.make("Pendulum-v1") for _ in range(num_envs)])
    envs.seed(0)
    action1 = [ac_space.sample() for ac_space in envs.action_space]

    envs.seed(0)
    action2 = [ac_space.sample() for ac_space in envs.action_space]

    assert action1 == action2


def test_shmem_vec_env_action_space():
    num_envs = 4
    envs = ShmemVectorEnv([lambda: gym.make("Pendulum-v1") for _ in range(num_envs)])
    envs.seed(0)
    action1 = [ac_space.sample() for ac_space in envs.action_space]

    envs.seed(0)
    action2 = [ac_space.sample() for ac_space in envs.action_space]

    assert action1 == action2


if __name__ == "__main__":
    test_gym_env_action_space()
    test_dummy_vec_env_action_space()
    test_subproc_vec_env_action_space()
    test_shmem_vec_env_action_space()
