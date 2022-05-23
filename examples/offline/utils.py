import d4rl
import gym
import h5py

from tianshou.data import ReplayBuffer


def load_buffer_d4rl(expert_data_task: str) -> ReplayBuffer:
    dataset = d4rl.qlearning_dataset(gym.make(expert_data_task))
    replay_buffer = ReplayBuffer.from_data(
        obs=dataset["observations"],
        act=dataset["actions"],
        rew=dataset["rewards"],
        done=dataset["terminals"],
        obs_next=dataset["next_observations"]
    )
    return replay_buffer


def load_buffer(buffer_path: str) -> ReplayBuffer:
    with h5py.File(buffer_path, "r") as dataset:
        buffer = ReplayBuffer.from_data(
            obs=dataset["observations"],
            act=dataset["actions"],
            rew=dataset["rewards"],
            done=dataset["terminals"],
            obs_next=dataset["next_observations"]
        )
    return buffer
