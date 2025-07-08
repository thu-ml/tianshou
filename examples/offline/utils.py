import d4rl
import gymnasium as gym
import h5py
import numpy as np
import minari

from tianshou.data import ReplayBuffer, Batch
from tianshou.utils import RunningMeanStd


def load_buffer_minari(dataset_name: str) -> ReplayBuffer:
    dataset = minari.load_dataset(dataset_name)
    total_size = dataset.total_steps

    obs = np.empty((total_size, *dataset.observation_space.shape), dtype=dataset.observation_space.dtype)
    act = np.empty((total_size, *dataset.action_space.shape), dtype=dataset.action_space.dtype)
    rew = np.empty((total_size,))
    terminated = np.empty((total_size,))
    truncated = np.empty((total_size,))
    obs_next = np.empty((total_size, *dataset.observation_space.shape))

    current_index = 0
    for episode in dataset:
        next_index = current_index + len(episode.actions)
        obs[current_index:next_index] = episode.observations[:-1]
        act[current_index:next_index] = episode.actions
        rew[current_index:next_index] = episode.rewards
        obs_next[current_index:next_index] = episode.observations[1:]
        terminated[current_index:next_index] = episode.terminations
        truncated[current_index:next_index] = episode.truncations
        current_index = next_index

    done_all = np.logical_or(terminated, truncated)
    replay_buffer = ReplayBuffer(size=total_size)
    data_batch = Batch(
        obs=obs,
        act=act,
        rew=rew,
        terminated=terminated,
        truncated=truncated,
        done=done_all,
        obs_next=obs_next,
    )

    replay_buffer.set_batch(data_batch)
    replay_buffer._size = total_size
    replay_buffer._insertion_idx = 0
    replay_buffer.last_index[0] = (total_size - 1) % total_size
    return replay_buffer


def load_buffer_d4rl(expert_data_task: str) -> ReplayBuffer:
    dataset = d4rl.qlearning_dataset(gym.make(expert_data_task))
    return ReplayBuffer.from_data(
        obs=dataset["observations"],
        act=dataset["actions"],
        rew=dataset["rewards"],
        done=dataset["terminals"],
        obs_next=dataset["next_observations"],
        terminated=dataset["terminals"],
        truncated=np.zeros(len(dataset["terminals"])),
    )


def load_buffer(buffer_path: str) -> ReplayBuffer:
    with h5py.File(buffer_path, "r") as dataset:
        return ReplayBuffer.from_data(
            obs=dataset["observations"],
            act=dataset["actions"],
            rew=dataset["rewards"],
            done=dataset["terminals"],
            obs_next=dataset["next_observations"],
            terminated=dataset["terminals"],
            truncated=np.zeros(len(dataset["terminals"])),
        )


def normalize_all_obs_in_replay_buffer(
    replay_buffer: ReplayBuffer,
) -> tuple[ReplayBuffer, RunningMeanStd]:
    # compute obs mean and var
    obs_rms = RunningMeanStd()
    obs_rms.update(replay_buffer.obs)
    _eps = np.finfo(np.float32).eps.item()
    # normalize obs
    replay_buffer._meta["obs"] = (replay_buffer.obs - obs_rms.mean) / np.sqrt(obs_rms.var + _eps)
    replay_buffer._meta["obs_next"] = (replay_buffer.obs_next - obs_rms.mean) / np.sqrt(
        obs_rms.var + _eps,
    )
    return replay_buffer, obs_rms
