import sys
import time

import gymnasium as gym
import numpy as np
import tqdm

from tianshou.data import Batch, ReplayBuffer, VectorReplayBuffer


def test_replaybuffer(task="Pendulum-v1"):
    total_count = 5
    for _ in tqdm.trange(total_count, desc="ReplayBuffer"):
        env = gym.make(task)
        buf = ReplayBuffer(10000)
        obs, info = env.reset()
        for _ in range(100000):
            act = env.action_space.sample()
            obs_next, rew, terminated, truncated, info = env.step(act)
            done = terminated or truncated
            batch = Batch(
                obs=np.array([obs]),
                act=np.array([act]),
                rew=np.array([rew]),
                terminated=np.array([terminated]),
                truncated=np.array([truncated]),
                done=np.array([done]),
                obs_next=np.array([obs_next]),
                info=np.array([info]),
            )
            buf.add(batch, buffer_ids=[0])
            obs = obs_next
            if done:
                obs, info = env.reset()


def test_vectorbuffer(task="Pendulum-v1"):
    total_count = 5
    for _ in tqdm.trange(total_count, desc="VectorReplayBuffer"):
        env = gym.make(task)
        buf = VectorReplayBuffer(total_size=10000, buffer_num=1)
        obs, info = env.reset()
        for _ in range(100000):
            act = env.action_space.sample()
            obs_next, rew, terminated, truncated, info = env.step(act)
            done = terminated or truncated
            batch = Batch(
                obs=np.array([obs]),
                act=np.array([act]),
                rew=np.array([rew]),
                terminated=np.array([terminated]),
                truncated=np.array([truncated]),
                obs_next=np.array([obs_next]),
                info=np.array([info]),
            )
            buf.add(batch)
            obs = obs_next
            if done:
                obs, info = env.reset()


if __name__ == "__main__":
    t0 = time.time()
    test_replaybuffer(sys.argv[-1])
    print("test replaybuffer: ", time.time() - t0)
    t0 = time.time()
    test_vectorbuffer(sys.argv[-1])
    print("test vectorbuffer: ", time.time() - t0)
