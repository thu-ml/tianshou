import time
import torch
import numpy as np
from copy import deepcopy

from tianshou.env import BaseVectorEnv
from tianshou.data import Batch, ReplayBuffer
from tianshou.utils import MovAvg


class Collector(object):
    """docstring for Collector"""

    def __init__(self, policy, env, buffer=None, stat_size=100):
        super().__init__()
        self.env = env
        self.env_num = 1
        self.collect_step = 0
        self.collect_episode = 0
        self.collect_time = 0
        if buffer is None:
            self.buffer = ReplayBuffer(20000)
        else:
            self.buffer = buffer
        self.policy = policy
        self.process_fn = policy.process_fn
        self._multi_env = isinstance(env, BaseVectorEnv)
        self._multi_buf = False  # True if buf is a list
        # need multiple cache buffers only if storing in one buffer
        self._cached_buf = []
        if self._multi_env:
            self.env_num = len(env)
            if isinstance(self.buffer, list):
                assert len(self.buffer) == self.env_num, \
                    'The number of data buffer does not match the number of ' \
                    'input env.'
                self._multi_buf = True
            elif isinstance(self.buffer, ReplayBuffer):
                self._cached_buf = [
                    deepcopy(self.buffer) for _ in range(self.env_num)]
            else:
                raise TypeError('The buffer in data collector is invalid!')
        self.reset_env()
        self.reset_buffer()
        # state over batch is either a list, an np.ndarray, or a torch.Tensor
        self.state = None
        self.step_speed = MovAvg(stat_size)
        self.episode_speed = MovAvg(stat_size)

    def reset_buffer(self):
        if self._multi_buf:
            for b in self.buffer:
                b.reset()
        else:
            self.buffer.reset()

    def get_env_num(self):
        return self.env_num

    def reset_env(self):
        self._obs = self.env.reset()
        self._act = self._rew = self._done = self._info = None
        if self._multi_env:
            self.reward = np.zeros(self.env_num)
            self.length = np.zeros(self.env_num)
        else:
            self.reward, self.length = 0, 0
        for b in self._cached_buf:
            b.reset()

    def seed(self, seed=None):
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)

    def render(self, **kwargs):
        if hasattr(self.env, 'render'):
            return self.env.render(**kwargs)

    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()

    def _make_batch(self, data):
        if isinstance(data, np.ndarray):
            return data[None]
        else:
            return np.array([data])

    def collect(self, n_step=0, n_episode=0, render=0):
        if not self._multi_env:
            n_episode = np.sum(n_episode)
        start_time = time.time()
        assert sum([(n_step != 0), (n_episode != 0)]) == 1, \
            "One and only one collection number specification permitted!"
        cur_step = 0
        cur_episode = np.zeros(self.env_num) if self._multi_env else 0
        reward_sum = 0
        length_sum = 0
        while True:
            if self._multi_env:
                batch_data = Batch(
                    obs=self._obs, act=self._act, rew=self._rew,
                    done=self._done, obs_next=None, info=self._info)
            else:
                batch_data = Batch(
                    obs=self._make_batch(self._obs),
                    act=self._make_batch(self._act),
                    rew=self._make_batch(self._rew),
                    done=self._make_batch(self._done),
                    obs_next=None,
                    info=self._make_batch(self._info))
            result = self.policy(batch_data, self.state)
            self.state = result.state if hasattr(result, 'state') else None
            if isinstance(result.act, torch.Tensor):
                self._act = result.act.detach().cpu().numpy()
            elif not isinstance(self._act, np.ndarray):
                self._act = np.array(result.act)
            else:
                self._act = result.act
            obs_next, self._rew, self._done, self._info = self.env.step(
                self._act if self._multi_env else self._act[0])
            if render > 0:
                self.env.render()
                time.sleep(render)
            self.length += 1
            self.reward += self._rew
            if self._multi_env:
                for i in range(self.env_num):
                    data = {
                        'obs': self._obs[i], 'act': self._act[i],
                        'rew': self._rew[i], 'done': self._done[i],
                        'obs_next': obs_next[i], 'info': self._info[i]}
                    if self._cached_buf:
                        self._cached_buf[i].add(**data)
                    elif self._multi_buf:
                        self.buffer[i].add(**data)
                        cur_step += 1
                    else:
                        self.buffer.add(**data)
                        cur_step += 1
                    if self._done[i]:
                        if n_step != 0 or np.isscalar(n_episode) or \
                                cur_episode[i] < n_episode[i]:
                            cur_episode[i] += 1
                            reward_sum += self.reward[i]
                            length_sum += self.length[i]
                            if self._cached_buf:
                                cur_step += len(self._cached_buf[i])
                                self.buffer.update(self._cached_buf[i])
                        self.reward[i], self.length[i] = 0, 0
                        if self._cached_buf:
                            self._cached_buf[i].reset()
                        if isinstance(self.state, list):
                            self.state[i] = None
                        elif self.state is not None:
                            if isinstance(self.state[i], dict):
                                self.state[i] = {}
                            else:
                                self.state[i] = self.state[i] * 0
                            if isinstance(self.state, torch.Tensor):
                                # remove ref count in pytorch (?)
                                self.state = self.state.detach()
                if sum(self._done):
                    obs_next = self.env.reset(np.where(self._done)[0])
                if n_episode != 0:
                    if isinstance(n_episode, list) and \
                            (cur_episode >= np.array(n_episode)).all() or \
                            np.isscalar(n_episode) and \
                            cur_episode.sum() >= n_episode:
                        break
            else:
                self.buffer.add(
                    self._obs, self._act[0], self._rew,
                    self._done, obs_next, self._info)
                cur_step += 1
                if self._done:
                    cur_episode += 1
                    reward_sum += self.reward
                    length_sum += self.length
                    self.reward, self.length = 0, 0
                    self.state = None
                    obs_next = self.env.reset()
                if n_episode != 0 and cur_episode >= n_episode:
                    break
            if n_step != 0 and cur_step >= n_step:
                break
            self._obs = obs_next
        self._obs = obs_next
        if self._multi_env:
            cur_episode = sum(cur_episode)
        duration = time.time() - start_time
        self.step_speed.add(cur_step / duration)
        self.episode_speed.add(cur_episode / duration)
        self.collect_step += cur_step
        self.collect_episode += cur_episode
        self.collect_time += duration
        if isinstance(n_episode, list):
            n_episode = np.sum(n_episode)
        else:
            n_episode = max(cur_episode, 1)
        return {
            'n/ep': cur_episode,
            'n/st': cur_step,
            'v/st': self.step_speed.get(),
            'v/ep': self.episode_speed.get(),
            'rew': reward_sum / n_episode,
            'len': length_sum / n_episode,
        }

    def sample(self, batch_size):
        if self._multi_buf:
            if batch_size > 0:
                lens = [len(b) for b in self.buffer]
                total = sum(lens)
                batch_index = np.random.choice(
                    total, batch_size, p=np.array(lens) / total)
            else:
                batch_index = np.array([])
            batch_data = Batch()
            for i, b in enumerate(self.buffer):
                cur_batch = (batch_index == i).sum()
                if batch_size and cur_batch or batch_size <= 0:
                    batch, indice = b.sample(cur_batch)
                    batch = self.process_fn(batch, b, indice)
                    batch_data.append(batch)
        else:
            batch_data, indice = self.buffer.sample(batch_size)
            batch_data = self.process_fn(batch_data, self.buffer, indice)
        return batch_data
