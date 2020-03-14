import torch
import numpy as np
from copy import deepcopy

from tianshou.env import BaseVectorEnv
from tianshou.data import Batch, ReplayBuffer
from tianshou.utils import MovAvg


class Collector(object):
    """docstring for Collector"""

    def __init__(self, policy, env, buffer, contiguous=True):
        super().__init__()
        self.env = env
        self.env_num = 1
        self.buffer = buffer
        self.policy = policy
        self.process_fn = policy.process_fn
        self._multi_env = isinstance(env, BaseVectorEnv)
        self._multi_buf = False  # buf is a list
        # need multiple cache buffers only if contiguous in one buffer
        self._cached_buf = []
        if self._multi_env:
            self.env_num = len(env)
            if isinstance(self.buffer, list):
                assert len(self.buffer) == self.env_num,\
                    '# of data buffer does not match the # of input env.'
                self._multi_buf = True
            elif isinstance(self.buffer, ReplayBuffer) and contiguous:
                self._cached_buf = [
                    deepcopy(buffer) for _ in range(self.env_num)]
            else:
                raise TypeError('The buffer in data collector is invalid!')
        self.reset_env()
        self.clear_buffer()
        # state over batch is either a list, an np.ndarray, or torch.Tensor
        self.state = None
        self.stat_reward = MovAvg()
        self.stat_length = MovAvg()

    def clear_buffer(self):
        if self._multi_buf:
            for b in self.buffer:
                b.reset()
        else:
            self.buffer.reset()

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

    def _make_batch(data):
        if isinstance(data, np.ndarray):
            return data[None]
        else:
            return [data]

    def collect(self, n_step=0, n_episode=0):
        assert sum([(n_step > 0), (n_episode > 0)]) == 1,\
            "One and only one collection number specification permitted!"
        cur_step, cur_episode = 0, 0
        while True:
            if self.multi_env:
                batch_data = Batch(
                    obs=self._obs, act=self._act, rew=self._rew,
                    done=self._done, obs_next=None, info=self._info)
            else:
                batch_data = Batch(
                    obs=self._make_batch(self._obs),
                    act=self._make_batch(self._act),
                    rew=self._make_batch(self._rew),
                    done=self._make_batch(self._done),
                    obs_next=None, info=self._make_batch(self._info))
            result = self.policy.act(batch_data, self.state)
            self.state = result.state if hasattr(result, 'state') else None
            self._act = result.act
            obs_next, self._rew, self._done, self._info = self.env.step(
                self._act)
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
                        cur_episode += 1
                        self.stat_reward.add(self.reward[i])
                        self.stat_length.add(self.length[i])
                        self.reward[i], self.length[i] = 0, 0
                        if self._cached_buf:
                            self.buffer.update(self._cached_buf[i])
                            cur_step += len(self._cached_buf[i])
                            self._cached_buf[i].reset()
                        if isinstance(self.state, list):
                            self.state[i] = None
                        else:
                            self.state[i] = self.state[i] * 0
                            if isinstance(self.state, torch.Tensor):
                                # remove ref in torch (?)
                                self.state = self.state.detach()
                if n_episode > 0 and cur_episode >= n_episode:
                    break
            else:
                self.buffer.add(
                    self._obs, self._act[0], self._rew,
                    self._done, obs_next, self._info)
                cur_step += 1
                if self._done:
                    cur_episode += 1
                    self.stat_reward.add(self.reward)
                    self.stat_length.add(self.length)
                    self.reward, self.length = 0, 0
                    self.state = None
                if n_episode > 0 and cur_episode >= n_episode:
                    break
            if n_step > 0 and cur_step >= n_step:
                break
            self._obs = obs_next
        self._obs = obs_next

    def sample(self, batch_size):
        if self._multi_buf:
            if batch_size > 0:
                lens = [len(b) for b in self.buffer]
                total = sum(lens)
                ib = np.random.choice(
                    total, batch_size, p=np.array(lens) / total)
            else:
                ib = np.array([])
            batch_data = Batch()
            for i, b in enumerate(self.buffer):
                cur_batch = (ib == i).sum()
                if batch_size and cur_batch or batch_size <= 0:
                    batch, indice = b.sample(cur_batch)
                    batch = self.process_fn(batch, b, indice)
                    batch_data.append(batch)
        else:
            batch_data, indice = self.buffer.sample(batch_size)
            batch_data = self.process_fn(batch_data, self.buffer, indice)
        return batch_data

    def stat(self):
        return {
            'reward': self.stat_reward.get(),
            'length': self.stat_length.get(),
        }
