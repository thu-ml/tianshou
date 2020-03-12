import numpy as np
from copy import deepcopy

from tianshou.env import BaseVectorEnv
from tianshou.data import Batch, ReplayBuffer
from tianshou.utils import MovAvg

class Collector(object):
    """docstring for Collector"""
    def __init__(self, policy, env, buffer):
        super().__init__()
        self.env = env
        self.env_num = 1
        self.buffer = buffer
        self.policy = policy
        self.process_fn = policy.process_fn
        self.multi_env = isinstance(env, BaseVectorEnv)
        if self.multi_env:
            self.env_num = len(env)
            if isinstance(self.buffer, list):
                assert len(self.buffer) == self.env_num, 'The data buffer number does not match the input env number.'
            elif isinstance(self.buffer, ReplayBuffer):
                self.buffer = [deepcopy(buffer) for _ in range(self.env_num)]
            else:
                raise TypeError('The buffer in data collector is invalid!')
        self.reset_env()
        self.clear_buffer()
        # state over batch is either a list, an np.ndarray, or torch.Tensor (hasattr 'shape')
        self.state = None

    def clear_buffer(self):
        if self.multi_env:
            for b in self.buffer:
                b.reset()
        else:
            self.buffer.reset()

    def reset_env(self):
        self._obs = self.env.reset()
        self._act = self._rew = self._done = self._info = None

    def collect(self, n_step=0, n_episode=0, tqdm_hook=None):
        assert sum([(n_step > 0), (n_episode > 0)]) == 1, "One and only one collection number specification permitted!"
        cur_step = 0
        cur_episode = np.zeros(self.env_num) if self.multi_env else 0
        while True:
            if self.multi_env:
                batch_data = Batch(obs=self._obs, act=self._act, rew=self._rew, done=self._done, info=self._info)
            else:
                batch_data = Batch(obs=[self._obs], act=[self._act], rew=[self._rew], done=[self._done], info=[self_info])
            result = self.policy.act(batch_data, self.state)
            self.state = result.state
            self._act = result.act
            obs_next, self._rew, self._done, self._info = self.env.step(self._act)
            cur_step += 1
            if self.multi_env:
                for i in range(self.env_num):
                    if n_episode > 0 and cur_episode[i] < n_episode or n_episode == 0:
                        self.buffer[i].add(self._obs[i], self._act[i], self._rew[i], self._done[i], obs_next[i], self._info[i])
                        if self._done[i]:
                            cur_episode[i] += 1
                            if isinstance(self.state, list):
                                self.state[i] = None
                            else:
                                self.state[i] = self.state[i] * 0
                                if hasattr(self.state, 'detach'):  # remove count in torch
                                    self.state = self.state.detach()
                if n_episode > 0 and (cur_episode >= n_episode).all():
                    break
            else:
                self.buffer.add(self._obs, self._act[0], self._rew, self._done, obs_next, self._info)
                if self._done:
                    cur_episode += 1
                    self.state = None
                if n_episode > 0 and cur_episode >= n_episode:
                    break
            if n_step > 0 and cur_step >= n_step:
                break
            self._obs = obs_next
        self._obs = obs_next

    def sample(self):
        pass

    def stat(self):
        pass
