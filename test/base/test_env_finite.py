# see issue #322 for detail

import copy
from collections import Counter

import gym
import numpy as np
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from tianshou.data import Batch, Collector
from tianshou.env import BaseVectorEnv, DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import BasePolicy


class DummyDataset(Dataset):

    def __init__(self, length):
        self.length = length
        self.episodes = [3 * i % 5 + 1 for i in range(self.length)]

    def __getitem__(self, index):
        assert 0 <= index < self.length
        return index, self.episodes[index]

    def __len__(self):
        return self.length


class FiniteEnv(gym.Env):

    def __init__(self, dataset, num_replicas, rank):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.loader = DataLoader(
            dataset,
            sampler=DistributedSampler(dataset, num_replicas, rank),
            batch_size=None
        )
        self.iterator = None

    def reset(self):
        if self.iterator is None:
            self.iterator = iter(self.loader)
        try:
            self.current_sample, self.step_count = next(self.iterator)
            self.current_step = 0
            return self.current_sample
        except StopIteration:
            self.iterator = None
            return None

    def step(self, action):
        self.current_step += 1
        assert self.current_step <= self.step_count
        return 0, 1.0, self.current_step >= self.step_count, \
            {'sample': self.current_sample, 'action': action, 'metric': 2.0}


class FiniteVectorEnv(BaseVectorEnv):

    def __init__(self, env_fns, **kwargs):
        super().__init__(env_fns, **kwargs)
        self._alive_env_ids = set()
        self._reset_alive_envs()
        self._default_obs = self._default_info = None

    def _reset_alive_envs(self):
        if not self._alive_env_ids:
            # starting or running out
            self._alive_env_ids = set(range(self.env_num))

    # to workaround with tianshou's buffer and batch
    def _set_default_obs(self, obs):
        if obs is not None and self._default_obs is None:
            self._default_obs = copy.deepcopy(obs)

    def _set_default_info(self, info):
        if info is not None and self._default_info is None:
            self._default_info = copy.deepcopy(info)

    def _get_default_obs(self):
        return copy.deepcopy(self._default_obs)

    def _get_default_info(self):
        return copy.deepcopy(self._default_info)

    # END

    def reset(self, id=None):
        id = self._wrap_id(id)
        self._reset_alive_envs()

        # ask super to reset alive envs and remap to current index
        request_id = list(filter(lambda i: i in self._alive_env_ids, id))
        obs = [None] * len(id)
        id2idx = {i: k for k, i in enumerate(id)}
        if request_id:
            for i, o in zip(request_id, super().reset(request_id)):
                obs[id2idx[i]] = o
        for i, o in zip(id, obs):
            if o is None and i in self._alive_env_ids:
                self._alive_env_ids.remove(i)

        # fill empty observation with default(fake) observation
        for o in obs:
            self._set_default_obs(o)
        for i in range(len(obs)):
            if obs[i] is None:
                obs[i] = self._get_default_obs()

        if not self._alive_env_ids:
            self.reset()
            raise StopIteration

        return np.stack(obs)

    def step(self, action, id=None):
        id = self._wrap_id(id)
        id2idx = {i: k for k, i in enumerate(id)}
        request_id = list(filter(lambda i: i in self._alive_env_ids, id))
        result = [[None, 0., False, None] for _ in range(len(id))]

        # ask super to step alive envs and remap to current index
        if request_id:
            valid_act = np.stack([action[id2idx[i]] for i in request_id])
            for i, r in zip(request_id, zip(*super().step(valid_act, request_id))):
                result[id2idx[i]] = r

        # logging
        for i, r in zip(id, result):
            if i in self._alive_env_ids:
                self.tracker.log(*r)

        # fill empty observation/info with default(fake)
        for _, __, ___, i in result:
            self._set_default_info(i)
        for i in range(len(result)):
            if result[i][0] is None:
                result[i][0] = self._get_default_obs()
            if result[i][3] is None:
                result[i][3] = self._get_default_info()

        return list(map(np.stack, zip(*result)))


class FiniteDummyVectorEnv(FiniteVectorEnv, DummyVectorEnv):
    pass


class FiniteSubprocVectorEnv(FiniteVectorEnv, SubprocVectorEnv):
    pass


class AnyPolicy(BasePolicy):

    def forward(self, batch, state=None):
        return Batch(act=np.stack([1] * len(batch)))

    def learn(self, batch):
        pass


def _finite_env_factory(dataset, num_replicas, rank):
    return lambda: FiniteEnv(dataset, num_replicas, rank)


class MetricTracker:

    def __init__(self):
        self.counter = Counter()
        self.finished = set()

    def log(self, obs, rew, done, info):
        assert rew == 1.
        index = info['sample']
        if done:
            assert index not in self.finished
            self.finished.add(index)
        self.counter[index] += 1

    def validate(self):
        assert len(self.finished) == 100
        for k, v in self.counter.items():
            assert v == k * 3 % 5 + 1


def test_finite_dummy_vector_env():
    dataset = DummyDataset(100)
    envs = FiniteSubprocVectorEnv(
        [_finite_env_factory(dataset, 5, i) for i in range(5)]
    )
    policy = AnyPolicy()
    test_collector = Collector(policy, envs, exploration_noise=True)

    for _ in range(3):
        envs.tracker = MetricTracker()
        try:
            test_collector.collect(n_step=10**18)
        except StopIteration:
            envs.tracker.validate()


def test_finite_subproc_vector_env():
    dataset = DummyDataset(100)
    envs = FiniteSubprocVectorEnv(
        [_finite_env_factory(dataset, 5, i) for i in range(5)]
    )
    policy = AnyPolicy()
    test_collector = Collector(policy, envs, exploration_noise=True)

    for _ in range(3):
        envs.tracker = MetricTracker()
        try:
            test_collector.collect(n_step=10**18)
        except StopIteration:
            envs.tracker.validate()


if __name__ == '__main__':
    test_finite_dummy_vector_env()
    test_finite_subproc_vector_env()
