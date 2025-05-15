# see issue #322 for detail

import copy
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from tianshou.data import Batch, Collector, CollectStats
from tianshou.data.types import (
    ActBatchProtocol,
    BatchProtocol,
    ObsBatchProtocol,
)
from tianshou.env import BaseVectorEnv, DummyVectorEnv, SubprocVectorEnv
from tianshou.env.utils import ENV_TYPE, gym_new_venv_step_type
from tianshou.algorithm.algorithm_base import Policy


class DummyDataset(Dataset):
    def __init__(self, length: int) -> None:
        self.length = length
        self.episodes = [3 * i % 5 + 1 for i in range(self.length)]

    def __getitem__(self, index: int) -> tuple[int, int]:
        assert 0 <= index < self.length
        return index, self.episodes[index]

    def __len__(self) -> int:
        return self.length


class FiniteEnv(gym.Env):
    def __init__(self, dataset: Dataset, num_replicas: int | None, rank: int | None) -> None:
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.loader = DataLoader(
            dataset,
            sampler=DistributedSampler(dataset, num_replicas, rank),
            batch_size=None,
        )
        self.iterator: Iterator | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        if self.iterator is None:
            self.iterator = iter(self.loader)
        try:
            self.current_sample, self.step_count = next(self.iterator)
            self.current_step = 0
            return self.current_sample, {}
        except StopIteration:
            self.iterator = None
            return None, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        self.current_step += 1
        assert self.current_step <= self.step_count
        return (
            0,
            1.0,
            self.current_step >= self.step_count,
            False,
            {"sample": self.current_sample, "action": action, "metric": 2.0},
        )


class FiniteVectorEnv(BaseVectorEnv):
    def __init__(self, env_fns: Sequence[Callable[[], ENV_TYPE]], **kwargs: Any) -> None:
        super().__init__(env_fns, **kwargs)
        self._alive_env_ids: set[int] = set()
        self._reset_alive_envs()
        self._default_obs: np.ndarray | None = None
        self._default_info: dict | None = None
        self.tracker: MetricTracker

    def _reset_alive_envs(self) -> None:
        if not self._alive_env_ids:
            # starting or running out
            self._alive_env_ids = set(range(self.env_num))

    # to workaround with tianshou's buffer and batch
    def _set_default_obs(self, obs: np.ndarray) -> None:
        if obs is not None and self._default_obs is None:
            self._default_obs = copy.deepcopy(obs)

    def _set_default_info(self, info: dict) -> None:
        if info is not None and self._default_info is None:
            self._default_info = copy.deepcopy(info)

    def _get_default_obs(self) -> np.ndarray | None:
        return copy.deepcopy(self._default_obs)

    def _get_default_info(self) -> dict | None:
        return copy.deepcopy(self._default_info)

    # END

    def reset(
        self,
        env_id: int | list[int] | np.ndarray | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        env_id = self._wrap_id(env_id)
        self._reset_alive_envs()

        # ask super to reset alive envs and remap to current index
        request_id = list(filter(lambda i: i in self._alive_env_ids, env_id))
        obs_list: list[np.ndarray | None] = [None] * len(env_id)
        infos: list[dict | None] = [None] * len(env_id)
        id2idx = {i: k for k, i in enumerate(env_id)}
        if request_id:
            for k, o, info in zip(request_id, *super().reset(request_id), strict=True):
                obs_list[id2idx[k]] = o
                infos[id2idx[k]] = info
        for i, o in zip(env_id, obs_list, strict=True):
            if o is None and i in self._alive_env_ids:
                self._alive_env_ids.remove(i)

        # fill empty observation with default(fake) observation
        for o in obs_list:
            self._set_default_obs(o)

        for i in range(len(obs_list)):
            if obs_list[i] is None:
                obs_list[i] = self._get_default_obs()
            if infos[i] is None:
                infos[i] = self._get_default_info()

        if not self._alive_env_ids:
            self.reset()
            raise StopIteration

        obs_list = cast(list[np.ndarray], obs_list)
        infos = cast(list[dict], infos)

        return np.stack(obs_list), np.array(infos)

    def step(
        self,
        action: np.ndarray | torch.Tensor | None,
        id: int | list[int] | np.ndarray | None = None,
    ) -> gym_new_venv_step_type:
        ids: list[int] | np.ndarray = self._wrap_id(id)
        id2idx = {i: k for k, i in enumerate(ids)}
        request_id = list(filter(lambda i: i in self._alive_env_ids, ids))
        result: list[list] = [[None, 0.0, False, False, None] for _ in range(len(ids))]

        # ask super to step alive envs and remap to current index
        assert action is not None
        if request_id:
            valid_act = np.stack([action[id2idx[i]] for i in request_id])
            for i, (r_obs, r_reward, r_term, r_trunc, r_info) in zip(
                request_id,
                zip(*super().step(valid_act, request_id), strict=True),
                strict=True,
            ):
                result[id2idx[i]] = [r_obs, r_reward, r_term, r_trunc, r_info]

        # logging
        for i, r in zip(ids, result, strict=True):
            if i in self._alive_env_ids:
                self.tracker.log(*r)

        # fill empty observation/info with default(fake)
        for _, __, ___, ____, i in result:
            self._set_default_info(i)
        for i in range(len(result)):
            if result[i][0] is None:
                result[i][0] = self._get_default_obs()
            if result[i][-1] is None:
                result[i][-1] = self._get_default_info()

        obs_list, rew_list, term_list, trunc_list, info_list = zip(*result, strict=True)
        try:
            obs_stack = np.stack(obs_list)
        except ValueError:  # different len(obs)
            obs_stack = np.array(obs_list, dtype=object)
        return (
            obs_stack,
            np.stack(rew_list),
            np.stack(term_list),
            np.stack(trunc_list),
            np.stack(info_list),
        )


class FiniteDummyVectorEnv(FiniteVectorEnv, DummyVectorEnv):
    pass


class FiniteSubprocVectorEnv(FiniteVectorEnv, SubprocVectorEnv):
    pass


class DummyPolicy(Policy):
    def __init__(self) -> None:
        super().__init__(action_space=Box(-1, 1, (1,)))

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        **kwargs: Any,
    ) -> ActBatchProtocol:
        return cast(ActBatchProtocol, Batch(act=np.stack([1] * len(batch))))


def _finite_env_factory(dataset: Dataset, num_replicas: int, rank: int) -> Callable[[], FiniteEnv]:
    return lambda: FiniteEnv(dataset, num_replicas, rank)


class MetricTracker:
    def __init__(self) -> None:
        self.counter: Counter = Counter()
        self.finished: set[int] = set()

    def log(self, obs: Any, rew: float, terminated: bool, truncated: bool, info: dict) -> None:
        assert rew == 1.0
        done = terminated or truncated
        index = info["sample"]
        if done:
            assert index not in self.finished
            self.finished.add(index)
        self.counter[index] += 1

    def validate(self) -> None:
        assert len(self.finished) == 100
        for k, v in self.counter.items():
            assert v == k * 3 % 5 + 1


def test_finite_dummy_vector_env() -> None:
    dataset = DummyDataset(100)
    envs = FiniteSubprocVectorEnv([_finite_env_factory(dataset, 5, i) for i in range(5)])
    policy = DummyPolicy()
    test_collector = Collector[CollectStats](policy, envs, exploration_noise=True)
    test_collector.reset()

    for _ in range(3):
        envs.tracker = MetricTracker()
        try:
            # TODO: why on earth 10**18?
            test_collector.collect(n_step=10**18)
        except StopIteration:
            envs.tracker.validate()


def test_finite_subproc_vector_env() -> None:
    dataset = DummyDataset(100)
    envs = FiniteSubprocVectorEnv([_finite_env_factory(dataset, 5, i) for i in range(5)])
    policy = DummyPolicy()
    test_collector = Collector[CollectStats](policy, envs, exploration_noise=True)
    test_collector.reset()

    for _ in range(3):
        envs.tracker = MetricTracker()
        try:
            test_collector.collect(n_step=10**18)
        except StopIteration:
            envs.tracker.validate()
