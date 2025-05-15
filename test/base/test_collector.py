from collections.abc import Callable, Sequence
from test.base.env import MoveToRightEnv, NXEnv
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import tqdm

from tianshou.algorithm.algorithm_base import Policy, episode_mc_return_to_go
from tianshou.data import (
    AsyncCollector,
    Batch,
    CachedReplayBuffer,
    Collector,
    CollectStats,
    PrioritizedReplayBuffer,
    ReplayBuffer,
    VectorReplayBuffer,
)
from tianshou.data.batch import BatchProtocol
from tianshou.data.collector import (
    CollectActionBatchProtocol,
    EpisodeRolloutHookMCReturn,
    StepHook,
)
from tianshou.data.types import ObsBatchProtocol, RolloutBatchProtocol
from tianshou.env import DummyVectorEnv, SubprocVectorEnv

try:
    import envpool
except ImportError:
    envpool = None


class MaxActionPolicy(Policy):
    def __init__(
        self,
        action_space: gym.spaces.Space | None = None,
        dict_state: bool = False,
        need_state: bool = True,
        action_shape: Sequence[int] | int | None = None,
    ) -> None:
        """Mock policy for testing, will always return an array of ones of the shape of the action space.
        Note that this doesn't make much sense for discrete action space (the output is then intepreted as
        logits, meaning all actions would be equally likely).

        :param action_space: the action space of the environment. If None, a dummy Box space will be used.
        :param bool dict_state: if the observation of the environment is a dict
        :param bool need_state: if the policy needs the hidden state (for RNN)
        """
        action_space = action_space or gym.spaces.Box(-1, 1, (1,))
        super().__init__(action_space=action_space)
        self.dict_state = dict_state
        self.need_state = need_state
        self.action_shape = action_shape

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        **kwargs: Any,
    ) -> Batch:
        if self.need_state:
            if state is None:
                state = np.zeros((len(batch.obs), 2))
            elif isinstance(state, np.ndarray | BatchProtocol):
                state += np.int_(1)
            elif isinstance(state, dict) and state.get("hidden") is not None:
                state["hidden"] += np.int_(1)
        if self.dict_state:
            if self.action_shape:
                action_shape = self.action_shape
            elif isinstance(batch.obs, BatchProtocol):
                action_shape = len(batch.obs["index"])
            else:
                action_shape = len(batch.obs)
            return Batch(act=np.ones(action_shape), state=state)
        action_shape = self.action_shape if self.action_shape else len(batch.obs)
        return Batch(act=np.ones(action_shape), state=state)


@pytest.fixture()
def collector_with_single_env() -> Collector[CollectStats]:
    """The env will be a MoveToRightEnv with size 5, sleep 0."""
    env = MoveToRightEnv(size=5, sleep=0)
    policy = MaxActionPolicy()
    collector = Collector[CollectStats](policy, env, ReplayBuffer(size=100))
    collector.reset()
    return collector


def test_collector() -> None:
    env_fns = [lambda x=i: MoveToRightEnv(size=x, sleep=0) for i in [2, 3, 4, 5]]

    subproc_venv_4_envs = SubprocVectorEnv(env_fns)
    dummy_venv_4_envs = DummyVectorEnv(env_fns)
    policy = MaxActionPolicy()
    single_env = env_fns[0]()
    c_single_env = Collector[CollectStats](
        policy,
        single_env,
        ReplayBuffer(size=100),
    )
    c_single_env.reset()
    c_single_env.collect(n_step=3)
    assert len(c_single_env.buffer) == 3
    # TODO: direct attr access is an arcane way of using the buffer, it should be never done
    #  The placeholders for entries are all zeros, so buffer.obs is an array filled with 3
    #  observations, and 97 zeros.
    #  However, buffer[:] will have all attributes with length three... The non-filled entries are removed there

    # See above. For the single env, we start with obs=0, obs_next=1.
    # We move to obs=1, obs_next=2,
    # then the env is reset and we move to obs=0
    # Making one more step results in obs_next=1
    # The final 0 in the buffer.obs is because the buffer is initialized with zeros and the direct attr access
    assert np.allclose(c_single_env.buffer.obs[:4, 0], [0, 1, 0, 0])
    obs_next = c_single_env.buffer[:].obs_next[..., 0]
    assert isinstance(obs_next, np.ndarray)
    assert np.allclose(obs_next, [1, 2, 1])
    keys = np.zeros(100)
    keys[:3] = 1
    assert np.allclose(c_single_env.buffer.info["key"], keys)
    for e in c_single_env.buffer.info["env"][:3]:
        assert isinstance(e, MoveToRightEnv)
    assert np.allclose(c_single_env.buffer.info["env_id"], 0)
    rews = np.zeros(100)
    rews[:3] = [0, 1, 0]
    assert np.allclose(c_single_env.buffer.rew, rews)
    # At this point, the buffer contains obs 0 -> 1 -> 0

    # At start we have 3 entries in the buffer
    # We collect 3 episodes, in addition to the transitions we have collected before
    # 0 -> 1 -> 0 -> 0 (reset at collection start) -> 1 -> done (0) -> 1 -> done(0)
    # obs_next: 1 -> 2 -> 1 -> 1 (reset at collection start) -> 2 -> 1 -> 2 -> 1 -> 2
    # In total, we will have 3 + 6 = 9 entries in the buffer
    c_single_env.collect(n_episode=3)
    assert len(c_single_env.buffer) == 8
    assert np.allclose(c_single_env.buffer.obs[:10, 0], [0, 1, 0, 1, 0, 1, 0, 1, 0, 0])
    obs_next = c_single_env.buffer[:].obs_next[..., 0]
    assert isinstance(obs_next, np.ndarray)
    assert np.allclose(obs_next, [1, 2, 1, 2, 1, 2, 1, 2])
    assert np.allclose(c_single_env.buffer.info["key"][:8], 1)
    for e in c_single_env.buffer.info["env"][:8]:
        assert isinstance(e, MoveToRightEnv)
    assert np.allclose(c_single_env.buffer.info["env_id"][:8], 0)
    assert np.allclose(c_single_env.buffer.rew[:8], [0, 1, 0, 1, 0, 1, 0, 1])
    c_single_env.collect(n_step=3, random=True)

    c_subproc_venv_4_envs = Collector[CollectStats](
        policy,
        subproc_venv_4_envs,
        VectorReplayBuffer(total_size=100, buffer_num=4),
    )
    c_subproc_venv_4_envs.reset()

    # Collect some steps
    c_subproc_venv_4_envs.collect(n_step=8)
    obs = np.zeros(100)
    valid_indices = [0, 1, 25, 26, 50, 51, 75, 76]
    obs[valid_indices] = [0, 1, 0, 1, 0, 1, 0, 1]
    assert np.allclose(c_subproc_venv_4_envs.buffer.obs[:, 0], obs)
    obs_next = c_subproc_venv_4_envs.buffer[:].obs_next[..., 0]
    assert isinstance(obs_next, np.ndarray)
    assert np.allclose(obs_next, [1, 2, 1, 2, 1, 2, 1, 2])
    keys = np.zeros(100)
    keys[valid_indices] = [1, 1, 1, 1, 1, 1, 1, 1]
    assert np.allclose(c_subproc_venv_4_envs.buffer.info["key"], keys)
    for e in c_subproc_venv_4_envs.buffer.info["env"][valid_indices]:
        assert isinstance(e, MoveToRightEnv)
    env_ids = np.zeros(100)
    env_ids[valid_indices] = [0, 0, 1, 1, 2, 2, 3, 3]
    assert np.allclose(c_subproc_venv_4_envs.buffer.info["env_id"], env_ids)
    rews = np.zeros(100)
    rews[valid_indices] = [0, 1, 0, 0, 0, 0, 0, 0]
    assert np.allclose(c_subproc_venv_4_envs.buffer.rew, rews)

    # we previously collected 8 steps, 2 from each env, now we collect 4 episodes
    # each env will contribute an episode, which will be of lens 2 (first env was reset), 1, 2, 3
    # So we get 8 + 2+1+2+3 = 16 steps
    c_subproc_venv_4_envs.collect(n_episode=4)
    assert len(c_subproc_venv_4_envs.buffer) == 16

    valid_indices = [2, 3, 27, 52, 53, 77, 78, 79]
    obs[valid_indices] = [0, 1, 2, 2, 3, 2, 3, 4]
    assert np.allclose(c_subproc_venv_4_envs.buffer.obs[:, 0], obs)
    obs_next = c_subproc_venv_4_envs.buffer[:].obs_next[..., 0]
    assert isinstance(obs_next, np.ndarray)
    assert np.allclose(
        obs_next,
        [1, 2, 1, 2, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 5],
    )
    keys[valid_indices] = [1, 1, 1, 1, 1, 1, 1, 1]
    assert np.allclose(c_subproc_venv_4_envs.buffer.info["key"], keys)
    for e in c_subproc_venv_4_envs.buffer.info["env"][valid_indices]:
        assert isinstance(e, MoveToRightEnv)
    env_ids[valid_indices] = [0, 0, 1, 2, 2, 3, 3, 3]
    assert np.allclose(c_subproc_venv_4_envs.buffer.info["env_id"], env_ids)
    rews[valid_indices] = [0, 1, 1, 0, 1, 0, 0, 1]
    assert np.allclose(c_subproc_venv_4_envs.buffer.rew, rews)
    c_subproc_venv_4_envs.collect(n_episode=4, random=True)

    c_dummy_venv_4_envs = Collector[CollectStats](
        policy,
        dummy_venv_4_envs,
        VectorReplayBuffer(total_size=100, buffer_num=4),
    )
    c_dummy_venv_4_envs.reset()
    c_dummy_venv_4_envs.collect(n_episode=7)
    obs1 = obs.copy()
    obs1[[4, 5, 28, 29, 30]] = [0, 1, 0, 1, 2]
    obs2 = obs.copy()
    obs2[[28, 29, 30, 54, 55, 56, 57]] = [0, 1, 2, 0, 1, 2, 3]
    c2obs = c_dummy_venv_4_envs.buffer.obs[:, 0]
    assert np.all(c2obs == obs1) or np.all(c2obs == obs2)
    c_dummy_venv_4_envs.reset_env()
    c_dummy_venv_4_envs.reset_buffer()
    assert c_dummy_venv_4_envs.collect(n_episode=8).n_collected_episodes == 8
    valid_indices = [4, 5, 28, 29, 30, 54, 55, 56, 57]
    obs[valid_indices] = [0, 1, 0, 1, 2, 0, 1, 2, 3]
    assert np.all(c_dummy_venv_4_envs.buffer.obs[:, 0] == obs)
    keys[valid_indices] = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert np.allclose(c_dummy_venv_4_envs.buffer.info["key"], keys)
    for e in c_dummy_venv_4_envs.buffer.info["env"][valid_indices]:
        assert isinstance(e, MoveToRightEnv)
    env_ids[valid_indices] = [0, 0, 1, 1, 1, 2, 2, 2, 2]
    assert np.allclose(c_dummy_venv_4_envs.buffer.info["env_id"], env_ids)
    rews[valid_indices] = [0, 1, 0, 0, 1, 0, 0, 0, 1]
    assert np.allclose(c_dummy_venv_4_envs.buffer.rew, rews)
    c_dummy_venv_4_envs.collect(n_episode=4, random=True)

    # test corner case
    with pytest.raises(ValueError):
        Collector[CollectStats](policy, dummy_venv_4_envs, ReplayBuffer(10))
    with pytest.raises(ValueError):
        Collector[CollectStats](policy, dummy_venv_4_envs, PrioritizedReplayBuffer(10, 0.5, 0.5))
    with pytest.raises(ValueError):
        c_dummy_venv_4_envs.collect()

    def get_env_factory(i: int, t: str) -> Callable[[], NXEnv]:
        return lambda: NXEnv(i, t)

    # test NXEnv
    for obs_type in ["array", "object"]:
        envs = SubprocVectorEnv([get_env_factory(i=i, t=obs_type) for i in [5, 10, 15, 20]])
        c_suproc_new = Collector[CollectStats](
            policy,
            envs,
            VectorReplayBuffer(total_size=100, buffer_num=4),
        )
        c_suproc_new.reset()
        c_suproc_new.collect(n_step=6)
        assert c_suproc_new.buffer.obs.dtype == object


@pytest.fixture()
def async_collector_and_env_lens() -> tuple[AsyncCollector, list[int]]:
    env_lens = [2, 3, 4, 5]
    env_fns = [lambda x=i: MoveToRightEnv(size=x, sleep=0.001, random_sleep=True) for i in env_lens]

    venv = SubprocVectorEnv(env_fns, wait_num=len(env_fns) - 1)
    policy = MaxActionPolicy()
    bufsize = 60
    async_collector = AsyncCollector(
        policy,
        venv,
        VectorReplayBuffer(total_size=bufsize * 4, buffer_num=4),
    )
    async_collector.reset()
    return async_collector, env_lens


class TestAsyncCollector:
    def test_collect_without_argument_gives_error(
        self,
        async_collector_and_env_lens: tuple[AsyncCollector, list[int]],
    ) -> None:
        c1, env_lens = async_collector_and_env_lens
        with pytest.raises(ValueError):
            c1.collect()

    def test_collect_one_episode_async(
        self,
        async_collector_and_env_lens: tuple[AsyncCollector, list[int]],
    ) -> None:
        c1, env_lens = async_collector_and_env_lens
        result = c1.collect(n_episode=1)
        assert result.n_collected_episodes >= 1

    def test_enough_episodes_two_collection_cycles_n_episode_without_reset(
        self,
        async_collector_and_env_lens: tuple[AsyncCollector, list[int]],
    ) -> None:
        c1, env_lens = async_collector_and_env_lens
        n_episode = 2
        result_c1 = c1.collect(n_episode=n_episode, reset_before_collect=False)
        assert result_c1.n_collected_episodes >= n_episode
        result_c2 = c1.collect(n_episode=n_episode, reset_before_collect=False)
        assert result_c2.n_collected_episodes >= n_episode

    def test_enough_episodes_two_collection_cycles_n_episode_with_reset(
        self,
        async_collector_and_env_lens: tuple[AsyncCollector, list[int]],
    ) -> None:
        c1, env_lens = async_collector_and_env_lens
        n_episode = 2
        result_c1 = c1.collect(n_episode=n_episode, reset_before_collect=True)
        assert result_c1.n_collected_episodes >= n_episode
        result_c2 = c1.collect(n_episode=n_episode, reset_before_collect=True)
        assert result_c2.n_collected_episodes >= n_episode

    def test_enough_episodes_and_correct_obs_indices_and_obs_next_iterative_collection_cycles_n_episode(
        self,
        async_collector_and_env_lens: tuple[AsyncCollector, list[int]],
    ) -> None:
        c1, env_lens = async_collector_and_env_lens
        ptr = [0, 0, 0, 0]
        bufsize = 60
        for n_episode in tqdm.trange(1, 30, desc="test async n_episode"):
            result = c1.collect(n_episode=n_episode)
            assert result.n_collected_episodes >= n_episode
            # check buffer data, obs and obs_next, env_id
            for i, count in enumerate(np.bincount(result.lens, minlength=6)[2:]):
                env_len = i + 2
                total = env_len * count
                indices = np.arange(ptr[i], ptr[i] + total) % bufsize
                ptr[i] = (ptr[i] + total) % bufsize
                seq = np.arange(env_len)
                buf = c1.buffer.buffers[i]
                assert np.all(buf.info.env_id[indices] == i)
                assert np.all(buf.obs[indices].reshape(count, env_len) == seq)
                assert np.all(buf.obs_next[indices].reshape(count, env_len) == seq + 1)

    def test_enough_episodes_and_correct_obs_indices_and_obs_next_iterative_collection_cycles_n_step(
        self,
        async_collector_and_env_lens: tuple[AsyncCollector, list[int]],
    ) -> None:
        c1, env_lens = async_collector_and_env_lens
        bufsize = 60
        ptr = [0, 0, 0, 0]
        for n_step in tqdm.trange(1, 15, desc="test async n_step"):
            result = c1.collect(n_step=n_step)
            assert result.n_collected_steps >= n_step
            for i, count in enumerate(np.bincount(result.lens, minlength=6)[2:]):
                env_len = i + 2
                total = env_len * count
                indices = np.arange(ptr[i], ptr[i] + total) % bufsize
                ptr[i] = (ptr[i] + total) % bufsize
                seq = np.arange(env_len)
                buf = c1.buffer.buffers[i]
                assert np.all(buf.info.env_id[indices] == i)
                assert np.all(buf.obs[indices].reshape(count, env_len) == seq)
                assert np.all(buf.obs_next[indices].reshape(count, env_len) == seq + 1)

    def test_enough_episodes_and_correct_obs_indices_and_obs_next_iterative_collection_cycles_first_n_episode_then_n_step(
        self,
        async_collector_and_env_lens: tuple[AsyncCollector, list[int]],
    ) -> None:
        c1, env_lens = async_collector_and_env_lens
        bufsize = 60
        ptr = [0, 0, 0, 0]
        for n_episode in tqdm.trange(1, 30, desc="test async n_episode"):
            result = c1.collect(n_episode=n_episode)
            assert result.n_collected_episodes >= n_episode
            # check buffer data, obs and obs_next, env_id
            for i, count in enumerate(np.bincount(result.lens, minlength=6)[2:]):
                env_len = i + 2
                total = env_len * count
                indices = np.arange(ptr[i], ptr[i] + total) % bufsize
                ptr[i] = (ptr[i] + total) % bufsize
                seq = np.arange(env_len)
                buf = c1.buffer.buffers[i]
                assert np.all(buf.info.env_id[indices] == i)
                assert np.all(buf.obs[indices].reshape(count, env_len) == seq)
                assert np.all(buf.obs_next[indices].reshape(count, env_len) == seq + 1)
        # test async n_step, for now the buffer should be full of data, thus no bincount stuff as above
        for n_step in tqdm.trange(1, 15, desc="test async n_step"):
            result = c1.collect(n_step=n_step)
            assert result.n_collected_steps >= n_step
            for i in range(4):
                env_len = i + 2
                seq = np.arange(env_len)
                buf = c1.buffer.buffers[i]
                assert np.all(buf.info.env_id == i)
                assert np.all(buf.obs.reshape(-1, env_len) == seq)
                assert np.all(buf.obs_next.reshape(-1, env_len) == seq + 1)


def test_collector_with_dict_state() -> None:
    env = MoveToRightEnv(size=5, sleep=0, dict_state=True)
    policy = MaxActionPolicy(dict_state=True)
    c0 = Collector[CollectStats](policy, env, ReplayBuffer(size=100))
    c0.reset()
    c0.collect(n_step=3)
    c0.collect(n_episode=2)
    assert len(c0.buffer) == 10  # 3 + two episodes with 5 steps each
    env_fns = [lambda x=i: MoveToRightEnv(size=x, sleep=0, dict_state=True) for i in [2, 3, 4, 5]]
    envs = DummyVectorEnv(env_fns)
    envs.seed(666)
    obs, info = envs.reset()
    assert not np.isclose(obs[0]["rand"], obs[1]["rand"])
    c1 = Collector[CollectStats](
        policy,
        envs,
        VectorReplayBuffer(total_size=100, buffer_num=4),
    )
    c1.reset()
    c1.collect(n_step=12)
    result = c1.collect(n_episode=8)
    assert result.n_collected_episodes == 8
    lens = np.bincount(result.lens)
    assert (
        result.n_collected_steps == 21
        and np.all(lens == [0, 0, 2, 2, 2, 2])
        or result.n_collected_steps == 20
        and np.all(lens == [0, 0, 3, 1, 2, 2])
    )
    batch, _ = c1.buffer.sample(10)
    c0.buffer.update(c1.buffer)
    assert len(c0.buffer) in [42, 43]
    cur_obs = c0.buffer[:].obs
    assert isinstance(cur_obs, Batch)
    if len(c0.buffer) == 42:
        assert np.all(
            cur_obs.index[..., 0]
            == [
                0,
                1,
                2,
                3,
                4,
                0,
                1,
                2,
                3,
                4,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                2,
                0,
                1,
                2,
                0,
                1,
                2,
                3,
                0,
                1,
                2,
                3,
                0,
                1,
                2,
                3,
                4,
                0,
                1,
                2,
                3,
                4,
            ],
        ), cur_obs.index[..., 0]
    else:
        assert np.all(
            cur_obs.index[..., 0]
            == [
                0,
                1,
                2,
                3,
                4,
                0,
                1,
                2,
                3,
                4,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                2,
                0,
                1,
                2,
                0,
                1,
                2,
                0,
                1,
                2,
                3,
                0,
                1,
                2,
                3,
                0,
                1,
                2,
                3,
                4,
                0,
                1,
                2,
                3,
                4,
            ],
        ), cur_obs.index[..., 0]
    c2 = Collector[CollectStats](
        policy,
        envs,
        VectorReplayBuffer(total_size=100, buffer_num=4, stack_num=4),
    )
    c2.reset()
    c2.collect(n_episode=10)
    batch, _ = c2.buffer.sample(10)


def test_collector_with_multi_agent() -> None:
    multi_agent_env = MoveToRightEnv(size=5, sleep=0, ma_rew=4)
    policy = MaxActionPolicy()
    c_single_env = Collector[CollectStats](policy, multi_agent_env, ReplayBuffer(size=100))
    c_single_env.reset()
    multi_env_returns = c_single_env.collect(n_step=3).returns
    # c_single_env has length 3
    # We have no full episodes, so no returns yet
    assert len(multi_env_returns) == 0

    single_env_returns = c_single_env.collect(n_episode=2).returns
    # now two episodes. Since we have 4 a agents, the returns have shape (2, 4)
    assert single_env_returns.shape == (2, 4)
    assert np.all(single_env_returns == 1)

    env_fns = [lambda x=i: MoveToRightEnv(size=x, sleep=0, ma_rew=4) for i in [2, 3, 4, 5]]
    envs = DummyVectorEnv(env_fns)
    c_multi_env_ma = Collector[CollectStats](
        policy,
        envs,
        VectorReplayBuffer(total_size=100, buffer_num=4),
    )
    c_multi_env_ma.reset()
    multi_env_returns = c_multi_env_ma.collect(n_step=12).returns
    # each env makes 3 steps, the first two envs are done and result in two finished episodes
    assert multi_env_returns.shape == (2, 4) and np.all(multi_env_returns == 1), multi_env_returns
    multi_env_returns = c_multi_env_ma.collect(n_episode=8).returns
    assert multi_env_returns.shape == (8, 4)
    assert np.all(multi_env_returns == 1)
    batch, _ = c_multi_env_ma.buffer.sample(10)
    print(batch)
    c_single_env.buffer.update(c_multi_env_ma.buffer)
    assert len(c_single_env.buffer) in [42, 43]
    if len(c_single_env.buffer) == 42:
        multi_env_returns = np.array(
            [
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
            ],
        )
    else:
        multi_env_returns = np.array(
            [
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
            ],
        )
    assert np.all(c_single_env.buffer[:].rew == [[x] * 4 for x in multi_env_returns])
    assert np.all(c_single_env.buffer[:].done == multi_env_returns)
    c2 = Collector[CollectStats](
        policy,
        envs,
        VectorReplayBuffer(total_size=100, buffer_num=4, stack_num=4),
    )
    c2.reset()
    multi_env_returns = c2.collect(n_episode=10).returns
    assert multi_env_returns.shape == (10, 4)
    assert np.all(multi_env_returns == 1)
    batch, _ = c2.buffer.sample(10)


def test_collector_with_atari_setting() -> None:
    reference_obs = np.zeros([6, 4, 84, 84])
    for i in range(6):
        reference_obs[i, 3, np.arange(84), np.arange(84)] = i
        reference_obs[i, 2, np.arange(84)] = i
        reference_obs[i, 1, :, np.arange(84)] = i
        reference_obs[i, 0] = i

    # atari single buffer
    env = MoveToRightEnv(size=5, sleep=0, array_state=True)
    policy = MaxActionPolicy()
    c0 = Collector[CollectStats](policy, env, ReplayBuffer(size=100))
    c0.reset()
    c0.collect(n_step=6)
    c0.collect(n_episode=2)
    assert c0.buffer.obs.shape == (100, 4, 84, 84)
    assert c0.buffer.obs_next.shape == (100, 4, 84, 84)
    assert len(c0.buffer) == 15  # 6 + 2 episodes with 5 steps each
    obs = np.zeros_like(c0.buffer.obs)
    obs[np.arange(15)] = reference_obs[np.arange(15) % 5]
    assert np.all(obs == c0.buffer.obs)

    c1 = Collector[CollectStats](policy, env, ReplayBuffer(size=100, ignore_obs_next=True))
    c1.collect(n_episode=3, reset_before_collect=True)
    assert np.allclose(c0.buffer.obs, c1.buffer.obs)
    with pytest.raises(AttributeError):
        c1.buffer.obs_next  # noqa: B018
    assert np.all(reference_obs[[1, 2, 3, 4, 4] * 3] == c1.buffer[:].obs_next)

    c2 = Collector[CollectStats](
        policy,
        env,
        ReplayBuffer(size=100, ignore_obs_next=True, save_only_last_obs=True),
    )
    c2.reset()
    c2.collect(n_step=8)
    assert c2.buffer.obs.shape == (100, 84, 84)
    obs = np.zeros_like(c2.buffer.obs)
    obs[np.arange(8)] = reference_obs[[0, 1, 2, 3, 4, 0, 1, 2], -1]
    assert np.all(c2.buffer.obs == obs)
    obs_next = c2.buffer[:].obs_next
    assert isinstance(obs_next, np.ndarray)
    assert np.allclose(obs_next, reference_obs[[1, 2, 3, 4, 4, 1, 2, 2], -1])

    # atari multi buffer
    env_fns = [lambda x=i: MoveToRightEnv(size=x, sleep=0, array_state=True) for i in [2, 3, 4, 5]]
    envs = DummyVectorEnv(env_fns)
    c3 = Collector[CollectStats](policy, envs, VectorReplayBuffer(total_size=100, buffer_num=4))
    c3.reset()
    c3.collect(n_step=12)
    result_cached_buffer_collect_9_episodes = c3.collect(n_episode=9)
    assert result_cached_buffer_collect_9_episodes.n_collected_episodes == 9
    assert result_cached_buffer_collect_9_episodes.n_collected_steps == 23
    assert c3.buffer.obs.shape == (100, 4, 84, 84)
    obs = np.zeros_like(c3.buffer.obs)
    obs[np.arange(8)] = reference_obs[[0, 1, 0, 1, 0, 1, 0, 1]]
    obs[np.arange(25, 34)] = reference_obs[[0, 1, 2, 0, 1, 2, 0, 1, 2]]
    obs[np.arange(50, 58)] = reference_obs[[0, 1, 2, 3, 0, 1, 2, 3]]
    obs[np.arange(75, 85)] = reference_obs[[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]]
    assert np.all(obs == c3.buffer.obs)
    obs_next = np.zeros_like(c3.buffer.obs_next)
    obs_next[np.arange(8)] = reference_obs[[1, 2, 1, 2, 1, 2, 1, 2]]
    obs_next[np.arange(25, 34)] = reference_obs[[1, 2, 3, 1, 2, 3, 1, 2, 3]]
    obs_next[np.arange(50, 58)] = reference_obs[[1, 2, 3, 4, 1, 2, 3, 4]]
    obs_next[np.arange(75, 85)] = reference_obs[[1, 2, 3, 4, 5, 1, 2, 3, 4, 5]]
    assert np.all(obs_next == c3.buffer.obs_next)
    c4 = Collector[CollectStats](
        policy,
        envs,
        VectorReplayBuffer(
            total_size=100,
            buffer_num=4,
            stack_num=4,
            ignore_obs_next=True,
            save_only_last_obs=True,
        ),
    )
    c4.reset()
    c4.collect(n_step=12)
    result_cached_buffer_collect_9_episodes = c4.collect(n_episode=9)
    assert result_cached_buffer_collect_9_episodes.n_collected_episodes == 9
    assert result_cached_buffer_collect_9_episodes.n_collected_steps == 23
    assert c4.buffer.obs.shape == (100, 84, 84)
    obs = np.zeros_like(c4.buffer.obs)
    slice_obs = reference_obs[:, -1]
    obs[np.arange(8)] = slice_obs[[0, 1, 0, 1, 0, 1, 0, 1]]
    obs[np.arange(25, 34)] = slice_obs[[0, 1, 2, 0, 1, 2, 0, 1, 2]]
    obs[np.arange(50, 58)] = slice_obs[[0, 1, 2, 3, 0, 1, 2, 3]]
    obs[np.arange(75, 85)] = slice_obs[[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]]
    assert np.all(c4.buffer.obs == obs)
    obs_next = np.zeros([len(c4.buffer), 4, 84, 84])
    ref_index = np.array(
        [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            1,
            2,
            2,
            1,
            2,
            2,
            1,
            2,
            3,
            3,
            1,
            2,
            3,
            3,
            1,
            2,
            3,
            4,
            4,
            1,
            2,
            3,
            4,
            4,
        ],
    )
    obs_next[:, -1] = slice_obs[ref_index]
    ref_index -= 1
    ref_index[ref_index < 0] = 0
    obs_next[:, -2] = slice_obs[ref_index]
    ref_index -= 1
    ref_index[ref_index < 0] = 0
    obs_next[:, -3] = slice_obs[ref_index]
    ref_index -= 1
    ref_index[ref_index < 0] = 0
    obs_next[:, -4] = slice_obs[ref_index]
    assert np.all(obs_next == c4.buffer[:].obs_next)

    buf = ReplayBuffer(100, stack_num=4, ignore_obs_next=True, save_only_last_obs=True)
    collector_cached_buffer = Collector[CollectStats](policy, envs, CachedReplayBuffer(buf, 4, 10))
    collector_cached_buffer.reset()
    result_cached_buffer_collect_12_steps = collector_cached_buffer.collect(n_step=12)
    assert len(buf) == 5
    assert len(collector_cached_buffer.buffer) == 12
    result_cached_buffer_collect_9_episodes = collector_cached_buffer.collect(n_episode=9)
    assert result_cached_buffer_collect_9_episodes.n_collected_episodes == 9
    assert result_cached_buffer_collect_9_episodes.n_collected_steps == 23
    assert len(buf) == 35
    assert np.all(
        buf.obs[: len(buf)]
        == slice_obs[
            [
                0,
                1,
                0,
                1,
                2,
                0,
                1,
                0,
                1,
                2,
                3,
                0,
                1,
                2,
                3,
                4,
                0,
                1,
                0,
                1,
                2,
                0,
                1,
                0,
                1,
                2,
                3,
                0,
                1,
                2,
                0,
                1,
                2,
                3,
                4,
            ]
        ],
    )
    assert np.all(
        buf[:].obs_next[:, -1]
        == slice_obs[
            [
                1,
                1,
                1,
                2,
                2,
                1,
                1,
                1,
                2,
                3,
                3,
                1,
                2,
                3,
                4,
                4,
                1,
                1,
                1,
                2,
                2,
                1,
                1,
                1,
                2,
                3,
                3,
                1,
                2,
                2,
                1,
                2,
                3,
                4,
                4,
            ]
        ],
    )
    assert len(buf) == len(collector_cached_buffer.buffer)

    # test buffer=None
    collector_default_buffer = Collector[CollectStats](policy, envs)
    collector_default_buffer.reset()
    result_default_buffer_collect_12_steps = collector_default_buffer.collect(n_step=12)
    for key in ["n_collected_episodes", "n_collected_steps", "returns", "lens"]:
        assert np.allclose(
            getattr(result_default_buffer_collect_12_steps, key),
            getattr(result_cached_buffer_collect_12_steps, key),
        )
    result2 = collector_default_buffer.collect(n_episode=9)
    for key in ["n_collected_episodes", "n_collected_steps", "returns", "lens"]:
        assert np.allclose(
            getattr(result2, key),
            getattr(result_cached_buffer_collect_9_episodes, key),
        )


@pytest.mark.skipif(envpool is None, reason="EnvPool doesn't support this platform")
def test_collector_envpool_gym_reset_return_info() -> None:
    envs = envpool.make_gymnasium("Pendulum-v1", num_envs=4, gym_reset_return_info=True)
    policy = MaxActionPolicy(action_shape=(len(envs), 1))

    c0 = Collector[CollectStats](
        policy,
        envs,
        VectorReplayBuffer(len(envs) * 10, len(envs)),
        exploration_noise=True,
    )
    c0.reset()
    c0.collect(n_step=8)
    env_ids = np.zeros(len(envs) * 10)
    env_ids[[0, 1, 10, 11, 20, 21, 30, 31]] = [0, 0, 1, 1, 2, 2, 3, 3]
    assert np.allclose(c0.buffer.info["env_id"], env_ids)


def test_collector_with_vector_env() -> None:
    env_fns = [lambda x=i: MoveToRightEnv(size=x, sleep=0) for i in [1, 8, 9, 10]]

    dum = DummyVectorEnv(env_fns)
    policy = MaxActionPolicy()

    c2 = Collector[CollectStats](
        policy,
        dum,
        VectorReplayBuffer(total_size=100, buffer_num=4),
    )

    c2.reset()

    c1r = c2.collect(n_episode=2)
    assert np.array_equal(np.array([1, 8]), c1r.lens)
    c2r = c2.collect(n_episode=10)
    assert np.array_equal(np.array([1, 1, 1, 1, 1, 1, 1, 8, 9, 10]), c2r.lens)
    c3r = c2.collect(n_step=20)
    assert np.array_equal(np.array([1, 1, 1, 1, 1]), c3r.lens)
    c4r = c2.collect(n_step=20)
    assert np.array_equal(np.array([1, 1, 1, 8, 1, 9, 1, 10]), c4r.lens)


def test_async_collector_with_vector_env() -> None:
    env_fns = [lambda x=i: MoveToRightEnv(size=x, sleep=0) for i in [1, 8, 9, 10]]

    dum = DummyVectorEnv(env_fns)
    policy = MaxActionPolicy()
    c1 = AsyncCollector(
        policy,
        dum,
        VectorReplayBuffer(total_size=100, buffer_num=4),
    )

    c1r = c1.collect(n_episode=10, reset_before_collect=True)
    assert np.array_equal(np.array([1, 1, 1, 1, 1, 1, 1, 1, 8, 1, 9]), c1r.lens)
    c2r = c1.collect(n_step=20)
    assert np.array_equal(np.array([1, 10, 1, 1, 1, 1]), c2r.lens)


class StepHookAddFieldToBatch(StepHook):
    def __call__(
        self,
        action_batch: CollectActionBatchProtocol,
        rollout_batch: RolloutBatchProtocol,
    ) -> None:
        rollout_batch.set_array_at_key(np.array([1]), "added_by_hook")


class TestCollectStatsAndHooks:
    @staticmethod
    def test_on_step_hook(collector_with_single_env: Collector) -> None:
        collector_with_single_env.set_on_step_hook(StepHookAddFieldToBatch())
        collect_stats = collector_with_single_env.collect(n_step=3)
        assert collect_stats.n_collected_steps == 3
        # a was added by the hook
        assert np.array_equal(
            collector_with_single_env.buffer[:].added_by_hook,
            np.array([1, 1, 1]),
        )

    @staticmethod
    def test_episode_mc_hook(collector_with_single_env: Collector) -> None:
        collector_with_single_env.set_on_episode_done_hook(EpisodeRolloutHookMCReturn())
        collector_with_single_env.collect(n_episode=1)
        collected_batch = collector_with_single_env.buffer[:]
        return_to_go = collected_batch.get(EpisodeRolloutHookMCReturn.MC_RETURN_TO_GO_KEY)
        full_return = collected_batch.get(EpisodeRolloutHookMCReturn.FULL_EPISODE_MC_RETURN_KEY)
        assert np.array_equal(return_to_go, episode_mc_return_to_go(collected_batch.rew))
        assert np.array_equal(full_return, np.ones(5) * return_to_go[0])
