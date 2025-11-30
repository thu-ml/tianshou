import sys
import time
from collections.abc import Callable
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.spaces.discrete import Discrete

from test.base.env import MoveToRightEnv, NXEnv
from tianshou.data import Batch
from tianshou.env import (
    ContinuousToDiscrete,
    DummyVectorEnv,
    MultiDiscreteToDiscrete,
    RayVectorEnv,
    ShmemVectorEnv,
    SubprocVectorEnv,
    VectorEnvNormObs,
)
from tianshou.env.gym_wrappers import TruncatedAsTerminated
from tianshou.env.venvs import BaseVectorEnv
from tianshou.utils import RunningMeanStd

try:
    import envpool
except ImportError:
    envpool = None


def has_ray() -> bool:
    try:
        import ray  # noqa: F401

        return True
    except ImportError:
        return False


def recurse_comp(a: np.ndarray | list | tuple | dict, b: Any) -> np.bool_ | bool | None:
    try:
        if isinstance(a, np.ndarray):
            if a.dtype == object:
                return np.array([recurse_comp(m, n) for m, n in zip(a, b, strict=True)]).all()
            return np.allclose(a, b)
        if isinstance(a, list | tuple):
            return np.array([recurse_comp(m, n) for m, n in zip(a, b, strict=True)]).all()
        if isinstance(a, dict):
            return np.array([recurse_comp(a[k], b[k]) for k in a]).all()
    except Exception:
        return False


def test_async_env(size: int = 10000, num: int = 8, sleep: float = 0.1) -> None:
    # simplify the test case, just keep stepping
    env_fns = [
        lambda i=i: MoveToRightEnv(size=i, sleep=sleep, random_sleep=True)
        for i in range(size, size + num)
    ]
    test_cls = [SubprocVectorEnv, ShmemVectorEnv]
    if has_ray():
        test_cls += [RayVectorEnv]
    for cls in test_cls:
        v = cls(env_fns, wait_num=num // 2, timeout=1e-3)
        v.seed(None)
        v.reset()
        # for a random variable u ~ U[0, 1], let v = max{u1, u2, ..., un}
        # P(v <= x) = x^n (0 <= x <= 1), pdf of v is nx^{n-1}
        # expectation of v is n / (n + 1)
        # for a synchronous environment, the following actions should take
        # about 7 * sleep * num / (num + 1) seconds
        # for async simulation, the analysis is complicated, but the time cost
        # should be smaller
        action_list = [1] * num + [0] * (num * 2) + [1] * (num * 4)
        current_idx_start = 0
        act = action_list[:num]
        env_ids = list(range(num))
        o = []
        spent_time = time.time()
        while current_idx_start < len(action_list):
            (
                A,
                B,
                C,
                D,
                E,
            ) = v.step(action=act, id=env_ids)
            b = Batch({"obs": A, "rew": B, "terminate": C, "truncated": D, "info": E})
            env_ids = b.info.env_id
            o.append(b)
            current_idx_start += len(act)
            # len of action may be smaller than len(A) in the end
            act = action_list[current_idx_start : current_idx_start + len(A)]
            # truncate env_ids with the first terms
            # typically len(env_ids) == len(A) == len(action), except for the
            # last batch when actions are not enough
            env_ids = env_ids[: len(act)]
        spent_time = time.time() - spent_time
        Batch.cat(o)
        v.close()
        # assure 1/7 improvement
        if sys.platform == "linux" and cls != RayVectorEnv:
            # macOS/Windows cannot pass this check
            assert spent_time < 6.0 * sleep * num / (num + 1)


def test_async_check_id(
    size: int = 100,
    num: int = 4,
    sleep: float = 0.2,
    timeout: float = 0.7,
) -> None:
    env_fns = [
        lambda: MoveToRightEnv(size=size, sleep=sleep * 2),
        lambda: MoveToRightEnv(size=size, sleep=sleep * 3),
        lambda: MoveToRightEnv(size=size, sleep=sleep * 5),
        lambda: MoveToRightEnv(size=size, sleep=sleep * 7),
    ]
    test_cls = [SubprocVectorEnv, ShmemVectorEnv]
    if has_ray():
        test_cls += [RayVectorEnv]
    total_pass = 0
    for cls in test_cls:
        pass_check = 1
        v = cls(env_fns, wait_num=num - 1, timeout=timeout)
        t = time.time()
        v.reset()
        t = time.time() - t
        print(f"{cls} reset {t}")
        if t > sleep * 9:  # huge than maximum sleep time (7 sleep)
            pass_check = 0
        expect_result = [
            [0, 1],
            [0, 1, 2],
            [0, 1, 3],
            [0, 1, 2],
            [0, 1],
            [0, 2, 3],
            [0, 1],
        ]
        ids = np.arange(num)
        for res in expect_result:
            t = time.time()
            _, _, _, _, info = v.step([1] * len(ids), ids)
            t = time.time() - t
            ids = Batch(info).env_id
            print(ids, t)
            if not (
                len(ids) == len(res)
                and np.allclose(sorted(ids), res)
                and (t < timeout) == (len(res) == num - 1)
            ):
                pass_check = 0
                break
        total_pass += pass_check
    if sys.platform == "linux":  # Windows/macOS may not pass this check
        assert total_pass >= 2


def test_vecenv(size: int = 10, num: int = 8, sleep: float = 0.001) -> None:
    env_fns = [
        lambda i=i: MoveToRightEnv(size=i, sleep=sleep, recurse_state=True)
        for i in range(size, size + num)
    ]
    venv = [
        DummyVectorEnv(env_fns),
        SubprocVectorEnv(env_fns),
        ShmemVectorEnv(env_fns),
    ]
    if has_ray() and sys.platform == "linux":
        venv += [RayVectorEnv(env_fns)]
    for v in venv:
        v.seed(0)
    action_list = [1] * 5 + [0] * 10 + [1] * 20
    for a in action_list:
        o = []
        for v in venv:
            A, B, C, D, E = v.step(np.array([a] * num))
            if sum(C + D):
                A, _ = v.reset(np.where(C + D)[0])
            o.append([A, B, C, D, E])
        for index, infos in enumerate(zip(*o, strict=True)):
            if index == 4:  # do not check info here
                continue
            for info in infos:
                assert recurse_comp(infos[0], info)

    def assert_get(v: BaseVectorEnv, expected: list) -> None:
        assert v.get_env_attr("size") == expected
        assert v.get_env_attr("size", id=0) == [expected[0]]
        assert v.get_env_attr("size", id=[0, 1, 2]) == expected[:3]

    for v in venv:
        assert_get(v, list(range(size, size + num)))
        assert v.env_num == num
        assert v.action_space == [Discrete(2)] * num

        v.set_env_attr("size", 0)
        assert_get(v, [0] * num)

        v.set_env_attr("size", 1, 0)
        assert_get(v, [1] + [0] * (num - 1))

        v.set_env_attr("size", 2, [1, 2, 3])
        assert_get(v, [1] + [2] * 3 + [0] * (num - 4))

    for v in venv:
        v.close()


def test_attr_unwrapped() -> None:
    training_envs = DummyVectorEnv([lambda: gym.make("CartPole-v1")])
    training_envs.set_env_attr("test_attribute", 1337)
    assert training_envs.get_env_attr("test_attribute") == [1337]
    assert hasattr(training_envs.workers[0].env.unwrapped, "test_attribute")  # type: ignore


def test_env_obs_dtype() -> None:
    def create_env(i: int, t: str) -> Callable[[], NXEnv]:
        return lambda: NXEnv(i, t)

    for obs_type in ["array", "object"]:
        envs = SubprocVectorEnv([create_env(x, obs_type) for x in [5, 10, 15, 20]])
        obs, info = envs.reset()
        assert obs.dtype == object
        obs = envs.step(np.array([1, 1, 1, 1]))[0]
        assert obs.dtype == object


def test_env_reset_optional_kwargs(size: int = 10000, num: int = 8) -> None:
    env_fns = [lambda i=i: MoveToRightEnv(size=i) for i in range(size, size + num)]
    test_cls = [DummyVectorEnv, SubprocVectorEnv, ShmemVectorEnv]
    if has_ray():
        test_cls += [RayVectorEnv]
    for cls in test_cls:
        v = cls(env_fns, wait_num=num // 2, timeout=1e-3)
        _, info = v.reset(seed=1)
        assert len(info) == len(env_fns)
        assert isinstance(info[0], dict)


def test_venv_wrapper_gym(num_envs: int = 4) -> None:
    # Issue 697
    envs = DummyVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(num_envs)])
    envs = VectorEnvNormObs(envs)
    try:
        obs, info = envs.reset()
    except ValueError:
        obs, info = envs.reset(return_info=True)
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, np.ndarray)
    assert isinstance(info[0], dict)
    assert obs.shape[0] == len(info) == num_envs


def run_align_norm_obs(
    raw_env: DummyVectorEnv,
    train_env: VectorEnvNormObs,
    test_env: VectorEnvNormObs,
    action_list: list[np.ndarray],
) -> None:
    def reset_result_to_obs(
        reset_result: tuple[np.ndarray, dict | list[dict]],
    ) -> np.ndarray:
        """Extract observation from reset result (result is possibly a tuple containing info)."""
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, _ = reset_result
        else:
            obs = reset_result  # type: ignore
        return obs

    eps = np.finfo(np.float32).eps.item()
    raw_reset_result = raw_env.reset()
    train_reset_result = train_env.reset()
    initial_raw_obs = reset_result_to_obs(raw_reset_result)  # type: ignore
    initial_train_obs = reset_result_to_obs(train_reset_result)  # type: ignore
    raw_obs, train_obs = [initial_raw_obs], [initial_train_obs]
    for action in action_list:
        step_result = raw_env.step(action)
        if len(step_result) == 5:
            obs, rew, terminated, truncated, info = step_result
            done = np.logical_or(terminated, truncated)
        else:
            obs, rew, done, info = step_result  # type: ignore
        raw_obs.append(obs)
        if np.any(done):
            reset_result = raw_env.reset(np.where(done)[0])
            obs = reset_result_to_obs(reset_result)  # type: ignore
            raw_obs.append(obs)
        step_result = train_env.step(action)
        if len(step_result) == 5:
            obs, rew, terminated, truncated, info = step_result
            done = np.logical_or(terminated, truncated)
        else:
            obs, rew, done, info = step_result  # type: ignore
        train_obs.append(obs)
        if np.any(done):
            reset_result = train_env.reset(np.where(done)[0])
            obs = reset_result_to_obs(reset_result)  # type: ignore
            train_obs.append(obs)
    ref_rms = RunningMeanStd()
    for ro, to in zip(raw_obs, train_obs, strict=True):
        ref_rms.update(ro)
        no = (ro - ref_rms.mean) / np.sqrt(ref_rms.var + eps)
        assert np.allclose(no, to)
    assert np.allclose(ref_rms.mean, train_env.get_obs_rms().mean)
    assert np.allclose(ref_rms.var, train_env.get_obs_rms().var)
    assert np.allclose(ref_rms.mean, test_env.get_obs_rms().mean)
    assert np.allclose(ref_rms.var, test_env.get_obs_rms().var)
    reset_result = test_env.reset()
    obs = reset_result_to_obs(reset_result)  # type: ignore
    test_obs = [obs]
    for action in action_list:
        step_result = test_env.step(action)
        if len(step_result) == 5:
            obs, rew, terminated, truncated, info = step_result
            done = np.logical_or(terminated, truncated)
        else:
            obs, rew, done, info = step_result  # type: ignore
        test_obs.append(obs)
        if np.any(done):
            reset_result = test_env.reset(np.where(done)[0])
            obs = reset_result_to_obs(reset_result)  # type: ignore
            test_obs.append(obs)
    for ro, to in zip(raw_obs, test_obs, strict=True):
        no = (ro - ref_rms.mean) / np.sqrt(ref_rms.var + eps)
        assert np.allclose(no, to)


def test_venv_norm_obs() -> None:
    sizes = np.array([5, 10, 15, 20])
    action = np.array([1, 1, 1, 1])
    total_step = 30
    action_list = [action] * total_step
    env_fns = [lambda i=x: MoveToRightEnv(size=i, array_state=True) for x in sizes]
    raw = DummyVectorEnv(env_fns)
    train_env = VectorEnvNormObs(DummyVectorEnv(env_fns))
    print(train_env.observation_space)
    test_env = VectorEnvNormObs(DummyVectorEnv(env_fns), update_obs_rms=False)
    test_env.set_obs_rms(train_env.get_obs_rms())
    run_align_norm_obs(raw, train_env, test_env, action_list)


def test_gym_wrappers() -> None:
    class DummyEnv(gym.Env):
        def __init__(self) -> None:
            self.action_space = gym.spaces.Box(low=-1.0, high=2.0, shape=(4,), dtype=np.float32)
            self.observation_space = gym.spaces.Discrete(2)

        def step(self, act: Any) -> tuple[Any, Literal[-1], Literal[False], Literal[True], dict]:
            return self.observation_space.sample(), -1, False, True, {}

    bsz = 10
    action_per_branch = [4, 6, 10, 7]
    env = DummyEnv()
    assert isinstance(env.action_space, gym.spaces.Box)
    original_act = env.action_space.high
    # convert continous to multidiscrete action space
    # with different action number per dimension
    env_m = ContinuousToDiscrete(env, action_per_branch)
    assert isinstance(env_m.action_space, gym.spaces.MultiDiscrete)
    # check conversion is working properly for one action
    np.testing.assert_allclose(env_m.action(env_m.action_space.nvec - 1), original_act)
    # check conversion is working properly for a batch of actions
    np.testing.assert_allclose(
        env_m.action(np.array([env_m.action_space.nvec - 1] * bsz)),
        np.array([original_act] * bsz),
    )
    # convert multidiscrete with different action number per
    # dimension to discrete action space
    env_d = MultiDiscreteToDiscrete(env_m)
    assert isinstance(env_d.action_space, gym.spaces.Discrete)
    # check conversion is working properly for one action
    np.testing.assert_allclose(
        env_d.action(np.array(env_d.action_space.n - 1)),
        env_m.action_space.nvec - 1,
    )
    # check conversion is working properly for a batch of actions
    np.testing.assert_allclose(
        env_d.action(np.array([env_d.action_space.n - 1] * bsz)),
        np.array([env_m.action_space.nvec - 1] * bsz),
    )
    # check truncate is True when terminated
    try:
        env_t = TruncatedAsTerminated(env)
    except OSError:
        env_t = None
    if env_t is not None:
        _, _, truncated, _, _ = env_t.step(env_t.action_space.sample())
        assert truncated


# TODO: old gym envs are no longer supported! Replace by Ant-v4 and fix assoticiated tests
@pytest.mark.skipif(envpool is None, reason="EnvPool doesn't support this platform")
def test_venv_wrapper_envpool() -> None:
    raw = envpool.make_gymnasium("Ant-v3", num_envs=4)
    train = VectorEnvNormObs(envpool.make_gymnasium("Ant-v3", num_envs=4))
    test = VectorEnvNormObs(envpool.make_gymnasium("Ant-v3", num_envs=4), update_obs_rms=False)
    test.set_obs_rms(train.get_obs_rms())
    actions = [np.array([raw.action_space.sample() for _ in range(4)]) for i in range(30)]
    run_align_norm_obs(raw, train, test, actions)


@pytest.mark.skipif(envpool is None, reason="EnvPool doesn't support this platform")
def test_venv_wrapper_envpool_gym_reset_return_info() -> None:
    num_envs = 4
    env = VectorEnvNormObs(
        envpool.make_gymnasium("Ant-v3", num_envs=num_envs, gym_reset_return_info=True),
    )
    obs, info = env.reset()
    assert obs.shape[0] == num_envs
    # This is not actually unreachable b/c envpool does not return info in the right format
    if isinstance(info, dict):  # type: ignore[unreachable]
        for _, v in info.items():  # type: ignore[unreachable]
            if not isinstance(v, dict):
                assert v.shape[0] == num_envs
    else:
        for _info in info:
            for _, v in _info.items():
                if not isinstance(v, dict):
                    assert v.shape[0] == num_envs
