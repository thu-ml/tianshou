import sys
import time

import gym
import numpy as np
import pytest
from gym.spaces.discrete import Discrete

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
from tianshou.utils import RunningMeanStd

if __name__ == "__main__":
    from env import MyTestEnv, NXEnv
else:  # pytest
    from test.base.env import MyTestEnv, NXEnv

try:
    import envpool
except ImportError:
    envpool = None


def has_ray():
    try:
        import ray  # noqa: F401
        return True
    except ImportError:
        return False


def recurse_comp(a, b):
    try:
        if isinstance(a, np.ndarray):
            if a.dtype == object:
                return np.array([recurse_comp(m, n) for m, n in zip(a, b)]).all()
            else:
                return np.allclose(a, b)
        elif isinstance(a, (list, tuple)):
            return np.array([recurse_comp(m, n) for m, n in zip(a, b)]).all()
        elif isinstance(a, dict):
            return np.array([recurse_comp(a[k], b[k]) for k in a.keys()]).all()
    except (Exception):
        return False


def test_async_env(size=10000, num=8, sleep=0.1):
    # simplify the test case, just keep stepping
    env_fns = [
        lambda i=i: MyTestEnv(size=i, sleep=sleep, random_sleep=True)
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
            A, B, C, D = v.step(action=act, id=env_ids)
            b = Batch({"obs": A, "rew": B, "done": C, "info": D})
            env_ids = b.info.env_id
            o.append(b)
            current_idx_start += len(act)
            # len of action may be smaller than len(A) in the end
            act = action_list[current_idx_start:current_idx_start + len(A)]
            # truncate env_ids with the first terms
            # typically len(env_ids) == len(A) == len(action), except for the
            # last batch when actions are not enough
            env_ids = env_ids[:len(act)]
        spent_time = time.time() - spent_time
        Batch.cat(o)
        v.close()
        # assure 1/7 improvement
        if sys.platform == "linux" and cls != RayVectorEnv:
            # macOS/Windows cannot pass this check
            assert spent_time < 6.0 * sleep * num / (num + 1)


def test_async_check_id(size=100, num=4, sleep=.2, timeout=.7):
    env_fns = [
        lambda: MyTestEnv(size=size, sleep=sleep * 2),
        lambda: MyTestEnv(size=size, sleep=sleep * 3),
        lambda: MyTestEnv(size=size, sleep=sleep * 5),
        lambda: MyTestEnv(size=size, sleep=sleep * 7)
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
            _, _, _, info = v.step([1] * len(ids), ids)
            t = time.time() - t
            ids = Batch(info).env_id
            print(ids, t)
            if not (
                len(ids) == len(res) and np.allclose(sorted(ids), res) and
                (t < timeout) == (len(res) == num - 1)
            ):
                pass_check = 0
                break
        total_pass += pass_check
    if sys.platform == "linux":  # Windows/macOS may not pass this check
        assert total_pass >= 2


def test_vecenv(size=10, num=8, sleep=0.001):
    env_fns = [
        lambda i=i: MyTestEnv(size=i, sleep=sleep, recurse_state=True)
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
    o = [v.reset() for v in venv]
    for a in action_list:
        o = []
        for v in venv:
            A, B, C, D = v.step([a] * num)
            if sum(C):
                A = v.reset(np.where(C)[0])
            o.append([A, B, C, D])
        for index, infos in enumerate(zip(*o)):
            if index == 3:  # do not check info here
                continue
            for info in infos:
                assert recurse_comp(infos[0], info)

    if __name__ == "__main__":
        t = [0] * len(venv)
        for i, e in enumerate(venv):
            t[i] = time.time()
            e.reset()
            for a in action_list:
                done = e.step([a] * num)[2]
                if sum(done) > 0:
                    e.reset(np.where(done)[0])
            t[i] = time.time() - t[i]
        for i, v in enumerate(venv):
            print(f"{type(v)}: {t[i]:.6f}s")

    def assert_get(v, expected):
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


def test_attr_unwrapped():
    train_envs = DummyVectorEnv([lambda: gym.make("CartPole-v1")])
    train_envs.set_env_attr("test_attribute", 1337)
    assert train_envs.get_env_attr("test_attribute") == [1337]
    assert hasattr(train_envs.workers[0].env, "test_attribute")
    assert hasattr(train_envs.workers[0].env.unwrapped, "test_attribute")


def test_env_obs_dtype():
    for obs_type in ["array", "object"]:
        envs = SubprocVectorEnv(
            [lambda i=x, t=obs_type: NXEnv(i, t) for x in [5, 10, 15, 20]]
        )
        obs = envs.reset()
        assert obs.dtype == object
        obs = envs.step([1, 1, 1, 1])[0]
        assert obs.dtype == object


def test_env_reset_optional_kwargs(size=10000, num=8):
    env_fns = [lambda i=i: MyTestEnv(size=i) for i in range(size, size + num)]
    test_cls = [DummyVectorEnv, SubprocVectorEnv, ShmemVectorEnv]
    if has_ray():
        test_cls += [RayVectorEnv]
    for cls in test_cls:
        v = cls(env_fns, wait_num=num // 2, timeout=1e-3)
        _, info = v.reset(seed=1, return_info=True)
        assert len(info) == len(env_fns)
        assert isinstance(info[0], dict)


def test_venv_wrapper_gym(num_envs: int = 4):
    # Issue 697
    envs = DummyVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(num_envs)])
    envs = VectorEnvNormObs(envs)
    obs_ref = envs.reset(return_info=False)
    obs, info = envs.reset(return_info=True)
    assert isinstance(obs_ref, np.ndarray)
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, list)
    assert isinstance(info[0], dict)
    assert obs_ref.shape[0] == obs.shape[0] == len(info) == num_envs


def run_align_norm_obs(raw_env, train_env, test_env, action_list):
    eps = np.finfo(np.float32).eps.item()
    raw_obs, train_obs = [raw_env.reset()], [train_env.reset()]
    for action in action_list:
        obs, rew, done, info = raw_env.step(action)
        raw_obs.append(obs)
        if np.any(done):
            raw_obs.append(raw_env.reset(np.where(done)[0]))
        obs, rew, done, info = train_env.step(action)
        train_obs.append(obs)
        if np.any(done):
            train_obs.append(train_env.reset(np.where(done)[0]))
    ref_rms = RunningMeanStd()
    for ro, to in zip(raw_obs, train_obs):
        ref_rms.update(ro)
        no = (ro - ref_rms.mean) / np.sqrt(ref_rms.var + eps)
        assert np.allclose(no, to)
    assert np.allclose(ref_rms.mean, train_env.get_obs_rms().mean)
    assert np.allclose(ref_rms.var, train_env.get_obs_rms().var)
    assert np.allclose(ref_rms.mean, test_env.get_obs_rms().mean)
    assert np.allclose(ref_rms.var, test_env.get_obs_rms().var)
    test_obs = [test_env.reset()]
    for action in action_list:
        obs, rew, done, info = test_env.step(action)
        test_obs.append(obs)
        if np.any(done):
            test_obs.append(test_env.reset(np.where(done)[0]))
    for ro, to in zip(raw_obs, test_obs):
        no = (ro - ref_rms.mean) / np.sqrt(ref_rms.var + eps)
        assert np.allclose(no, to)


def test_venv_norm_obs():
    sizes = np.array([5, 10, 15, 20])
    action = np.array([1, 1, 1, 1])
    total_step = 30
    action_list = [action] * total_step
    env_fns = [lambda i=x: MyTestEnv(size=i, array_state=True) for x in sizes]
    raw = DummyVectorEnv(env_fns)
    train_env = VectorEnvNormObs(DummyVectorEnv(env_fns))
    print(train_env.observation_space)
    test_env = VectorEnvNormObs(DummyVectorEnv(env_fns), update_obs_rms=False)
    test_env.set_obs_rms(train_env.get_obs_rms())
    run_align_norm_obs(raw, train_env, test_env, action_list)


def test_gym_wrappers():

    class DummyEnv(gym.Env):

        def __init__(self):
            self.action_space = gym.spaces.Box(
                low=-1.0, high=2.0, shape=(4, ), dtype=np.float32
            )

    bsz = 10
    action_per_branch = [4, 6, 10, 7]
    env = DummyEnv()
    original_act = env.action_space.high
    # convert continous to multidiscrete action space
    # with different action number per dimension
    env_m = ContinuousToDiscrete(env, action_per_branch)
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
    # check conversion is working properly for one action
    np.testing.assert_allclose(
        env_d.action(env_d.action_space.n - 1), env_m.action_space.nvec - 1
    )
    # check conversion is working properly for a batch of actions
    np.testing.assert_allclose(
        env_d.action(np.array([env_d.action_space.n - 1] * bsz)),
        np.array([env_m.action_space.nvec - 1] * bsz),
    )


@pytest.mark.skipif(envpool is None, reason="EnvPool doesn't support this platform")
def test_venv_wrapper_envpool():
    raw = envpool.make_gym("Ant-v3", num_envs=4)
    train = VectorEnvNormObs(envpool.make_gym("Ant-v3", num_envs=4))
    test = VectorEnvNormObs(
        envpool.make_gym("Ant-v3", num_envs=4), update_obs_rms=False
    )
    test.set_obs_rms(train.get_obs_rms())
    actions = [
        np.array([raw.action_space.sample() for _ in range(4)]) for i in range(30)
    ]
    run_align_norm_obs(raw, train, test, actions)


@pytest.mark.skipif(envpool is None, reason="EnvPool doesn't support this platform")
def test_venv_wrapper_envpool_gym_reset_return_info():
    num_envs = 4
    env = VectorEnvNormObs(
        envpool.make_gym("Ant-v3", num_envs=num_envs, gym_reset_return_info=True)
    )
    obs, info = env.reset()
    assert obs.shape[0] == num_envs
    for _, v in info.items():
        if not isinstance(v, dict):
            assert v.shape[0] == num_envs


if __name__ == "__main__":
    test_venv_norm_obs()
    test_venv_wrapper_gym()
    test_venv_wrapper_envpool()
    test_venv_wrapper_envpool_gym_reset_return_info()
    test_env_obs_dtype()
    test_vecenv()
    test_attr_unwrapped()
    test_async_env()
    test_async_check_id()
    test_env_reset_optional_kwargs()
    test_gym_wrappers()
