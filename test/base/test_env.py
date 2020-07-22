import time
import numpy as np
from gym.spaces.discrete import Discrete
from tianshou.data import Batch
from tianshou.env import VectorEnv, SubprocVectorEnv, \
    RayVectorEnv, AsyncVectorEnv

if __name__ == '__main__':
    from env import MyTestEnv
else:  # pytest
    from test.base.env import MyTestEnv


def test_async_env(num=8, sleep=0.1):
    # simplify the test case, just keep stepping
    size = 10000
    env_fns = [
        lambda i=i: MyTestEnv(size=i, sleep=sleep, random_sleep=True)
        for i in range(size, size + num)
    ]
    v = AsyncVectorEnv(env_fns, wait_num=num // 2)
    v.seed()
    v.reset()
    # for a random variable u ~ U[0, 1], let v = max{u1, u2, ..., un}
    # P(v <= x) = x^n (0 <= x <= 1), pdf of v is nx^{n-1}
    # expectation of v is n / (n + 1)
    # for a synchronous environment, the following actions should take
    # about 7 * sleep * num / (num + 1) seconds
    # for AsyncVectorEnv, the analysis is complicated, but the time cost
    # should be smaller
    action_list = [1] * num + [0] * (num * 2) + [1] * (num * 4)
    current_index_start = 0
    action = action_list[:num]
    o = []
    spent_time = time.time()
    while current_index_start < len(action_list):
        A, B, C, D = v.step(action)
        o.append(Batch({'obs': A, 'rew': B, 'done': C, 'info': D}))
        current_index_start += len(action)
        action = action_list[current_index_start: current_index_start + len(A)]
    spent_time = time.time() - spent_time
    data = Batch.cat(o)
    # assure 1/7 improvement
    assert spent_time < 6.0 * sleep * num / (num + 1)
    return spent_time, data


def test_vecenv(size=10, num=8, sleep=0.001):
    verbose = __name__ == '__main__'
    env_fns = [
        lambda i=i: MyTestEnv(size=i, sleep=sleep)
        for i in range(size, size + num)
    ]
    venv = [
        VectorEnv(env_fns),
        SubprocVectorEnv(env_fns),
    ]
    if verbose:
        venv.append(RayVectorEnv(env_fns))
    for v in venv:
        v.seed()
    action_list = [1] * 5 + [0] * 10 + [1] * 20
    if not verbose:
        o = [v.reset() for v in venv]
        for i, a in enumerate(action_list):
            o = []
            for v in venv:
                A, B, C, D = v.step([a] * num)
                if sum(C):
                    A = v.reset(np.where(C)[0])
                o.append([A, B, C, D])
            for i in zip(*o):
                for j in range(1, len(i) - 1):
                    assert (i[0] == i[j]).all()
    else:
        t = [0, 0, 0]
        for i, e in enumerate(venv):
            t[i] = time.time()
            e.reset()
            for a in action_list:
                done = e.step([a] * num)[2]
                if sum(done) > 0:
                    e.reset(np.where(done)[0])
            t[i] = time.time() - t[i]
        print(f'VectorEnv: {t[0]:.6f}s')
        print(f'SubprocVectorEnv: {t[1]:.6f}s')
        print(f'RayVectorEnv: {t[2]:.6f}s')
    for v in venv:
        assert v.size == list(range(size, size + num))
        assert v.env_num == num
        assert v.action_space == [Discrete(2)] * num

    for v in venv:
        v.close()


if __name__ == '__main__':
    test_vecenv()
    test_async_env()
