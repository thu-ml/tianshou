import torch
import numpy as np
from timeit import timeit

from tianshou.policy import BasePolicy
from tianshou.data import Batch, ReplayBuffer, to_numpy


def compute_episodic_return_base(batch, gamma):
    returns = np.zeros_like(batch.rew)
    last = 0
    for i in reversed(range(len(batch.rew))):
        returns[i] = batch.rew[i]
        if not batch.done[i]:
            returns[i] += last * gamma
        last = returns[i]
    batch.returns = returns
    return batch


def test_episodic_returns(size=2560):
    fn = BasePolicy.compute_episodic_return
    batch = Batch(
        done=np.array([1, 0, 0, 1, 0, 1, 0, 1.]),
        rew=np.array([0, 1, 2, 3, 4, 5, 6, 7.]),
    )
    batch = fn(batch, None, gamma=.1, gae_lambda=1)
    ans = np.array([0, 1.23, 2.3, 3, 4.5, 5, 6.7, 7])
    assert np.allclose(batch.returns, ans)
    batch = Batch(
        done=np.array([0, 1, 0, 1, 0, 1, 0.]),
        rew=np.array([7, 6, 1, 2, 3, 4, 5.]),
    )
    batch = fn(batch, None, gamma=.1, gae_lambda=1)
    ans = np.array([7.6, 6, 1.2, 2, 3.4, 4, 5])
    assert np.allclose(batch.returns, ans)
    batch = Batch(
        done=np.array([0, 1, 0, 1, 0, 0, 1.]),
        rew=np.array([7, 6, 1, 2, 3, 4, 5.]),
    )
    batch = fn(batch, None, gamma=.1, gae_lambda=1)
    ans = np.array([7.6, 6, 1.2, 2, 3.45, 4.5, 5])
    assert np.allclose(batch.returns, ans)
    batch = Batch(
        done=np.array([0, 0, 0, 1., 0, 0, 0, 1, 0, 0, 0, 1]),
        rew=np.array([
            101, 102, 103., 200, 104, 105, 106, 201, 107, 108, 109, 202])
    )
    v = np.array([2., 3., 4, -1, 5., 6., 7, -2, 8., 9., 10, -3])
    ret = fn(batch, v, gamma=0.99, gae_lambda=0.95)
    returns = np.array([
        454.8344, 376.1143, 291.298, 200.,
        464.5610, 383.1085, 295.387, 201.,
        474.2876, 390.1027, 299.476, 202.])
    assert np.allclose(ret.returns, returns)
    if __name__ == '__main__':
        batch = Batch(
            done=np.random.randint(100, size=size) == 0,
            rew=np.random.random(size),
        )

        def vanilla():
            return compute_episodic_return_base(batch, gamma=.1)

        def optimized():
            return fn(batch, gamma=.1)

        cnt = 3000
        print('GAE vanilla', timeit(vanilla, setup=vanilla, number=cnt))
        print('GAE optim  ', timeit(optimized, setup=optimized, number=cnt))


def target_q_fn(buffer, indice):
    # return the next reward
    indice = (indice + 1 - buffer.done[indice]) % len(buffer)
    return torch.tensor(-buffer.rew[indice], dtype=torch.float32)


def compute_nstep_return_base(nstep, gamma, buffer, indice):
    returns = np.zeros_like(indice, dtype=np.float)
    buf_len = len(buffer)
    for i in range(len(indice)):
        flag, r = False, 0.
        for n in range(nstep):
            idx = (indice[i] + n) % buf_len
            r += buffer.rew[idx] * gamma ** n
            if buffer.done[idx]:
                flag = True
                break
        if not flag:
            idx = (indice[i] + nstep - 1) % buf_len
            r += to_numpy(target_q_fn(buffer, idx)) * gamma ** nstep
        returns[i] = r
    return returns


def test_nstep_returns(size=10000):
    buf = ReplayBuffer(10)
    for i in range(12):
        buf.add(obs=0, act=0, rew=i + 1, done=i % 4 == 3)
    batch, indice = buf.sample(0)
    assert np.allclose(indice, [2, 3, 4, 5, 6, 7, 8, 9, 0, 1])
    # rew:  [10, 11, 2, 3, 4, 5, 6, 7, 8, 9]
    # done: [ 0,  1, 0, 1, 0, 0, 0, 1, 0, 0]
    # test nstep = 1
    returns = to_numpy(BasePolicy.compute_nstep_return(
        batch, buf, indice, target_q_fn, gamma=.1, n_step=1).pop('returns'))
    assert np.allclose(returns, [2.6, 4, 4.4, 5.3, 6.2, 8, 8, 8.9, 9.8, 12])
    r_ = compute_nstep_return_base(1, .1, buf, indice)
    assert np.allclose(returns, r_), (r_, returns)
    # test nstep = 2
    returns = to_numpy(BasePolicy.compute_nstep_return(
        batch, buf, indice, target_q_fn, gamma=.1, n_step=2).pop('returns'))
    assert np.allclose(returns, [
        3.4, 4, 5.53, 6.62, 7.8, 8, 9.89, 10.98, 12.2, 12])
    r_ = compute_nstep_return_base(2, .1, buf, indice)
    assert np.allclose(returns, r_)
    # test nstep = 10
    returns = to_numpy(BasePolicy.compute_nstep_return(
        batch, buf, indice, target_q_fn, gamma=.1, n_step=10).pop('returns'))
    assert np.allclose(returns, [
        3.4, 4, 5.678, 6.78, 7.8, 8, 10.122, 11.22, 12.2, 12])
    r_ = compute_nstep_return_base(10, .1, buf, indice)
    assert np.allclose(returns, r_)

    if __name__ == '__main__':
        buf = ReplayBuffer(size)
        for i in range(int(size * 1.5)):
            buf.add(obs=0, act=0, rew=i + 1, done=np.random.randint(3) == 0)
        batch, indice = buf.sample(256)

        def vanilla():
            return compute_nstep_return_base(3, .1, buf, indice)

        def optimized():
            return BasePolicy.compute_nstep_return(
                batch, buf, indice, target_q_fn, gamma=.1, n_step=3)

        cnt = 3000
        print('nstep vanilla', timeit(vanilla, setup=vanilla, number=cnt))
        print('nstep optim  ', timeit(optimized, setup=optimized, number=cnt))


if __name__ == '__main__':
    test_nstep_returns()
    test_episodic_returns()
