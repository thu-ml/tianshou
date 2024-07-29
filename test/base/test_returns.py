from typing import cast

import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer, to_numpy
from tianshou.data.types import RolloutBatchProtocol
from tianshou.policy import BasePolicy


def compute_episodic_return_base(batch: Batch, gamma: float) -> Batch:
    returns = np.zeros_like(batch.rew)
    last = 0
    for i in reversed(range(len(batch.rew))):
        returns[i] = batch.rew[i]
        if not batch.done[i]:
            returns[i] += last * gamma
        last = returns[i]
    batch.returns = returns
    return batch


def test_episodic_returns(size: int = 2560) -> None:
    fn = BasePolicy.compute_episodic_return
    buf = ReplayBuffer(20)
    batch = cast(
        RolloutBatchProtocol,
        Batch(
            terminated=np.array([1, 0, 0, 1, 0, 0, 0, 1.0]),
            truncated=np.array([0, 0, 0, 0, 0, 1, 0, 0]),
            rew=np.array([0, 1, 2, 3, 4, 5, 6, 7.0]),
            info=Batch(
                {
                    "TimeLimit.truncated": np.array(
                        [False, False, False, False, False, True, False, False],
                    ),
                },
            ),
        ),
    )
    for b in iter(batch):
        b.obs = b.act = 1  # type: ignore[assignment]
        buf.add(b)
    returns, _ = fn(batch, buf, buf.sample_indices(0), gamma=0.1, gae_lambda=1)
    ans = np.array([0, 1.23, 2.3, 3, 4.5, 5, 6.7, 7])
    assert np.allclose(returns, ans)
    buf.reset()
    batch = cast(
        RolloutBatchProtocol,
        Batch(
            terminated=np.array([0, 1, 0, 1, 0, 1, 0.0]),
            truncated=np.array([0, 0, 0, 0, 0, 0, 0.0]),
            rew=np.array([7, 6, 1, 2, 3, 4, 5.0]),
        ),
    )
    for b in iter(batch):
        b.obs = b.act = 1  # type: ignore[assignment]
        buf.add(b)
    returns, _ = fn(batch, buf, buf.sample_indices(0), gamma=0.1, gae_lambda=1)
    ans = np.array([7.6, 6, 1.2, 2, 3.4, 4, 5])
    assert np.allclose(returns, ans)
    buf.reset()
    batch = cast(
        RolloutBatchProtocol,
        Batch(
            terminated=np.array([0, 1, 0, 1, 0, 0, 1.0]),
            truncated=np.array([0, 0, 0, 0, 0, 0, 0]),
            rew=np.array([7, 6, 1, 2, 3, 4, 5.0]),
        ),
    )
    for b in iter(batch):
        b.obs = b.act = 1  # type: ignore[assignment]
        buf.add(b)
    returns, _ = fn(batch, buf, buf.sample_indices(0), gamma=0.1, gae_lambda=1)
    ans = np.array([7.6, 6, 1.2, 2, 3.45, 4.5, 5])
    assert np.allclose(returns, ans)
    buf.reset()
    batch = cast(
        RolloutBatchProtocol,
        Batch(
            terminated=np.array([0, 0, 0, 1.0, 0, 0, 0, 1, 0, 0, 0, 1]),
            truncated=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            rew=np.array([101, 102, 103.0, 200, 104, 105, 106, 201, 107, 108, 109, 202]),
        ),
    )
    for b in batch:
        b.obs = b.act = 1  # type: ignore[assignment]
        buf.add(b)
    v = np.array([2.0, 3.0, 4, -1, 5.0, 6.0, 7, -2, 8.0, 9.0, 10, -3])
    returns, _ = fn(batch, buf, buf.sample_indices(0), v, gamma=0.99, gae_lambda=0.95)
    ground_truth = np.array(
        [
            454.8344,
            376.1143,
            291.298,
            200.0,
            464.5610,
            383.1085,
            295.387,
            201.0,
            474.2876,
            390.1027,
            299.476,
            202.0,
        ],
    )
    assert np.allclose(returns, ground_truth)
    buf.reset()
    batch = cast(
        RolloutBatchProtocol,
        Batch(
            terminated=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
            truncated=np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]),
            rew=np.array([101, 102, 103.0, 200, 104, 105, 106, 201, 107, 108, 109, 202]),
            info=Batch(
                {
                    "TimeLimit.truncated": np.array(
                        [
                            False,
                            False,
                            False,
                            True,
                            False,
                            False,
                            False,
                            True,
                            False,
                            False,
                            False,
                            False,
                        ],
                    ),
                },
            ),
        ),
    )
    for b in iter(batch):
        b.obs = b.act = 1  # type: ignore[assignment]
        buf.add(b)
    v = np.array([2.0, 3.0, 4, -1, 5.0, 6.0, 7, -2, 8.0, 9.0, 10, -3])
    returns, _ = fn(batch, buf, buf.sample_indices(0), v, gamma=0.99, gae_lambda=0.95)
    ground_truth = np.array(
        [
            454.0109,
            375.2386,
            290.3669,
            199.01,
            462.9138,
            381.3571,
            293.5248,
            199.02,
            474.2876,
            390.1027,
            299.476,
            202.0,
        ],
    )
    assert np.allclose(returns, ground_truth)


def target_q_fn(buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
    # return the next reward
    indices = buffer.next(indices)
    return torch.tensor(-buffer.rew[indices], dtype=torch.float32)


def target_q_fn_multidim(buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
    return target_q_fn(buffer, indices).unsqueeze(1).repeat(1, 51)


def compute_nstep_return_base(
    nstep: int,
    gamma: float,
    buffer: ReplayBuffer,
    indices: np.ndarray,
) -> np.ndarray:
    returns = np.zeros_like(indices, dtype=float)
    buf_len = len(buffer)
    for i in range(len(indices)):
        flag, rew = False, 0.0
        real_step_n = nstep
        for n in range(nstep):
            idx = (indices[i] + n) % buf_len
            rew += buffer.rew[idx] * gamma**n
            if buffer.done[idx]:
                if not (hasattr(buffer, "info") and buffer.info["TimeLimit.truncated"][idx]):
                    flag = True
                real_step_n = n + 1
                break
        if not flag:
            idx = (indices[i] + real_step_n - 1) % buf_len
            rew += to_numpy(target_q_fn(buffer, idx)) * gamma**real_step_n
        returns[i] = rew
    return returns


def test_nstep_returns(size: int = 10000) -> None:
    buf = ReplayBuffer(10)
    for i in range(12):
        buf.add(
            cast(
                RolloutBatchProtocol,
                Batch(
                    obs=0,
                    act=0,
                    rew=i + 1,
                    terminated=i % 4 == 3,
                    truncated=False,
                ),
            ),
        )
    batch, indices = buf.sample(0)
    assert np.allclose(indices, [2, 3, 4, 5, 6, 7, 8, 9, 0, 1])
    # rew:  [11, 12, 3, 4, 5, 6, 7, 8, 9, 10]
    # done: [ 0,  1, 0, 1, 0, 0, 0, 1, 0, 0]
    # test nstep = 1
    returns = to_numpy(
        BasePolicy.compute_nstep_return(batch, buf, indices, target_q_fn, gamma=0.1, n_step=1)
        .pop("returns")
        .reshape(-1),
    )
    assert np.allclose(returns, [2.6, 4, 4.4, 5.3, 6.2, 8, 8, 8.9, 9.8, 12])
    r_ = compute_nstep_return_base(1, 0.1, buf, indices)
    assert np.allclose(returns, r_), (r_, returns)
    returns_multidim = to_numpy(
        BasePolicy.compute_nstep_return(
            batch,
            buf,
            indices,
            target_q_fn_multidim,
            gamma=0.1,
            n_step=1,
        ).pop("returns"),
    )
    assert np.allclose(returns_multidim, returns[:, np.newaxis])
    # test nstep = 2
    returns = to_numpy(
        BasePolicy.compute_nstep_return(batch, buf, indices, target_q_fn, gamma=0.1, n_step=2)
        .pop("returns")
        .reshape(-1),
    )
    assert np.allclose(returns, [3.4, 4, 5.53, 6.62, 7.8, 8, 9.89, 10.98, 12.2, 12])
    r_ = compute_nstep_return_base(2, 0.1, buf, indices)
    assert np.allclose(returns, r_)
    returns_multidim = to_numpy(
        BasePolicy.compute_nstep_return(
            batch,
            buf,
            indices,
            target_q_fn_multidim,
            gamma=0.1,
            n_step=2,
        ).pop("returns"),
    )
    assert np.allclose(returns_multidim, returns[:, np.newaxis])
    # test nstep = 10
    returns = to_numpy(
        BasePolicy.compute_nstep_return(batch, buf, indices, target_q_fn, gamma=0.1, n_step=10)
        .pop("returns")
        .reshape(-1),
    )
    assert np.allclose(returns, [3.4, 4, 5.678, 6.78, 7.8, 8, 10.122, 11.22, 12.2, 12])
    r_ = compute_nstep_return_base(10, 0.1, buf, indices)
    assert np.allclose(returns, r_)
    returns_multidim = to_numpy(
        BasePolicy.compute_nstep_return(
            batch,
            buf,
            indices,
            target_q_fn_multidim,
            gamma=0.1,
            n_step=10,
        ).pop("returns"),
    )
    assert np.allclose(returns_multidim, returns[:, np.newaxis])


def test_nstep_returns_with_timelimit(size: int = 10000) -> None:
    buf = ReplayBuffer(10)
    for i in range(12):
        buf.add(
            cast(
                RolloutBatchProtocol,
                Batch(
                    obs=0,
                    act=0,
                    rew=i + 1,
                    terminated=i % 4 == 3 and i != 3,
                    truncated=i == 3,
                    info={"TimeLimit.truncated": i == 3},
                ),
            ),
        )
    batch, indices = buf.sample(0)
    assert np.allclose(indices, [2, 3, 4, 5, 6, 7, 8, 9, 0, 1])
    # rew:  [11, 12, 3, 4, 5, 6, 7, 8, 9, 10]
    # done: [ 0,  1, 0, 1, 0, 0, 0, 1, 0, 0]
    # test nstep = 1
    returns = to_numpy(
        BasePolicy.compute_nstep_return(batch, buf, indices, target_q_fn, gamma=0.1, n_step=1)
        .pop("returns")
        .reshape(-1),
    )
    assert np.allclose(returns, [2.6, 3.6, 4.4, 5.3, 6.2, 8, 8, 8.9, 9.8, 12])
    r_ = compute_nstep_return_base(1, 0.1, buf, indices)
    assert np.allclose(returns, r_), (r_, returns)
    returns_multidim = to_numpy(
        BasePolicy.compute_nstep_return(
            batch,
            buf,
            indices,
            target_q_fn_multidim,
            gamma=0.1,
            n_step=1,
        ).pop("returns"),
    )
    assert np.allclose(returns_multidim, returns[:, np.newaxis])
    # test nstep = 2
    returns = to_numpy(
        BasePolicy.compute_nstep_return(batch, buf, indices, target_q_fn, gamma=0.1, n_step=2)
        .pop("returns")
        .reshape(-1),
    )
    assert np.allclose(returns, [3.36, 3.6, 5.53, 6.62, 7.8, 8, 9.89, 10.98, 12.2, 12])
    r_ = compute_nstep_return_base(2, 0.1, buf, indices)
    assert np.allclose(returns, r_)
    returns_multidim = to_numpy(
        BasePolicy.compute_nstep_return(
            batch,
            buf,
            indices,
            target_q_fn_multidim,
            gamma=0.1,
            n_step=2,
        ).pop("returns"),
    )
    assert np.allclose(returns_multidim, returns[:, np.newaxis])
    # test nstep = 10
    returns = to_numpy(
        BasePolicy.compute_nstep_return(batch, buf, indices, target_q_fn, gamma=0.1, n_step=10)
        .pop("returns")
        .reshape(-1),
    )
    assert np.allclose(returns, [3.36, 3.6, 5.678, 6.78, 7.8, 8, 10.122, 11.22, 12.2, 12])
    r_ = compute_nstep_return_base(10, 0.1, buf, indices)
    assert np.allclose(returns, r_)
    returns_multidim = to_numpy(
        BasePolicy.compute_nstep_return(
            batch,
            buf,
            indices,
            target_q_fn_multidim,
            gamma=0.1,
            n_step=10,
        ).pop("returns"),
    )
    assert np.allclose(returns_multidim, returns[:, np.newaxis])
