import ctypes
import multiprocessing
import time
from collections.abc import Callable
from multiprocessing import connection
from multiprocessing.context import BaseContext
from typing import Any, Literal

import gymnasium as gym
import numpy as np

from tianshou.env.utils import CloudpickleWrapper, gym_new_venv_step_type
from tianshou.env.worker import EnvWorker

# mypy: disable-error-code="unused-ignore"


_NP_TO_CT = {
    np.bool_: ctypes.c_bool,
    np.uint8: ctypes.c_uint8,
    np.uint16: ctypes.c_uint16,
    np.uint32: ctypes.c_uint32,
    np.uint64: ctypes.c_uint64,
    np.int8: ctypes.c_int8,
    np.int16: ctypes.c_int16,
    np.int32: ctypes.c_int32,
    np.int64: ctypes.c_int64,
    np.float32: ctypes.c_float,
    np.float64: ctypes.c_double,
}


class ShArray:
    """Wrapper of multiprocessing Array.

    Example usage:

    ::

        import numpy as np
        import multiprocessing as mp
        from tianshou.env.worker.subproc import ShArray
        ctx = mp.get_context('fork')  # set an explicit context
        arr = ShArray(np.dtype(np.float32), (2, 3), ctx)
        arr.save(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
        print(arr.get())

    """

    def __init__(self, dtype: np.generic, shape: tuple[int], ctx: BaseContext | None) -> None:
        if ctx is None:
            ctx = multiprocessing.get_context()
        self.arr = ctx.Array(_NP_TO_CT[dtype.type], int(np.prod(shape)))  # type: ignore
        self.dtype = dtype
        self.shape = shape

    def save(self, ndarray: np.ndarray) -> None:
        assert isinstance(ndarray, np.ndarray)
        dst = self.arr.get_obj()
        dst_np = np.frombuffer(dst, dtype=self.dtype).reshape(self.shape)  # type: ignore
        np.copyto(dst_np, ndarray)

    def get(self) -> np.ndarray:
        obj = self.arr.get_obj()
        return np.frombuffer(obj, dtype=self.dtype).reshape(self.shape)  # type: ignore


def _setup_buf(space: gym.Space, ctx: BaseContext) -> dict | tuple | ShArray:
    if isinstance(space, gym.spaces.Dict):
        return {k: _setup_buf(v, ctx) for k, v in space.spaces.items()}
    if isinstance(space, gym.spaces.Tuple):
        assert isinstance(space.spaces, tuple)
        return tuple([_setup_buf(t, ctx) for t in space.spaces])
    return ShArray(space.dtype, space.shape, ctx)  # type: ignore


def _worker(
    parent: connection.Connection,
    p: connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
    obs_bufs: dict | tuple | ShArray | None = None,
) -> None:
    def _encode_obs(
        obs: dict | tuple | np.ndarray,
        buffer: dict | tuple | ShArray,
    ) -> None:
        if isinstance(buffer, ShArray):
            # if buffer is an ShArray, obs must be array-like
            obs = np.asarray(obs, dtype=buffer.dtype)
            buffer.save(obs)
        elif isinstance(obs, tuple) and isinstance(buffer, tuple):
            for o, b in zip(obs, buffer, strict=True):
                _encode_obs(o, b)
        elif isinstance(obs, dict) and isinstance(buffer, dict):
            for k in obs:
                _encode_obs(obs[k], buffer[k])

    parent.close()
    env = env_fn_wrapper.data()
    try:
        while True:
            try:
                cmd, data = p.recv()
            except EOFError:  # the pipe has been closed
                p.close()
                break
            if cmd == "step":
                env_return = env.step(data)
                if obs_bufs is not None:
                    _encode_obs(env_return[0], obs_bufs)
                    env_return = (None, *env_return[1:])
                p.send(env_return)
            elif cmd == "reset":
                obs, info = env.reset(**data)
                if obs_bufs is not None:
                    _encode_obs(obs, obs_bufs)
                    obs = None
                p.send((obs, info))
            elif cmd == "close":
                p.send(env.close())
                p.close()
                break
            elif cmd == "render":
                p.send(env.render(**data) if hasattr(env, "render") else None)
            elif cmd == "seed":
                if hasattr(env, "seed"):
                    p.send(env.seed(data))
                else:
                    env.action_space.seed(seed=data)
                    env.reset(seed=data)
                    p.send(None)
            elif cmd == "getattr":
                p.send(getattr(env, data) if hasattr(env, data) else None)
            elif cmd == "setattr":
                setattr(env.unwrapped, data["key"], data["value"])
            else:
                p.close()
                raise NotImplementedError
    except KeyboardInterrupt:
        p.close()


class SubprocEnvWorker(EnvWorker):
    """Subprocess worker used in SubprocVectorEnv and ShmemVectorEnv."""

    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        share_memory: bool = False,
        context: BaseContext | Literal["fork", "spawn"] | None = None,
    ) -> None:
        if not isinstance(context, BaseContext):
            context = multiprocessing.get_context(context)
        self.parent_remote, self.child_remote = context.Pipe()
        self.share_memory = share_memory
        self.buffer: dict | tuple | ShArray | None = None
        assert hasattr(context, "Process")  # for mypy
        if self.share_memory:
            dummy = env_fn()
            obs_space = dummy.observation_space
            dummy.close()
            del dummy
            self.buffer = _setup_buf(obs_space, context)
        args = (
            self.parent_remote,
            self.child_remote,
            CloudpickleWrapper(env_fn),
            self.buffer,
        )
        self.process = context.Process(target=_worker, args=args, daemon=True)
        self.process.start()
        self.child_remote.close()
        super().__init__(env_fn)

    def get_env_attr(self, key: str) -> Any:
        self.parent_remote.send(["getattr", key])
        return self.parent_remote.recv()

    def set_env_attr(self, key: str, value: Any) -> None:
        self.parent_remote.send(["setattr", {"key": key, "value": value}])

    def _decode_obs(self) -> dict | tuple | np.ndarray:
        def decode_obs(
            buffer: dict | tuple | ShArray | None,
        ) -> dict | tuple | np.ndarray:
            if isinstance(buffer, ShArray):
                return buffer.get()
            if isinstance(buffer, tuple):
                return tuple([decode_obs(b) for b in buffer])
            if isinstance(buffer, dict):
                return {k: decode_obs(v) for k, v in buffer.items()}
            raise NotImplementedError

        return decode_obs(self.buffer)

    @staticmethod
    def wait(  # type: ignore
        workers: list["SubprocEnvWorker"],
        wait_num: int,
        timeout: float | None = None,
    ) -> list["SubprocEnvWorker"]:
        remain_conns = conns = [x.parent_remote for x in workers]
        ready_conns: list[connection.Connection] = []
        remain_time, t1 = timeout, time.time()
        while len(remain_conns) > 0 and len(ready_conns) < wait_num:
            if timeout:
                remain_time = timeout - (time.time() - t1)
                if remain_time <= 0:
                    break
            # connection.wait hangs if the list is empty
            new_ready_conns = connection.wait(remain_conns, timeout=remain_time)  # type: ignore
            ready_conns.extend(new_ready_conns)  # type: ignore
            remain_conns = [conn for conn in remain_conns if conn not in ready_conns]  # type: ignore
        return [workers[conns.index(con)] for con in ready_conns]  # type: ignore

    def send(self, action: np.ndarray | None, **kwargs: Any) -> None:
        if action is None:
            if "seed" in kwargs:
                super().seed(kwargs["seed"])
            self.parent_remote.send(["reset", kwargs])
        else:
            self.parent_remote.send(["step", action])

    def recv(self) -> gym_new_venv_step_type | tuple[np.ndarray, dict]:
        result = self.parent_remote.recv()
        if isinstance(result, tuple):
            if len(result) == 2:
                obs, info = result
                if self.share_memory:
                    obs = self._decode_obs()
                return obs, info
            obs = result[0]
            if self.share_memory:
                obs = self._decode_obs()
            # TODO: figure out the typing issue, simplify and document this method
            return (obs, *result[1:])
        obs = result
        if self.share_memory:
            obs = self._decode_obs()
        return obs

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict]:
        if "seed" in kwargs:
            super().seed(kwargs["seed"])
        self.parent_remote.send(["reset", kwargs])

        result = self.parent_remote.recv()
        if isinstance(result, tuple):
            obs, info = result
            if self.share_memory:
                obs = self._decode_obs()
            return obs, info
        obs = result
        if self.share_memory:
            obs = self._decode_obs()
        return obs

    def seed(self, seed: int | None = None) -> list[int] | None:
        super().seed(seed)
        self.parent_remote.send(["seed", seed])
        return self.parent_remote.recv()

    def render(self, **kwargs: Any) -> Any:
        self.parent_remote.send(["render", kwargs])
        return self.parent_remote.recv()

    def close_env(self) -> None:
        try:
            self.parent_remote.send(["close", None])
            # mp may be deleted so it may raise AttributeError
            self.parent_remote.recv()
            self.process.join()
        except (BrokenPipeError, EOFError, AttributeError):
            pass
        # ensure the subproc is terminated
        self.process.terminate()
