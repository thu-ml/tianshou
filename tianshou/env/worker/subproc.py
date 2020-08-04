import ctypes
from collections import OrderedDict
from multiprocessing import Array, Pipe, connection
from multiprocessing.context import Process
from typing import Callable, Any, List, Tuple, Optional

import gym
import numpy as np

from tianshou.env.utils import CloudpickleWrapper
from tianshou.env.worker.base import EnvWorker


def _worker(parent, p, env_fn_wrapper, obs_bufs=None):
    def _encode_obs(obs, buffer):
        if isinstance(obs, np.ndarray):
            buffer.save(obs)
        elif isinstance(obs, tuple):
            for o, b in zip(obs, buffer):
                _encode_obs(o, b)
        elif isinstance(obs, dict):
            for k in obs.keys():
                _encode_obs(obs[k], buffer[k])
        return None
    parent.close()
    env = env_fn_wrapper.data()
    try:
        while True:
            cmd, data = p.recv()
            if cmd == 'step':
                obs, reward, done, info = env.step(data)
                if obs_bufs is not None:
                    obs = _encode_obs(obs, obs_bufs)
                p.send((obs, reward, done, info))
            elif cmd == 'reset':
                obs = env.reset()
                if obs_bufs is not None:
                    obs = _encode_obs(obs, obs_bufs)
                p.send(obs)
            elif cmd == 'close':
                p.send(env.close())
                p.close()
                break
            elif cmd == 'render':
                p.send(env.render(**data) if hasattr(env, 'render') else None)
            elif cmd == 'seed':
                p.send(env.seed(data) if hasattr(env, 'seed') else None)
            elif cmd == 'getattr':
                p.send(getattr(env, data) if hasattr(env, data) else None)
            else:
                p.close()
                raise NotImplementedError
    except KeyboardInterrupt:
        p.close()


_NP_TO_CT = {np.bool: ctypes.c_bool,
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
             np.float64: ctypes.c_double}


class ShArray:
    """Wrapper of multiprocessing Array"""

    def __init__(self, dtype, shape):
        self.arr = Array(_NP_TO_CT[dtype.type], int(np.prod(shape)))
        self.dtype = dtype
        self.shape = shape

    def save(self, ndarray):
        assert isinstance(ndarray, np.ndarray)
        dst = self.arr.get_obj()
        dst_np = np.frombuffer(dst, dtype=self.dtype).reshape(self.shape)
        np.copyto(dst_np, ndarray)

    def get(self):
        return np.frombuffer(self.arr.get_obj(),
                             dtype=self.dtype).reshape(self.shape)


class SubProcEnvWorker(EnvWorker):

    def __init__(self, env_fn: Callable[[], gym.Env], share_memory=False
                 ) -> None:
        super(SubProcEnvWorker, self).__init__(env_fn)
        self.parent_remote, self.child_remote = Pipe()
        self.share_memory = share_memory
        self.buffer = None
        if self.share_memory:
            dummy = env_fn()
            obs_space = dummy.observation_space
            dummy.close()
            del dummy
            self.buffer = SubProcEnvWorker._setup_buf(obs_space)
        args = (self.parent_remote, self.child_remote,
                CloudpickleWrapper(env_fn), self.buffer)
        self.process = Process(target=_worker, args=args, daemon=True)
        self.process.start()
        self.child_remote.close()
        self.closed = False

    @staticmethod
    def _setup_buf(space):
        if isinstance(space, gym.spaces.Dict):
            assert isinstance(space.spaces, OrderedDict)
            buffer = {k: SubProcEnvWorker._setup_buf(v)
                      for k, v in space.spaces.items()}
        elif isinstance(space, gym.spaces.Tuple):
            assert isinstance(space.spaces, tuple)
            buffer = tuple([SubProcEnvWorker._setup_buf(t)
                            for t in space.spaces])
        else:
            buffer = ShArray(space.dtype, space.shape)
        return buffer

    def _decode_obs(self, isNone):
        def decode_obs(buffer):
            if isinstance(buffer, ShArray):
                return buffer.get()
            elif isinstance(buffer, tuple):
                return tuple([decode_obs(b) for b in buffer])
            elif isinstance(buffer, dict):
                return {k: decode_obs(v) for k, v in buffer.items()}
            else:
                raise NotImplementedError
        return decode_obs(self.buffer)

    def render(self, **kwargs) -> None:
        self.parent_remote.send(['render', kwargs])
        return self.parent_remote.recv()

    def close(self) -> Any:
        if self.closed:
            return []
        self.parent_remote.send(['close', None])
        result = self.parent_remote.recv()
        self.closed = True
        self.process.join()
        return result

    @staticmethod
    def wait(workers: List['SubProcEnvWorker']) -> List['SubProcEnvWorker']:
        conns = [x.parent_remote for x in workers]
        ready_conns = connection.wait(conns)
        return [workers[conns.index(con)] for con in ready_conns]

    def __getattr__(self, key: str):
        self.parent_remote.send(['getattr', key])
        return self.parent_remote.recv()

    def step(self, action: np.ndarray
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.parent_remote.send(['step', action])
        obs, rew, done, info = self.parent_remote.recv()
        if self.share_memory:
            obs = self._decode_obs(obs)
        return obs, rew, done, info

    def reset(self):
        self.parent_remote.send(['reset', None])
        obs = self.parent_remote.recv()
        if self.share_memory:
            obs = self._decode_obs(obs)
        return obs

    def seed(self, seed: Optional[int] = None):
        self.parent_remote.send(['seed', seed])
        return self.parent_remote.recv()
