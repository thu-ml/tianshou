import gym
import ctypes
import numpy as np
from collections import OrderedDict
from multiprocessing import Pipe, Process, Array
from typing import Callable, List, Optional, Tuple, Union


from tianshou.env import BaseVectorEnv, SubprocVectorEnv
from tianshou.env.utils import CloudpickleWrapper

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


def _shmem_worker(parent, p, env_fn_wrapper, obs_bufs):
    """Control a single environment instance using IPC and shared memory."""
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
                p.send((_encode_obs(obs, obs_bufs), reward, done, info))
            elif cmd == 'reset':
                p.send(_encode_obs(env.reset(), obs_bufs))
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


class ShmemVectorEnv(SubprocVectorEnv):
    """Optimized version of SubprocVectorEnv that uses shared variables to
    communicate observations. SubprocVectorEnv has exactly the same API as
    SubprocVectorEnv.

    .. seealso::

        Please refer to :class:`~tianshou.env.SubprocVectorEnv` for more
        detailed explanation.

    ShmemVectorEnv Class was inspired by openai baseline's implementation.
    Please refer to 'https://github.com/openai/baselines/blob/master/baselines/
    common/vec_env/shmem_vec_env.py' for more info if you are interested.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]]) -> None:
        BaseVectorEnv.__init__(self, env_fns)
        # Mind that SubprocVectorEnv is not initialised.
        self.closed = False
        dummy = env_fns[0]()
        obs_space = dummy.observation_space
        dummy.close()
        del dummy
        self.obs_bufs = [ShmemVectorEnv._setup_buf(obs_space)
                         for _ in range(self.env_num)]
        self.parent_remote, self.child_remote = \
            zip(*[Pipe() for _ in range(self.env_num)])
        self.processes = [
            Process(target=_shmem_worker, args=(
                parent, child, CloudpickleWrapper(env_fn),
                obs_buf), daemon=True)
            for (parent, child, env_fn, obs_buf) in zip(
                self.parent_remote, self.child_remote, env_fns, self.obs_bufs)
        ]
        for p in self.processes:
            p.start()
        for c in self.child_remote:
            c.close()

    def step(self,
             action: np.ndarray,
             id: Optional[Union[int, List[int]]] = None
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if id is None:
            id = range(self.env_num)
        elif np.isscalar(id):
            id = [id]
        assert len(action) == len(id)
        for i, j in enumerate(id):
            self.parent_remote[j].send(['step', action[i]])
        result = []
        for i in id:
            obs, rew, done, info = self.parent_remote[i].recv()
            obs = self._decode_obs(obs, i)
            result.append((obs, rew, done, info))
        obs, rew, done, info = map(np.stack, zip(*result))
        return obs, rew, done, info

    def reset(self, id: Optional[Union[int, List[int]]] = None) -> np.ndarray:
        if id is None:
            id = range(self.env_num)
        elif np.isscalar(id):
            id = [id]
        for i in id:
            self.parent_remote[i].send(['reset', None])
        obs = np.stack(
            [self._decode_obs(self.parent_remote[i].recv(), i) for i in id])
        return obs

    @staticmethod
    def _setup_buf(space):
        if isinstance(space, gym.spaces.Dict):
            assert isinstance(space.spaces, OrderedDict)
            buffer = {k: ShmemVectorEnv._setup_buf(v)
                      for k, v in space.spaces.items()}
        elif isinstance(space, gym.spaces.Tuple):
            assert isinstance(space.spaces, tuple)
            buffer = tuple([ShmemVectorEnv._setup_buf(t)
                            for t in space.spaces])
        else:
            buffer = ShArray(space.dtype, space.shape)
        return buffer

    def _decode_obs(self, isNone, index):
        def decode_obs(buffer):
            if isinstance(buffer, ShArray):
                return buffer.get()
            elif isinstance(buffer, tuple):
                return tuple([decode_obs(b) for b in buffer])
            elif isinstance(buffer, dict):
                return {k: decode_obs(v) for k, v in buffer.items()}
            else:
                raise NotImplementedError
        assert isNone is None
        return decode_obs(self.obs_bufs[index])
