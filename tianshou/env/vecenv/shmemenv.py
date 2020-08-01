import ctypes
from collections import OrderedDict
from multiprocessing import Pipe, Process, Array
from typing import Callable, List, Optional, Tuple, Union

import gym
import numpy as np

from tianshou.env import BaseVectorEnv, SubprocVectorEnv
from tianshou.env.utils import CloudpickleWrapper

_NP_TO_CT = {np.float64: ctypes.c_double,
             np.float32: ctypes.c_float,
             np.int32: ctypes.c_int32,
             np.int8: ctypes.c_int8,
             np.uint8: ctypes.c_char,
             np.bool: ctypes.c_bool}


def _shmem_worker(parent, p, env_fn_wrapper, obs_bufs,
                  obs_shapes, obs_dtypes, keys):
    """Control a single environment instance using IPC and
    shared memory.
    """
    def _encode_obs(maybe_dict):
        flatdict = maybe_dict if isinstance(maybe_dict, dict) else {
            None: maybe_dict}
        for k in keys:
            dst = obs_bufs[k].get_obj()
            dst_np = np.frombuffer(dst, dtype=obs_dtypes[k]).reshape(
                obs_shapes[k])
            np.copyto(dst_np, flatdict[k])
        return None

    parent.close()
    env = env_fn_wrapper.data()
    try:
        while True:
            cmd, data = p.recv()
            if cmd == 'step':
                obs, reward, done, info = env.step(data)
                p.send((_encode_obs(obs), reward, done, info))
            elif cmd == 'reset':
                p.send(_encode_obs(env.reset()))
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


class ShmemVectorEnv(SubprocVectorEnv):
    """Optimized version of SubprocVectorEnv that uses shared
    variables to communicate observations.

    .. seealso::

        Please refer to :class:`~tianshou.env.BaseVectorEnv` for more detailed
        explanation.

    I borrow heavily from openai baseline to implement ShmemVectorEnv Class.
    Please refer to 'https://github.com/openai/baselines/blob/master/baselines/
    common/vec_env/shmem_vec_env.py' for more info if you are interested.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]]) -> None:
        BaseVectorEnv.__init__(self, env_fns)
        self.closed = False
        self._setup_obs_space(env_fns[0])
        self.obs_bufs = [
            {k: Array(_NP_TO_CT[self.obs_dtypes[k].type], int(
                np.prod(self.obs_shapes[k]))) for k in self.obs_keys}
            for _ in range(self.env_num)]
        self.parent_remote, self.child_remote = \
            zip(*[Pipe() for _ in range(self.env_num)])
        self.processes = [
            Process(target=_shmem_worker, args=(
                parent, child, CloudpickleWrapper(env_fn),
                obs_buf, self.obs_shapes,
                self.obs_dtypes, self.obs_keys), daemon=True)
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

    def _setup_obs_space(self, env):
        dummy = env()
        obs_space = dummy.observation_space
        dummy.close()
        del dummy
        if isinstance(obs_space, gym.spaces.Dict):
            assert isinstance(obs_space.spaces, OrderedDict)
            subspaces = obs_space.spaces
        elif isinstance(obs_space, gym.spaces.Tuple):
            assert isinstance(obs_space.spaces, tuple)
            subspaces = {i: obs_space.spaces[i]
                         for i in range(len(obs_space.spaces))}
        else:
            subspaces = {None: obs_space}
        self.obs_keys = []
        self.obs_shapes = {}
        self.obs_dtypes = {}
        for key, box in subspaces.items():
            self.obs_keys.append(key)
            self.obs_shapes[key] = box.shape
            self.obs_dtypes[key] = box.dtype

    def _decode_obs(self, isNone, index):
        assert isNone is None
        result = {}
        for k in self.obs_keys:
            result[k] = np.frombuffer(
                self.obs_bufs[index][k].get_obj(),
                dtype=self.obs_dtypes[k]).reshape(self.obs_shapes[k])
        return result if not result.keys() == {None} else result[None]
