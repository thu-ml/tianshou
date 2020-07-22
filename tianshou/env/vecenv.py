import gym
import numpy as np
from multiprocessing import Process, Pipe, connection
from typing import List, Tuple, Union, Optional, Callable, Any

try:
    import ray
except ImportError:
    pass

from tianshou.env import BaseVectorEnv
from tianshou.env.utils import CloudpickleWrapper


class VectorEnv(BaseVectorEnv):
    """Dummy vectorized environment wrapper, implemented in for-loop.

    .. seealso::

        Please refer to :class:`~tianshou.env.BaseVectorEnv` for more detailed
        explanation.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]]) -> None:
        super().__init__(env_fns)
        self.envs = [_() for _ in env_fns]

    def __getattr__(self, key):
        return [getattr(env, key) if hasattr(env, key) else None
                for env in self.envs]

    def reset(self, id: Optional[Union[int, List[int]]] = None) -> np.ndarray:
        if id is None:
            id = range(self.env_num)
        elif np.isscalar(id):
            id = [id]
        obs = np.stack([self.envs[i].reset() for i in id])
        return obs

    def step(self,
             action: np.ndarray,
             id: Optional[Union[int, List[int]]] = None
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if id is None:
            id = range(self.env_num)
        elif np.isscalar(id):
            id = [id]
        assert len(action) == len(id)
        result = [self.envs[i].step(action[i]) for i in id]
        obs, rew, done, info = map(np.stack, zip(*result))
        return obs, rew, done, info

    def seed(self, seed: Optional[Union[int, List[int]]] = None) -> List[int]:
        if np.isscalar(seed):
            seed = [seed + _ for _ in range(self.env_num)]
        elif seed is None:
            seed = [seed] * self.env_num
        result = []
        for e, s in zip(self.envs, seed):
            if hasattr(e, 'seed'):
                result.append(e.seed(s))
        return result

    def render(self, **kwargs) -> List[Any]:
        result = []
        for e in self.envs:
            if hasattr(e, 'render'):
                result.append(e.render(**kwargs))
        return result

    def close(self) -> List[Any]:
        return [e.close() for e in self.envs]


def worker(parent, p, env_fn_wrapper):
    parent.close()
    env = env_fn_wrapper.data()
    try:
        while True:
            cmd, data = p.recv()
            if cmd == 'step':
                p.send(env.step(data))
            elif cmd == 'reset':
                p.send(env.reset())
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


class SubprocVectorEnv(BaseVectorEnv):
    """Vectorized environment wrapper based on subprocess.

    .. seealso::

        Please refer to :class:`~tianshou.env.BaseVectorEnv` for more detailed
        explanation.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]]) -> None:
        super().__init__(env_fns)
        self.closed = False
        self.parent_remote, self.child_remote = \
            zip(*[Pipe() for _ in range(self.env_num)])
        self.processes = [
            Process(target=worker, args=(
                parent, child, CloudpickleWrapper(env_fn)), daemon=True)
            for (parent, child, env_fn) in zip(
                self.parent_remote, self.child_remote, env_fns)
        ]
        for p in self.processes:
            p.start()
        for c in self.child_remote:
            c.close()

    def __getattr__(self, key):
        for p in self.parent_remote:
            p.send(['getattr', key])
        return [p.recv() for p in self.parent_remote]

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
        result = [self.parent_remote[i].recv() for i in id]
        obs, rew, done, info = map(np.stack, zip(*result))
        return obs, rew, done, info

    def reset(self, id: Optional[Union[int, List[int]]] = None) -> np.ndarray:
        if id is None:
            id = range(self.env_num)
        elif np.isscalar(id):
            id = [id]
        for i in id:
            self.parent_remote[i].send(['reset', None])
        obs = np.stack([self.parent_remote[i].recv() for i in id])
        return obs

    def seed(self, seed: Optional[Union[int, List[int]]] = None) -> List[int]:
        if np.isscalar(seed):
            seed = [seed + _ for _ in range(self.env_num)]
        elif seed is None:
            seed = [seed] * self.env_num
        for p, s in zip(self.parent_remote, seed):
            p.send(['seed', s])
        return [p.recv() for p in self.parent_remote]

    def render(self, **kwargs) -> List[Any]:
        for p in self.parent_remote:
            p.send(['render', kwargs])
        return [p.recv() for p in self.parent_remote]

    def close(self) -> List[Any]:
        if self.closed:
            return []
        for p in self.parent_remote:
            p.send(['close', None])
        result = [p.recv() for p in self.parent_remote]
        self.closed = True
        for p in self.processes:
            p.join()
        return result


class AsyncVectorEnv(SubprocVectorEnv):
    """Vectorized asynchronous environment wrapper based on subprocess.
    .. seealso::
        Please refer to :class:`~tianshou.env.BaseVectorEnv` for more detailed
        explanation.
    """
    def __init__(self, env_fns: List[Callable[[], gym.Env]],
                 wait_num: Optional[int] = None) -> None:
        """
        :param wait_num: used in asynchronous simulation if the time cost of
            ``env.step`` varies with time and synchronously waiting for all
            environments to finish a step is time-wasting. In that case, we
            can return when ``wait_num`` environments finish a step and keep
            on simulation in these environments. If ``None``, asynchronous
            simulation is disabled; else, ``1 <= wait_num <= env_num``.
        """
        super().__init__(env_fns)
        self.wait_num = wait_num or len(env_fns)
        assert 1 <= self.wait_num <= len(env_fns), \
            f'wait_num should be in [1, {len(env_fns)}], but got {wait_num}'
        self.waiting_conn = []
        self.waiting_id = []

    def reset(self, id: Optional[Union[int, List[int]]] = None) -> np.ndarray:
        if id is None:
            id = range(self.env_num)
        elif np.isscalar(id):
            id = [id]
        # reset envs are not waiting anymore
        rest_envs = [(i, conn) for i, conn in
                     zip(self.waiting_id, self.waiting_conn) if i not in id]
        if not rest_envs:
            self.waiting_id, self.waiting_conn = [], []
        else:
            self.waiting_id, self.waiting_conn = zip(*rest_envs)
        return super().reset(id)

    def render(self, **kwargs) -> List[Any]:
        if len(self.waiting_id) > 0:
            raise RuntimeError(
                f"environments {self.waiting_id} are still "
                f"stepping, cannot render now")
        return super().render(**kwargs)

    def close(self) -> List[Any]:
        if self.closed:
            return []
        # finish remaining steps, and close
        self.step(None)
        return super().close()

    def step(self,
             action: Optional[np.ndarray],
             id: Optional[Union[int, List[int]]] = None
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Provide the given action to the environments specified by the id.
        If action is None, fetch unfinished step() calls instead.
        """
        if id is not None:
            raise ValueError("cannot specify the id of environments"
                             " during step() for AsyncVectorEnv")
        if action is not None:
            assert len(action) + len(self.waiting_id) <= self.env_num
            available_env_ids = [i for i in range(self.env_num)
                                 if i not in self.waiting_id]
            for i, act in enumerate(action):
                i = available_env_ids[i]
                self.parent_remote[i].send(['step', act])
                self.waiting_conn.append(self.parent_remote[i])
                self.waiting_id.append(i)
        result = []
        while len(self.waiting_conn) > 0 and len(result) < self.wait_num:
            ready_conns = connection.wait(self.waiting_conn)
            for conn in ready_conns:
                waiting_index = self.waiting_conn.index(conn)
                self.waiting_conn.pop(waiting_index)
                env_id = self.waiting_id.pop(waiting_index)
                ans = conn.recv()
                obs, rew, done, info = ans
                info["env_id"] = env_id
                result.append((obs, rew, done, info))
        obs, rew, done, info = map(np.stack, zip(*result))
        return obs, rew, done, info


class RayVectorEnv(BaseVectorEnv):
    """Vectorized environment wrapper based on
    `ray <https://github.com/ray-project/ray>`_. However, according to our
    test, it is about two times slower than
    :class:`~tianshou.env.SubprocVectorEnv`.

    .. seealso::

        Please refer to :class:`~tianshou.env.BaseVectorEnv` for more detailed
        explanation.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]]) -> None:
        super().__init__(env_fns)
        try:
            if not ray.is_initialized():
                ray.init()
        except NameError:
            raise ImportError(
                'Please install ray to support RayVectorEnv: pip3 install ray')
        self.envs = [
            ray.remote(gym.Wrapper).options(num_cpus=0).remote(e())
            for e in env_fns]

    def __getattr__(self, key):
        return ray.get([e.__getattr__.remote(key) for e in self.envs])

    def step(self,
             action: np.ndarray,
             id: Optional[Union[int, List[int]]] = None
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if id is None:
            id = range(self.env_num)
        elif np.isscalar(id):
            id = [id]
        assert len(action) == len(id)
        result = ray.get([self.envs[j].step.remote(action[i])
                          for i, j in enumerate(id)])
        obs, rew, done, info = map(np.stack, zip(*result))
        return obs, rew, done, info

    def reset(self, id: Optional[Union[int, List[int]]] = None) -> np.ndarray:
        if id is None:
            id = range(self.env_num)
        elif np.isscalar(id):
            id = [id]
        obs = np.stack(ray.get([self.envs[i].reset.remote() for i in id]))
        return obs

    def seed(self, seed: Optional[Union[int, List[int]]] = None) -> List[int]:
        if not hasattr(self.envs[0], 'seed'):
            return []
        if np.isscalar(seed):
            seed = [seed + _ for _ in range(self.env_num)]
        elif seed is None:
            seed = [seed] * self.env_num
        return ray.get([e.seed.remote(s) for e, s in zip(self.envs, seed)])

    def render(self, **kwargs) -> List[Any]:
        if not hasattr(self.envs[0], 'render'):
            return [None for e in self.envs]
        return ray.get([e.render.remote(**kwargs) for e in self.envs])

    def close(self) -> List[Any]:
        return ray.get([e.close.remote() for e in self.envs])
