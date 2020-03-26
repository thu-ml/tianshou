import numpy as np
from abc import ABC, abstractmethod
from multiprocessing import Process, Pipe

try:
    import ray
except ImportError:
    pass

from tianshou.env import EnvWrapper, CloudpickleWrapper


class BaseVectorEnv(ABC):
    def __init__(self, env_fns):
        self._env_fns = env_fns
        self.env_num = len(env_fns)

    def __len__(self):
        return self.env_num

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def seed(self, seed=None):
        pass

    @abstractmethod
    def render(self, **kwargs):
        pass

    @abstractmethod
    def close(self):
        pass


class VectorEnv(BaseVectorEnv):
    """docstring for VectorEnv"""

    def __init__(self, env_fns):
        super().__init__(env_fns)
        self.envs = [_() for _ in env_fns]

    def reset(self, id=None):
        if id is None:
            self._obs = np.stack([e.reset() for e in self.envs])
        else:
            if np.isscalar(id):
                id = [id]
            for i in id:
                self._obs[i] = self.envs[i].reset()
        return self._obs

    def step(self, action):
        assert len(action) == self.env_num
        result = [e.step(a) for e, a in zip(self.envs, action)]
        self._obs, self._rew, self._done, self._info = zip(*result)
        self._obs = np.stack(self._obs)
        self._rew = np.stack(self._rew)
        self._done = np.stack(self._done)
        self._info = np.stack(self._info)
        return self._obs, self._rew, self._done, self._info

    def seed(self, seed=None):
        if np.isscalar(seed):
            seed = [seed + _ for _ in range(self.env_num)]
        elif seed is None:
            seed = [seed] * self.env_num
        result = []
        for e, s in zip(self.envs, seed):
            if hasattr(e, 'seed'):
                result.append(e.seed(s))
        return result

    def render(self, **kwargs):
        result = []
        for e in self.envs:
            if hasattr(e, 'render'):
                result.append(e.render(**kwargs))
        return result

    def close(self):
        for e in self.envs:
            e.close()


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
                p.close()
                break
            elif cmd == 'render':
                p.send(env.render(**data) if hasattr(env, 'render') else None)
            elif cmd == 'seed':
                p.send(env.seed(data) if hasattr(env, 'seed') else None)
            else:
                p.close()
                raise NotImplementedError
    except KeyboardInterrupt:
        p.close()


class SubprocVectorEnv(BaseVectorEnv):
    """docstring for SubProcVectorEnv"""

    def __init__(self, env_fns):
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

    def step(self, action):
        assert len(action) == self.env_num
        for p, a in zip(self.parent_remote, action):
            p.send(['step', a])
        result = [p.recv() for p in self.parent_remote]
        self._obs, self._rew, self._done, self._info = zip(*result)
        self._obs = np.stack(self._obs)
        self._rew = np.stack(self._rew)
        self._done = np.stack(self._done)
        self._info = np.stack(self._info)
        return self._obs, self._rew, self._done, self._info

    def reset(self, id=None):
        if id is None:
            for p in self.parent_remote:
                p.send(['reset', None])
            self._obs = np.stack([p.recv() for p in self.parent_remote])
            return self._obs
        else:
            if np.isscalar(id):
                id = [id]
            for i in id:
                self.parent_remote[i].send(['reset', None])
            for i in id:
                self._obs[i] = self.parent_remote[i].recv()
            return self._obs

    def seed(self, seed=None):
        if np.isscalar(seed):
            seed = [seed + _ for _ in range(self.env_num)]
        elif seed is None:
            seed = [seed] * self.env_num
        for p, s in zip(self.parent_remote, seed):
            p.send(['seed', s])
        return [p.recv() for p in self.parent_remote]

    def render(self, **kwargs):
        for p in self.parent_remote:
            p.send(['render', kwargs])
        return [p.recv() for p in self.parent_remote]

    def close(self):
        if self.closed:
            return
        for p in self.parent_remote:
            p.send(['close', None])
        self.closed = True
        for p in self.processes:
            p.join()


class RayVectorEnv(BaseVectorEnv):
    """docstring for RayVectorEnv"""

    def __init__(self, env_fns):
        super().__init__(env_fns)
        try:
            if not ray.is_initialized():
                ray.init()
        except NameError:
            raise ImportError(
                'Please install ray to support RayVectorEnv: pip3 install ray')
        self.envs = [
            ray.remote(EnvWrapper).options(num_cpus=0).remote(e())
            for e in env_fns]

    def step(self, action):
        assert len(action) == self.env_num
        result_obj = [e.step.remote(a) for e, a in zip(self.envs, action)]
        result = [ray.get(r) for r in result_obj]
        self._obs, self._rew, self._done, self._info = zip(*result)
        self._obs = np.stack(self._obs)
        self._rew = np.stack(self._rew)
        self._done = np.stack(self._done)
        self._info = np.stack(self._info)
        return self._obs, self._rew, self._done, self._info

    def reset(self, id=None):
        if id is None:
            result_obj = [e.reset.remote() for e in self.envs]
            self._obs = np.stack([ray.get(r) for r in result_obj])
        else:
            result_obj = []
            if np.isscalar(id):
                id = [id]
            for i in id:
                result_obj.append(self.envs[i].reset.remote())
            for _, i in enumerate(id):
                self._obs[i] = ray.get(result_obj[_])
        return self._obs

    def seed(self, seed=None):
        if not hasattr(self.envs[0], 'seed'):
            return
        if np.isscalar(seed):
            seed = [seed + _ for _ in range(self.env_num)]
        elif seed is None:
            seed = [seed] * self.env_num
        result_obj = [e.seed.remote(s) for e, s in zip(self.envs, seed)]
        return [ray.get(r) for r in result_obj]

    def render(self, **kwargs):
        if not hasattr(self.envs[0], 'render'):
            return
        result_obj = [e.render.remote(**kwargs) for e in self.envs]
        return [ray.get(r) for r in result_obj]

    def close(self):
        result_obj = [e.close.remote() for e in self.envs]
        for r in result_obj:
            ray.get(r)
